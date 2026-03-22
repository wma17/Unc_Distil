"""
DeiT-Small model definitions for LoRA ensemble and distilled student.

Ensemble member:
    Pretrained DeiT-Small (patch16, 224×224) with LoRA adapters injected
    into attention layers, a new 200-class head, and optional partial
    fine-tuning of the last transformer blocks.

Student:
    Same DeiT-Small backbone + classification head + scalar EU head.
    EU head input: CLS features (384-d) || softmax(logits) (200-d) = 584-d.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from lora import apply_lora, lora_state_dict

EMBED_DIM = 384
NUM_CLASSES = 200


def _base_deit(pretrained: bool = True) -> nn.Module:
    model = timm.create_model("deit_small_patch16_224", pretrained=pretrained)
    model.head = nn.Linear(EMBED_DIM, NUM_CLASSES)
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)
    return model


# ── Ensemble member ──────────────────────────────────────────────────────

def create_ensemble_member(
    rank: int = 8,
    alpha: Optional[float] = None,
    lora_dropout: float = 0.0,
    targets: str = "qkv+proj",
    unfreeze_blocks: int = 0,
    pretrained: bool = True,
) -> nn.Module:
    """Create a DeiT-Small with LoRA adapters for ensemble training.

    Most backbone weights stay frozen. LoRA adapters are always trainable,
    and the final `unfreeze_blocks` transformer blocks are fully fine-tuned.
    """
    model = _base_deit(pretrained=pretrained)

    for p in model.parameters():
        p.requires_grad = False

    if alpha is None:
        alpha = float(rank)
    apply_lora(model, rank=rank, alpha=alpha,
               lora_dropout=lora_dropout, targets=targets)

    for p in model.head.parameters():
        p.requires_grad = True
    if unfreeze_blocks > 0:
        n_blocks = len(model.blocks)
        for p in model.norm.parameters():
            p.requires_grad = True
        for i in range(max(0, n_blocks - unfreeze_blocks), n_blocks):
            for p in model.blocks[i].parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA member: {trainable:,} trainable / {total:,} total "
          f"(rank={rank}, targets={targets}, unfreeze_blocks={unfreeze_blocks})")
    return model


def member_state_dict(model: nn.Module) -> OrderedDict:
    """Persist only trainable member parameters; frozen pretrained weights are implicit."""
    keep = OrderedDict()
    trainable_keys = {name for name, p in model.named_parameters() if p.requires_grad}
    for key, value in model.state_dict().items():
        if key in trainable_keys:
            keep[key] = value
    return keep


def load_saved_member_state(model: nn.Module, checkpoint_or_state, strict: bool = False):
    """Load trainable member weights from a checkpoint or raw state dict."""
    if isinstance(checkpoint_or_state, dict) and "member_state" in checkpoint_or_state:
        state = checkpoint_or_state["member_state"]
    elif isinstance(checkpoint_or_state, dict) and "lora_head_state" in checkpoint_or_state:
        state = checkpoint_or_state["lora_head_state"]
    else:
        state = checkpoint_or_state
    return model.load_state_dict(state, strict=strict)


def save_member(model: nn.Module, path: str, extra: dict | None = None):
    data = {
        "member_state": member_state_dict(model),
        "lora_head_state": lora_state_dict(model, include_head=True),
    }
    if extra:
        data.update(extra)
    torch.save(data, path)


def load_member(
    path: str,
    rank: int = 8,
    alpha: Optional[float] = None,
    lora_dropout: float = 0.0,
    targets: str = "qkv+proj",
    unfreeze_blocks: int = 0,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Recreate a LoRA member and load saved adapter + head weights."""
    model = create_ensemble_member(
        rank=rank, alpha=alpha, lora_dropout=lora_dropout,
        targets=targets, unfreeze_blocks=unfreeze_blocks, pretrained=True,
    )
    ckpt = torch.load(path, map_location=device, weights_only=True)
    load_saved_member_state(model, ckpt, strict=False)
    model.to(device)
    model.eval()
    return model


# ── Student ──────────────────────────────────────────────────────────────

class DeiTStudent(nn.Module):
    """DeiT-Small student with classification head + scalar EU head.

    The EU head takes ``[CLS_features, softmax(logits)]`` as input.
    No output activation on EU — raw linear value, clamped >= 0 at inference.
    """

    FEAT_DIM = EMBED_DIM  # 384

    def __init__(self, num_classes: int = NUM_CLASSES, eu_hidden: int = 256,
                 drop_path_rate: float = 0.1):
        super().__init__()
        self.num_classes = num_classes

        backbone = timm.create_model("deit_small_patch16_224", pretrained=True,
                                     drop_path_rate=drop_path_rate)
        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.pos_embed = backbone.pos_embed
        self.pos_drop = backbone.pos_drop
        self.blocks = backbone.blocks
        self.norm = backbone.norm

        self.head = nn.Linear(self.FEAT_DIM, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        eu_in = self.FEAT_DIM + num_classes
        self.eu_fc1 = nn.Linear(eu_in, 512)
        self.eu_drop = nn.Dropout(0.15)
        self.eu_fc2 = nn.Linear(512, 128)
        self.eu_fc3 = nn.Linear(128, 1)
        self.eu_scale = nn.Parameter(torch.ones(1))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token

    def forward(self, x: torch.Tensor):
        feat = self._features(x)
        logits = self.head(feat)

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
        eu_in = torch.cat([feat.detach(), probs], dim=-1)
        eu = F.leaky_relu(self.eu_fc1(eu_in), 0.1)
        eu = self.eu_drop(eu)
        eu = F.leaky_relu(self.eu_fc2(eu), 0.1)
        eu = (self.eu_scale * self.eu_fc3(eu)).squeeze(-1)
        return logits, eu

    @property
    def eu_head_parameters(self):
        yield from self.eu_fc1.parameters()
        yield from self.eu_fc2.parameters()
        yield from self.eu_fc3.parameters()
        yield self.eu_scale

    def reinit_eu_head(self):
        nn.init.kaiming_normal_(self.eu_fc1.weight)
        nn.init.zeros_(self.eu_fc1.bias)
        nn.init.kaiming_normal_(self.eu_fc2.weight)
        nn.init.zeros_(self.eu_fc2.bias)
        nn.init.kaiming_normal_(self.eu_fc3.weight)
        nn.init.zeros_(self.eu_fc3.bias)
        self.eu_scale.data.fill_(1.0)


def create_student(num_classes: int = NUM_CLASSES, eu_hidden: int = 256,
                   drop_path_rate: float = 0.1):
    return DeiTStudent(num_classes=num_classes, eu_hidden=eu_hidden,
                       drop_path_rate=drop_path_rate)
