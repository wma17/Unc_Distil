"""
SegFormer-B2 teacher (LoRA) + student with spatial EU head for Pascal VOC 2012.

Teacher: SegFormer-B2 pretrained on ImageNet-22K with LoRA on attention Q/K/V/Out.
Student: SegFormer-B2 pretrained backbone + decode head + per-pixel EU head.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig


NUM_CLASSES = 21  # 20 foreground + 1 background


# ── LoRA ─────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Low-rank adapter wrapping an existing nn.Linear."""

    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0,
                 lora_dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank

        in_f, out_f = original.in_features, original.out_features
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.lora_drop = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora = self.lora_B(self.lora_A(self.lora_drop(x))) * self.scaling
        return base + lora


def _is_lora_key(key: str) -> bool:
    return "lora_A" in key or "lora_B" in key


# ── Teacher (SegFormer-B2 + LoRA) ────────────────────────────────────────

class SegFormerTeacher(nn.Module):
    """SegFormer-B2 with LoRA adapters on all attention projections.

    The MLP layers and decode head are fully trainable.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, rank: int = 16,
                 alpha: float = 32.0, lora_dropout: float = 0.0,
                 init_scale: float = 1.0):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        # Freeze encoder
        for p in self.segformer.segformer.encoder.parameters():
            p.requires_grad = False

        # Apply LoRA to attention Q, K, V, Out projections
        self.lora_params = []
        for layer_idx, layer in enumerate(self.segformer.segformer.encoder.block):
            for block in layer:
                attn = block.attention.self
                for attr in ["query", "key", "value"]:
                    orig = getattr(attn, attr)
                    if isinstance(orig, nn.Linear):
                        wrapped = LoRALinear(orig, rank=rank, alpha=alpha,
                                            lora_dropout=lora_dropout)
                        setattr(attn, attr, wrapped)
                        self.lora_params.extend([wrapped.lora_A.weight, wrapped.lora_B.weight])
                # Output projection
                out_proj = block.attention.output.dense
                if isinstance(out_proj, nn.Linear):
                    wrapped = LoRALinear(out_proj, rank=rank, alpha=alpha,
                                        lora_dropout=lora_dropout)
                    block.attention.output.dense = wrapped
                    self.lora_params.extend([wrapped.lora_A.weight, wrapped.lora_B.weight])

        # Decode head is fully trainable
        for p in self.segformer.decode_head.parameters():
            p.requires_grad = True

        # Apply init scale to LoRA params
        if init_scale != 1.0:
            with torch.no_grad():
                for p in self.lora_params:
                    p.mul_(init_scale)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  SegFormer Teacher: {trainable:,} trainable / {total:,} total (LoRA r={rank})")

    def forward(self, pixel_values):
        outputs = self.segformer(pixel_values=pixel_values)
        # outputs.logits: (B, num_classes, H/4, W/4)
        return outputs.logits

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self) -> OrderedDict:
        keep = OrderedDict()
        trainable_keys = {name for name, p in self.named_parameters() if p.requires_grad}
        for key, value in self.state_dict().items():
            if key in trainable_keys or _is_lora_key(key):
                keep[key] = value
        return keep


# ── Student (SegFormer-B2 + spatial EU head) ──────────────────────────────

class SegFormerStudent(nn.Module):
    """SegFormer-B2 student with decode head + per-pixel EU head.

    EU head operates on the last encoder stage features (before decode head).
    Architecture:
        Conv2d(512 -> 128, 1x1) -> BN -> ReLU -> Conv2d(128 -> 1, 1x1) -> Softplus
        Output upsampled to input resolution.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        # EU head on last encoder stage features (512-d for B2)
        self.eu_conv1 = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.eu_bn = nn.BatchNorm2d(128)
        self.eu_conv2 = nn.Conv2d(128, 1, kernel_size=1)
        self.eu_act = nn.Softplus()

    def forward(self, pixel_values, return_eu=True):
        # Get encoder hidden states
        encoder_outputs = self.segformer.segformer.encoder(
            pixel_values, output_hidden_states=True, return_dict=True)
        hidden_states = encoder_outputs.hidden_states

        # Decode head for segmentation logits
        logits = self.segformer.decode_head(hidden_states)
        # logits: (B, num_classes, H/4, W/4)

        if not return_eu:
            return logits, None

        # EU head on last stage features
        last_feat = hidden_states[-1]  # (B, 512, H/32, W/32)
        eu = self.eu_act(self.eu_conv2(F.relu(self.eu_bn(self.eu_conv1(last_feat)))))
        # eu: (B, 1, H/32, W/32) -> upsample to match logits
        eu = F.interpolate(eu, size=logits.shape[2:], mode="bilinear", align_corners=False)
        eu = eu.squeeze(1)  # (B, H/4, W/4)

        return logits, eu

    @property
    def eu_head_parameters(self):
        yield from self.eu_conv1.parameters()
        yield from self.eu_bn.parameters()
        yield from self.eu_conv2.parameters()

    def reinit_eu_head(self):
        nn.init.kaiming_normal_(self.eu_conv1.weight)
        self.eu_bn.reset_parameters()
        nn.init.kaiming_normal_(self.eu_conv2.weight)
        nn.init.zeros_(self.eu_conv2.bias)


def create_teacher(num_classes=NUM_CLASSES, rank=16, alpha=32.0,
                   lora_dropout=0.0, init_scale=1.0):
    return SegFormerTeacher(num_classes=num_classes, rank=rank, alpha=alpha,
                            lora_dropout=lora_dropout, init_scale=init_scale)


def create_student(num_classes=NUM_CLASSES):
    return SegFormerStudent(num_classes=num_classes)
