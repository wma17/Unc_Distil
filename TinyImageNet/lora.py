"""
Low-Rank Adaptation (LoRA) utilities for Vision Transformers.

Provides:
    - LoRALinear      — drop-in wrapper that adds trainable low-rank A/B matrices
    - apply_lora      — inject LoRA into selected layers of a timm ViT model
    - lora_state_dict / load_lora_state_dict — save/load only LoRA + head weights
    - remove_lora     — strip LoRA wrappers (for clean student init)
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wrap an existing ``nn.Linear`` with a low-rank adapter.

    Forward:  ``original(x) + B(A(dropout(x))) * scaling``
    where ``scaling = alpha / rank``.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 8.0,
        lora_dropout: float = 0.0,
    ):
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

    @property
    def trainable_parameters(self):
        yield from self.lora_A.parameters()
        yield from self.lora_B.parameters()


# ── Target-layer specification ────────────────────────────────────────────

VALID_TARGETS = {"qkv", "proj", "mlp_fc1", "mlp_fc2"}


def _resolve_targets(target_str: str) -> Set[str]:
    """Parse a target specification string into a set of layer keys.

    Examples:
        "qkv_only"     → {"qkv"}
        "qkv+proj"     → {"qkv", "proj"}
        "qkv+proj+mlp" → {"qkv", "proj", "mlp_fc1", "mlp_fc2"}
    """
    t = target_str.lower().replace(" ", "")
    if t == "qkv_only":
        return {"qkv"}
    if t == "qkv+proj":
        return {"qkv", "proj"}
    if t in ("qkv+proj+mlp", "all"):
        return {"qkv", "proj", "mlp_fc1", "mlp_fc2"}
    parts = set(t.split("+"))
    if not parts.issubset(VALID_TARGETS):
        raise ValueError(f"Unknown LoRA targets: {parts - VALID_TARGETS}")
    return parts


# ── Apply / remove LoRA ──────────────────────────────────────────────────

def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    lora_dropout: float = 0.0,
    targets: str = "qkv+proj",
) -> List[nn.Parameter]:
    """Inject LoRA adapters into a timm ViT and return trainable params.

    After calling this the *original* linear weights are frozen; only the
    LoRA A/B matrices (and any unfrozen head) are trainable.
    """
    target_set = _resolve_targets(targets)
    lora_params: List[nn.Parameter] = []

    for block in model.blocks:
        attn = block.attn
        if "qkv" in target_set and isinstance(attn.qkv, nn.Linear):
            wrapped = LoRALinear(attn.qkv, rank, alpha, lora_dropout)
            attn.qkv = wrapped
            lora_params.extend(wrapped.trainable_parameters)
        if "proj" in target_set and isinstance(attn.proj, nn.Linear):
            wrapped = LoRALinear(attn.proj, rank, alpha, lora_dropout)
            attn.proj = wrapped
            lora_params.extend(wrapped.trainable_parameters)

        mlp = block.mlp
        if "mlp_fc1" in target_set and isinstance(mlp.fc1, nn.Linear):
            wrapped = LoRALinear(mlp.fc1, rank, alpha, lora_dropout)
            mlp.fc1 = wrapped
            lora_params.extend(wrapped.trainable_parameters)
        if "mlp_fc2" in target_set and isinstance(mlp.fc2, nn.Linear):
            wrapped = LoRALinear(mlp.fc2, rank, alpha, lora_dropout)
            mlp.fc2 = wrapped
            lora_params.extend(wrapped.trainable_parameters)

    return lora_params


def _is_lora_key(key: str) -> bool:
    return "lora_A" in key or "lora_B" in key


def lora_state_dict(model: nn.Module, include_head: bool = True) -> OrderedDict:
    """Return only LoRA adapter weights (and optionally the classification head)."""
    sd = model.state_dict()
    keep = OrderedDict()
    for k, v in sd.items():
        if _is_lora_key(k):
            keep[k] = v
        elif include_head and k.startswith("head."):
            keep[k] = v
    return keep


def load_lora_state_dict(
    model: nn.Module,
    state: Dict[str, torch.Tensor],
    strict: bool = False,
):
    """Load LoRA (+ head) weights into a model that already has LoRA wrappers."""
    model.load_state_dict(state, strict=strict)


def remove_lora(model: nn.Module) -> None:
    """Replace every ``LoRALinear`` with its frozen original ``nn.Linear``."""
    for block in model.blocks:
        attn = block.attn
        if isinstance(attn.qkv, LoRALinear):
            attn.qkv = attn.qkv.original
        if isinstance(attn.proj, LoRALinear):
            attn.proj = attn.proj.original
        mlp = block.mlp
        if isinstance(mlp.fc1, LoRALinear):
            mlp.fc1 = mlp.fc1.original
        if isinstance(mlp.fc2, LoRALinear):
            mlp.fc2 = mlp.fc2.original
