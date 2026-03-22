"""
BERT-base teacher (LoRA) + DistilBERT student for SST-2 sentiment classification.

Teacher: BERT-base-uncased with LoRA adapters on Q,V projections.
Student: DistilBERT-base-uncased with classification head + scalar EU head.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, DistilBertModel, DistilBertConfig


# ── LoRA ─────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Low-rank adapter wrapping an existing nn.Linear."""

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0,
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


# ── Teacher (BERT-base + LoRA) ───────────────────────────────────────────

class BERTTeacher(nn.Module):
    """BERT-base-uncased with LoRA adapters on Q and V attention projections.

    The classification head (on [CLS]) is fully trainable.
    """

    def __init__(self, num_classes: int = 2, rank: int = 8, alpha: float = 16.0,
                 lora_dropout: float = 0.0, attention_dropout: float = 0.1,
                 init_scale: float = 1.0):
        super().__init__()
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.attention_probs_dropout_prob = attention_dropout
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        # Freeze all BERT params
        for p in self.bert.parameters():
            p.requires_grad = False

        # Apply LoRA to Q and V in all attention layers
        self.lora_params = []
        for layer in self.bert.encoder.layer:
            attn = layer.attention.self
            for attr in ["query", "value"]:
                orig = getattr(attn, attr)
                wrapped = LoRALinear(orig, rank=rank, alpha=alpha, lora_dropout=lora_dropout)
                setattr(attn, attr, wrapped)
                self.lora_params.extend([wrapped.lora_A.weight, wrapped.lora_B.weight])

        # Classifier is trainable
        for p in self.classifier.parameters():
            p.requires_grad = True

        # Apply init scale
        if init_scale != 1.0:
            with torch.no_grad():
                for p in self.lora_params:
                    if p.requires_grad:
                        p.mul_(init_scale)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  BERT Teacher: {trainable:,} trainable / {total:,} total (LoRA r={rank})")

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0]  # [CLS]
        logits = self.classifier(cls_hidden)
        return logits

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self) -> OrderedDict:
        keep = OrderedDict()
        trainable_keys = {name for name, p in self.named_parameters() if p.requires_grad}
        for key, value in self.state_dict().items():
            if key in trainable_keys or _is_lora_key(key):
                keep[key] = value
        return keep


# ── Student (DistilBERT + EU head) ───────────────────────────────────────

class DistilBERTStudent(nn.Module):
    """DistilBERT-base-uncased student with classification head + scalar EU head.

    EU head input: [CLS] hidden (768-d) || softmax(logits) (2-d) = 770-d.
    """

    HIDDEN_DIM = 768  # DistilBERT hidden size

    def __init__(self, num_classes: int = 2, eu_hidden: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.HIDDEN_DIM, num_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        eu_in = self.HIDDEN_DIM + num_classes
        self.eu_fc1 = nn.Linear(eu_in, eu_hidden)
        self.eu_fc2 = nn.Linear(eu_hidden, 1)
        self.eu_act = nn.Softplus()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0]  # [CLS]
        logits = self.classifier(cls_hidden)

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
        eu_in = torch.cat([cls_hidden.detach(), probs], dim=-1)
        eu = self.eu_act(self.eu_fc2(F.relu(self.eu_fc1(eu_in)))).squeeze(-1)
        return logits, eu

    def get_cls_hidden(self, input_ids, attention_mask=None):
        """Return [CLS] hidden state (for embedding-space mixup)."""
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]

    @property
    def eu_head_parameters(self):
        yield from self.eu_fc1.parameters()
        yield from self.eu_fc2.parameters()

    def reinit_eu_head(self):
        nn.init.kaiming_normal_(self.eu_fc1.weight)
        nn.init.zeros_(self.eu_fc1.bias)
        nn.init.kaiming_normal_(self.eu_fc2.weight)
        nn.init.zeros_(self.eu_fc2.bias)


def create_teacher(num_classes=2, rank=8, alpha=16.0, lora_dropout=0.0,
                   attention_dropout=0.1, init_scale=1.0):
    return BERTTeacher(num_classes=num_classes, rank=rank, alpha=alpha,
                       lora_dropout=lora_dropout, attention_dropout=attention_dropout,
                       init_scale=init_scale)


def create_student(num_classes=2, eu_hidden=128):
    return DistilBERTStudent(num_classes=num_classes, eu_hidden=eu_hidden)
