"""
Data handling for SST-2 sentiment classification pipeline.

Provides:
    - SST-2 train/dev loaders (HuggingFace datasets)
    - Character-level + word-level perturbations for Phase 2 Tier 2
    - OOD dataset loaders: IMDB, Yelp Polarity, Amazon Polarity, AG News, 20 Newsgroups
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


MAX_SEQ_LEN = 128


# ═══════════════════════════════════════════════════════════════════════════
#  SST-2 Dataset
# ═══════════════════════════════════════════════════════════════════════════

class SST2Dataset(Dataset):
    """SST-2 dataset with tokenization."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer,
                 max_len: int = MAX_SEQ_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_sst2(tokenizer_name: str = "bert-base-uncased"):
    """Load SST-2 from HuggingFace datasets library."""
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_texts = ds["train"]["sentence"]
    train_labels = ds["train"]["label"]
    # Dev as test (test labels not public)
    dev_texts = ds["validation"]["sentence"]
    dev_labels = ds["validation"]["label"]

    train_ds = SST2Dataset(train_texts, train_labels, tokenizer)
    dev_ds = SST2Dataset(dev_texts, dev_labels, tokenizer)
    return train_ds, dev_ds, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
#  Text Perturbations (Phase 2 Tier 2)
# ═══════════════════════════════════════════════════════════════════════════

KEYBOARD_ADJACENT = {
    'a': 'sqwz', 'b': 'vngh', 'c': 'xdfv', 'd': 'sfce', 'e': 'wdrs',
    'f': 'dgcv', 'g': 'fhtb', 'h': 'gjyn', 'i': 'ujko', 'j': 'hkun',
    'k': 'jlim', 'l': 'kop', 'm': 'nkj', 'n': 'bmhj', 'o': 'iklp',
    'p': 'ol', 'q': 'wa', 'r': 'etfd', 's': 'adwx', 't': 'rfgy',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
    'z': 'xsa',
}


def char_swap(text: str, rate: float = 0.02, rng: random.Random = None) -> str:
    """Randomly swap adjacent characters."""
    if rng is None:
        rng = random.Random()
    chars = list(text)
    for i in range(len(chars) - 1):
        if rng.random() < rate:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def char_insert(text: str, rate: float = 0.02, rng: random.Random = None) -> str:
    """Randomly insert a character next to each character."""
    if rng is None:
        rng = random.Random()
    result = []
    for c in text:
        result.append(c)
        if rng.random() < rate and c.lower() in KEYBOARD_ADJACENT:
            result.append(rng.choice(KEYBOARD_ADJACENT[c.lower()]))
    return "".join(result)


def keyboard_sub(text: str, rate: float = 0.02, rng: random.Random = None) -> str:
    """Replace characters with keyboard-adjacent ones."""
    if rng is None:
        rng = random.Random()
    chars = list(text)
    for i, c in enumerate(chars):
        if rng.random() < rate and c.lower() in KEYBOARD_ADJACENT:
            adj = KEYBOARD_ADJACENT[c.lower()]
            replacement = rng.choice(adj)
            chars[i] = replacement.upper() if c.isupper() else replacement
    return "".join(chars)


def synonym_substitution(text: str, rate: float = 0.15, rng: random.Random = None) -> str:
    """Replace words with synonyms via WordNet. Falls back to identity if nltk unavailable."""
    if rng is None:
        rng = random.Random()
    try:
        from nltk.corpus import wordnet
        import nltk
        try:
            wordnet.synsets("test")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
    except ImportError:
        return text

    words = text.split()
    result = []
    for w in words:
        if rng.random() < rate:
            syns = wordnet.synsets(w.lower())
            synonyms = set()
            for s in syns:
                for lemma in s.lemmas():
                    name = lemma.name().replace("_", " ")
                    if name.lower() != w.lower():
                        synonyms.add(name)
            if synonyms:
                replacement = rng.choice(list(synonyms))
                result.append(replacement)
                continue
        result.append(w)
    return " ".join(result)


def apply_char_perturbations(text: str, seed: int = 0) -> str:
    """Apply all character-level perturbations."""
    rng = random.Random(seed)
    text = char_swap(text, rate=0.02, rng=rng)
    text = char_insert(text, rate=0.02, rng=rng)
    text = keyboard_sub(text, rate=0.02, rng=rng)
    return text


def apply_word_perturbations(text: str, seed: int = 0) -> str:
    """Apply word-level perturbations (synonym substitution)."""
    rng = random.Random(seed)
    return synonym_substitution(text, rate=0.15, rng=rng)


class PerturbedSST2Dataset(Dataset):
    """SST-2 with character and word perturbations applied on-the-fly."""

    def __init__(self, texts: List[str], eu_targets: np.ndarray, tokenizer,
                 perturbation: str = "char", max_len: int = MAX_SEQ_LEN,
                 seed: int = 2026):
        self.texts = texts
        self.eu_targets = torch.from_numpy(eu_targets).float()
        self.tokenizer = tokenizer
        self.perturbation = perturbation
        self.max_len = max_len
        self.seed = seed

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.perturbation == "char":
            text = apply_char_perturbations(self.texts[idx], seed=self.seed + idx)
        elif self.perturbation == "word":
            text = apply_word_perturbations(self.texts[idx], seed=self.seed + idx)
        else:
            text = self.texts[idx]

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "eu_target": self.eu_targets[idx],
        }


class TokenMaskedSST2Dataset(Dataset):
    """SST-2 with random token masking for synthetic OOD (Tier 3)."""

    def __init__(self, texts: List[str], eu_targets: np.ndarray, tokenizer,
                 mask_rate: float = 0.3, max_len: int = MAX_SEQ_LEN,
                 seed: int = 2027):
        self.texts = texts
        self.eu_targets = torch.from_numpy(eu_targets).float()
        self.tokenizer = tokenizer
        self.mask_rate = mask_rate
        self.max_len = max_len
        self.seed = seed
        self.mask_token_id = tokenizer.mask_token_id or tokenizer.unk_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        rng = torch.Generator().manual_seed(self.seed + idx)
        # Mask non-special tokens
        special = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                    self.tokenizer.pad_token_id}
        mask_probs = torch.rand(input_ids.shape, generator=rng)
        for i, tid in enumerate(input_ids.tolist()):
            if tid in special:
                mask_probs[i] = 1.0  # don't mask
        mask = mask_probs < self.mask_rate
        input_ids[mask] = self.mask_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "eu_target": self.eu_targets[idx],
        }


# ═══════════════════════════════════════════════════════════════════════════
#  EU-only Dataset (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════

class EUTextDataset(Dataset):
    """Text dataset with EU targets for Phase 2."""

    def __init__(self, texts: List[str], eu_targets: np.ndarray, tokenizer,
                 max_len: int = MAX_SEQ_LEN):
        self.texts = texts
        self.eu_targets = torch.from_numpy(eu_targets).float()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "eu_target": self.eu_targets[idx],
        }


# ═══════════════════════════════════════════════════════════════════════════
#  OOD Datasets
# ═══════════════════════════════════════════════════════════════════════════

class GenericTextDataset(Dataset):
    """Generic text classification dataset."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer,
                 max_len: int = MAX_SEQ_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_ood_datasets(tokenizer, max_samples: int = 5000) -> Dict[str, Dataset]:
    """Load OOD datasets for evaluation. Missing datasets are silently skipped."""
    from datasets import load_dataset
    ood = {}

    # IMDB (near-OOD)
    try:
        ds = load_dataset("imdb", split="test")
        texts = ds["text"][:max_samples]
        labels = ds["label"][:max_samples]
        ood["IMDB"] = GenericTextDataset(texts, labels, tokenizer)
        print(f"  IMDB: {len(texts)} samples")
    except Exception as e:
        print(f"  [skip] IMDB: {e}")

    # Yelp Polarity (near-OOD)
    try:
        ds = load_dataset("yelp_polarity", split="test")
        texts = ds["text"][:max_samples]
        labels = ds["label"][:max_samples]
        ood["Yelp"] = GenericTextDataset(texts, labels, tokenizer)
        print(f"  Yelp: {len(texts)} samples")
    except Exception as e:
        print(f"  [skip] Yelp: {e}")

    # Amazon Polarity (near-OOD)
    try:
        ds = load_dataset("amazon_polarity", split="test")
        texts = [f"{t} {c}" for t, c in zip(ds["title"][:max_samples],
                                              ds["content"][:max_samples])]
        labels = ds["label"][:max_samples]
        ood["Amazon"] = GenericTextDataset(texts, labels, tokenizer)
        print(f"  Amazon: {len(texts)} samples")
    except Exception as e:
        print(f"  [skip] Amazon: {e}")

    # AG News (far-OOD, 4 classes)
    try:
        ds = load_dataset("ag_news", split="test")
        texts = ds["text"][:max_samples]
        labels = ds["label"][:max_samples]
        ood["AG_News"] = GenericTextDataset(texts, labels, tokenizer)
        print(f"  AG News: {len(texts)} samples")
    except Exception as e:
        print(f"  [skip] AG News: {e}")

    # 20 Newsgroups (far-OOD, binary subset: 0-9 vs 10-19)
    try:
        from sklearn.datasets import fetch_20newsgroups
        ng = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
        texts = ng.data[:max_samples]
        labels = [0 if l < 10 else 1 for l in ng.target[:max_samples]]
        ood["20NG"] = GenericTextDataset(texts, labels, tokenizer)
        print(f"  20 Newsgroups: {len(texts)} samples")
    except Exception as e:
        print(f"  [skip] 20 Newsgroups: {e}")

    return ood


# ═══════════════════════════════════════════════════════════════════════════
#  Collate + DataLoader helpers
# ═══════════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    """Collate for datasets returning dicts."""
    result = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])
        else:
            result[key] = torch.tensor([b[key] for b in batch])
    return result


def get_dataloader(dataset, batch_size=32, shuffle=False, num_workers=2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
