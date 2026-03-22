"""
Run the trained BERT ensemble over SST-2 train/dev and cache teacher soft
labels and epistemic uncertainty (EU) per sentence.

Also caches EU for perturbed train sentences (Phase 2 Tier 2).

Usage:
    python cache_ensemble_targets.py --save_dir ./checkpoints --gpu 0
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import create_teacher
from data import (load_sst2, SST2Dataset, collate_fn,
                  apply_char_perturbations, apply_word_perturbations,
                  MAX_SEQ_LEN)


EPS = 1e-8


def load_ensemble(save_dir, device, num_classes=2):
    """Load all saved ensemble members."""
    members = []
    config_path = os.path.join(save_dir, "ensemble_configs.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            configs = json.load(f)
    else:
        configs = None

    for idx in range(20):
        path = os.path.join(save_dir, f"member_{idx}.pt")
        if not os.path.exists(path):
            break
        ckpt = torch.load(path, map_location=device, weights_only=False)
        mcfg = ckpt.get("member_config", {})
        model = create_teacher(
            num_classes=num_classes,
            rank=mcfg.get("rank", 8),
            alpha=mcfg.get("alpha", 16.0),
            attention_dropout=mcfg.get("attention_dropout", 0.1),
            init_scale=1.0,  # don't re-apply init scale when loading
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        print(f"  Loaded member_{idx}  dev_acc={ckpt.get('dev_acc', '?')}%")
        members.append(model)

    print(f"Ensemble size: {len(members)}")
    return members


def entropy(probs):
    return -(probs * torch.log(probs + EPS)).sum(dim=-1)


@torch.no_grad()
def compute_teacher_targets(members, loader, device):
    """Compute mean probs, TU, AU, EU for each sample."""
    all_member_probs = [[] for _ in members]
    all_labels = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if "label" in batch:
            all_labels.append(batch["label"])

        for m, model in enumerate(members):
            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
            all_member_probs[m].append(probs.cpu())

    # (M, N, C) -> (N, M, C)
    all_probs = torch.stack([torch.cat(mp, dim=0) for mp in all_member_probs], dim=0)
    all_probs = all_probs.permute(1, 0, 2)

    mean_probs = all_probs.mean(dim=1)
    tu = entropy(mean_probs)
    au = entropy(all_probs).mean(dim=1)
    eu = tu - au

    labels = torch.cat(all_labels, dim=0).numpy() if all_labels else None
    return mean_probs.numpy(), tu.numpy(), au.numpy(), eu.numpy(), labels


def main():
    parser = argparse.ArgumentParser(description="Cache ensemble teacher targets for SST-2")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Use only first K ensemble members (Ablation C)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading ensemble...")
    members = load_ensemble(args.save_dir, device)
    if args.subset_size is not None:
        members = members[:args.subset_size]
        print(f"Using subset of K={args.subset_size} members")

    train_ds, dev_ds, tokenizer = load_sst2("bert-base-uncased")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    results = {}

    # === 1. Clean SST-2 train ===
    print("\n--- Clean SST-2 train ---")
    train_probs, train_tu, train_au, train_eu, train_labels = compute_teacher_targets(
        members, train_loader, device)
    results["train_probs"] = train_probs
    results["train_tu"] = train_tu
    results["train_au"] = train_au
    results["train_eu"] = train_eu
    results["train_labels"] = train_labels
    print(f"  TU={train_tu.mean():.4f}  AU={train_au.mean():.4f}  EU={train_eu.mean():.4f}")

    # === 2. Clean SST-2 dev ===
    print("\n--- Clean SST-2 dev ---")
    dev_probs, dev_tu, dev_au, dev_eu, dev_labels = compute_teacher_targets(
        members, dev_loader, device)
    results["dev_probs"] = dev_probs
    results["dev_tu"] = dev_tu
    results["dev_au"] = dev_au
    results["dev_eu"] = dev_eu
    results["dev_labels"] = dev_labels
    print(f"  TU={dev_tu.mean():.4f}  AU={dev_au.mean():.4f}  EU={dev_eu.mean():.4f}")

    # === 3. Perturbed SST-2 train (character-level) ===
    print("\n--- Character-perturbed SST-2 train ---")
    from datasets import load_dataset
    raw_ds = load_dataset("glue", "sst2")
    train_texts = raw_ds["train"]["sentence"]

    char_perturbed_texts = [apply_char_perturbations(t, seed=2026 + i)
                            for i, t in enumerate(train_texts)]
    char_ds = SST2Dataset(char_perturbed_texts,
                           raw_ds["train"]["label"], tokenizer)
    char_loader = DataLoader(char_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)
    _, _, _, char_eu, _ = compute_teacher_targets(members, char_loader, device)
    results["char_perturbed_eu"] = char_eu
    print(f"  EU={char_eu.mean():.4f}")

    # === 4. Perturbed SST-2 train (word-level) ===
    print("\n--- Word-perturbed SST-2 train ---")
    word_perturbed_texts = [apply_word_perturbations(t, seed=3026 + i)
                            for i, t in enumerate(train_texts)]
    word_ds = SST2Dataset(word_perturbed_texts,
                           raw_ds["train"]["label"], tokenizer)
    word_loader = DataLoader(word_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)
    _, _, _, word_eu, _ = compute_teacher_targets(members, word_loader, device)
    results["word_perturbed_eu"] = word_eu
    print(f"  EU={word_eu.mean():.4f}")

    # === 5. Perturbed SST-2 dev (for evaluation) ===
    print("\n--- Character-perturbed SST-2 dev ---")
    dev_texts = raw_ds["validation"]["sentence"]
    char_dev_texts = [apply_char_perturbations(t, seed=4026 + i)
                      for i, t in enumerate(dev_texts)]
    char_dev_ds = SST2Dataset(char_dev_texts,
                               raw_ds["validation"]["label"], tokenizer)
    char_dev_loader = DataLoader(char_dev_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,
                                 collate_fn=collate_fn)
    _, _, _, char_dev_eu, _ = compute_teacher_targets(members, char_dev_loader, device)
    results["char_perturbed_dev_eu"] = char_dev_eu
    print(f"  EU={char_dev_eu.mean():.4f}")

    # === 6. Token-masked SST-2 train (for Tier 3 fake OOD targets) ===
    print("\n--- Token-masked SST-2 train (Tier 3 fake OOD) ---")
    mask_rates = [0.3, 0.5, 0.7]
    for rate in mask_rates:
        from data import TokenMaskedSST2Dataset
        mask_ds = TokenMaskedSST2Dataset(
            train_texts, train_eu, tokenizer,
            mask_rate=rate, seed=2027)
        # We need to compute teacher EU on the masked inputs
        mask_loader = DataLoader(mask_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=collate_fn)
        # Compute teacher targets on masked text
        all_member_probs_masked = [[] for _ in members]
        with torch.no_grad():
            for batch in mask_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                for m, model in enumerate(members):
                    logits = model(input_ids, attention_mask)
                    probs = F.softmax(logits, dim=-1)
                    all_member_probs_masked[m].append(probs.cpu())

        all_probs_m = torch.stack(
            [torch.cat(mp, 0) for mp in all_member_probs_masked], dim=0)
        all_probs_m = all_probs_m.permute(1, 0, 2)
        mean_p = all_probs_m.mean(dim=1)

        def _entropy(p):
            return -(p * torch.log(p + EPS)).sum(-1)

        tu_m = _entropy(mean_p)
        au_m = _entropy(all_probs_m).mean(dim=1)
        eu_m = (tu_m - au_m).numpy()
        results[f"masked_{rate}_eu"] = eu_m
        print(f"  mask_rate={rate}: EU={eu_m.mean():.4f}")

    # === Save ===
    out_path = os.path.join(args.save_dir, "teacher_targets.npz")
    np.savez(out_path, **results)
    print(f"\nSaved all targets to {out_path}")
    print(f"Keys: {sorted(results.keys())}")


if __name__ == "__main__":
    main()
