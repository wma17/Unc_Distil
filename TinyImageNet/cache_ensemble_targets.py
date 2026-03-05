"""
Compute and cache teacher ensemble targets for distillation.

For every data source (clean ID, corrupted ID, OOD) we store:
    - mean probability vectors   (Bayesian averaged)
    - Total Uncertainty   TU = H[E_θ p(y|x,θ)]
    - Aleatoric Uncertainty AU = E_θ[H(y|x,θ)]
    - Epistemic Uncertainty EU = TU - AU = I[y;θ|x,D]

Usage:
    python cache_ensemble_targets.py --save_dir ./checkpoints --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset

from data import (
    CORRUPTIONS,
    TinyImageNetDataset,
    apply_corruption,
    download_tiny_imagenet,
    get_ood_loaders,
    get_val_transform,
)
from models import create_ensemble_member
from lora import load_lora_state_dict


def load_ensemble(save_dir: str, device: torch.device):
    """Load all ensemble members from checkpoints."""
    cfg_path = os.path.join(save_dir, "ensemble_configs.json")
    with open(cfg_path) as f:
        configs = json.load(f)

    members = []
    for cfg in configs:
        mid = cfg["member_id"]
        ckpt_path = os.path.join(save_dir, f"member_{mid}.pt")
        model = create_ensemble_member(
            rank=cfg["rank"],
            alpha=cfg["alpha"],
            lora_dropout=cfg["lora_dropout"],
            targets=cfg["targets"],
            pretrained=True,
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        load_lora_state_dict(model, ckpt["lora_head_state"], strict=False)
        model.to(device).eval()
        members.append((cfg, model))
        print(f"  Loaded member {mid} (rank={cfg['rank']}, targets={cfg['targets']})")

    return members


@torch.no_grad()
def ensemble_predict(members, loader, device):
    """Run all members on a dataloader and return per-member probabilities + labels."""
    all_probs_per_member = [[] for _ in members]
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        all_labels.append(labels.numpy())
        for m_idx, (_, model) in enumerate(members):
            with autocast("cuda"):
                logits = model(imgs)
            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
            all_probs_per_member[m_idx].append(probs)

    labels_arr = np.concatenate(all_labels)
    probs_arr = np.stack(
        [np.concatenate(pm) for pm in all_probs_per_member], axis=0
    )  # (M, N, C)
    return probs_arr, labels_arr


def compute_uncertainties(probs_arr: np.ndarray):
    """Compute TU, AU, EU from per-member probabilities.

    Args:
        probs_arr: shape (M, N, C)

    Returns:
        mean_probs (N, C), TU (N,), AU (N,), EU (N,)
    """
    eps = 1e-10
    mean_probs = probs_arr.mean(axis=0)  # (N, C)
    TU = -np.sum(mean_probs * np.log(mean_probs + eps), axis=-1)  # H[E[p]]
    H_per_member = -np.sum(probs_arr * np.log(probs_arr + eps), axis=-1)  # (M, N)
    AU = H_per_member.mean(axis=0)  # E[H[p]]
    EU = TU - AU
    return mean_probs, TU, AU, EU


def cache_from_loader(members, loader, device):
    probs_arr, labels = ensemble_predict(members, loader, device)
    mean_probs, TU, AU, EU = compute_uncertainties(probs_arr)
    return {
        "mean_probs": mean_probs,
        "labels": labels,
        "TU": TU,
        "AU": AU,
        "EU": EU,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    root = download_tiny_imagenet(args.data_dir)
    val_tf = get_val_transform()

    print("Loading ensemble members ...")
    members = load_ensemble(args.save_dir, device)
    M = len(members)
    print(f"  {M} members loaded.\n")

    results = {}

    # ── 1. Clean train set ────────────────────────────────────────────────
    print("Caching: clean train ...")
    train_ds = TinyImageNetDataset(root, split="train", transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    r = cache_from_loader(members, train_loader, device)
    results["train_mean_probs"] = r["mean_probs"]
    results["train_labels"] = r["labels"]
    results["train_TU"] = r["TU"]
    results["train_AU"] = r["AU"]
    results["train_EU"] = r["EU"]
    print(f"  train: {len(r['labels'])} samples, EU mean={r['EU'].mean():.6f}")

    # ── 2. Clean val set ──────────────────────────────────────────────────
    print("Caching: clean val ...")
    val_ds = TinyImageNetDataset(root, split="val", transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    r = cache_from_loader(members, val_loader, device)
    results["val_mean_probs"] = r["mean_probs"]
    results["val_labels"] = r["labels"]
    results["val_TU"] = r["TU"]
    results["val_AU"] = r["AU"]
    results["val_EU"] = r["EU"]
    print(f"  val: {len(r['labels'])} samples, EU mean={r['EU'].mean():.6f}")

    # ── 3. Corrupted val ──────────────────────────────────────────────────
    print("Caching: corrupted val ...")
    for cname, cfn in CORRUPTIONS.items():
        print(f"  corruption: {cname} ...")
        imgs_t, labels_t = apply_corruption(val_ds, cfn, max_samples=5000)
        c_ds = TensorDataset(imgs_t, labels_t)
        c_loader = DataLoader(c_ds, batch_size=args.batch_size, num_workers=2)
        r = cache_from_loader(members, c_loader, device)
        results[f"corrupt_{cname}_mean_probs"] = r["mean_probs"]
        results[f"corrupt_{cname}_labels"] = r["labels"]
        results[f"corrupt_{cname}_TU"] = r["TU"]
        results[f"corrupt_{cname}_AU"] = r["AU"]
        results[f"corrupt_{cname}_EU"] = r["EU"]
        print(f"    {cname}: {len(r['labels'])} samples, EU mean={r['EU'].mean():.6f}")

    # ── 4. OOD datasets ──────────────────────────────────────────────────
    print("Caching: OOD datasets ...")
    ood_loaders = get_ood_loaders(args.data_dir, batch_size=args.batch_size)
    for name, loader in ood_loaders.items():
        print(f"  OOD: {name} ...")
        r = cache_from_loader(members, loader, device)
        safe = name.replace("-", "_").replace(" ", "_")
        results[f"ood_{safe}_mean_probs"] = r["mean_probs"]
        results[f"ood_{safe}_labels"] = r["labels"]
        results[f"ood_{safe}_TU"] = r["TU"]
        results[f"ood_{safe}_AU"] = r["AU"]
        results[f"ood_{safe}_EU"] = r["EU"]
        print(f"    {name}: {len(r['labels'])} samples, EU mean={r['EU'].mean():.6f}")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(args.save_dir, "teacher_targets.npz")
    np.savez_compressed(out_path, **results)
    print(f"\nSaved teacher targets to {out_path}")
    print(f"Keys: {sorted(results.keys())}")


if __name__ == "__main__":
    main()
