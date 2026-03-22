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
    FAKE_OOD_SEED,
    TinyImageNetDataset,
    apply_corruption,
    build_fake_ood_datasets,
    download_tiny_imagenet,
    generate_fake_ood_specs,
    get_ood_loaders,
    get_val_transform,
)
from models import create_ensemble_member, load_saved_member_state


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
            unfreeze_blocks=cfg.get("unfreeze_blocks", 0),
            pretrained=True,
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        load_saved_member_state(model, ckpt, strict=False)
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
    parser.add_argument("--p2_data_mode", type=str, default="fake_ood", choices=["ood", "fake_ood"],
                        help="Phase 2 data: 'ood'=real OOD fallback, 'fake_ood'=mixup+masked only")
    parser.add_argument("--fake_ood_mixup_frac", type=float, default=0.5,
                        help="Fraction of fake OOD allocated to mixup; remainder uses masking")
    parser.add_argument("--fake_ood_patchshuffle_frac", type=float, default=0.0,
                        help="Fraction of fake OOD allocated to patch-shuffled samples")
    parser.add_argument("--fake_ood_cutpaste_frac", type=float, default=0.0,
                        help="Fraction of fake OOD allocated to cut-paste samples")
    parser.add_argument("--fake_ood_heavy_noise_frac", type=float, default=0.0,
                        help="Fraction of fake OOD allocated to heavy Gaussian noise (sigma 0.3-1.0)")
    parser.add_argument("--fake_ood_multi_corrupt_frac", type=float, default=0.0,
                        help="Fraction of fake OOD allocated to multi-corruption stacking")
    parser.add_argument("--fake_ood_pixel_permute_frac", type=float, default=0.0,
                        help="Fraction of fake OOD allocated to pixel permutation within ViT patches")
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
    results["p2_data_mode"] = np.array(args.p2_data_mode)

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

    if args.p2_data_mode == "fake_ood":
        print("Caching: fake OOD datasets ...")
        n_fake = len(train_ds) // 2
        n_mixup = int(n_fake * args.fake_ood_mixup_frac)
        n_patchshuffle = int(n_fake * args.fake_ood_patchshuffle_frac)
        n_cutpaste = int(n_fake * args.fake_ood_cutpaste_frac)
        n_heavy_noise = int(n_fake * args.fake_ood_heavy_noise_frac)
        n_multi_corrupt = int(n_fake * args.fake_ood_multi_corrupt_frac)
        n_pixel_permute = int(n_fake * args.fake_ood_pixel_permute_frac)
        n_masked = n_fake - n_mixup - n_patchshuffle - n_cutpaste - n_heavy_noise - n_multi_corrupt - n_pixel_permute
        if n_masked < 0:
            raise ValueError("Fake OOD fractions must sum to at most 1.0")

        specs = generate_fake_ood_specs(
            len(train_ds),
            n_mixup,
            n_masked,
            n_patchshuffle=n_patchshuffle,
            n_cutpaste=n_cutpaste,
            n_heavy_noise=n_heavy_noise,
            n_multi_corrupt=n_multi_corrupt,
            n_pixel_permute=n_pixel_permute,
            seed=FAKE_OOD_SEED,
        )
        fake_datasets = build_fake_ood_datasets(train_ds, specs)

        all_fracs = (args.fake_ood_mixup_frac + args.fake_ood_patchshuffle_frac +
                     args.fake_ood_cutpaste_frac + args.fake_ood_heavy_noise_frac +
                     args.fake_ood_multi_corrupt_frac + args.fake_ood_pixel_permute_frac)
        results["fake_ood_mixup_frac"] = np.array(args.fake_ood_mixup_frac)
        results["fake_ood_patchshuffle_frac"] = np.array(args.fake_ood_patchshuffle_frac)
        results["fake_ood_cutpaste_frac"] = np.array(args.fake_ood_cutpaste_frac)
        results["fake_ood_heavy_noise_frac"] = np.array(args.fake_ood_heavy_noise_frac)
        results["fake_ood_multi_corrupt_frac"] = np.array(args.fake_ood_multi_corrupt_frac)
        results["fake_ood_pixel_permute_frac"] = np.array(args.fake_ood_pixel_permute_frac)
        results["fake_ood_masked_frac"] = np.array(max(0.0, 1.0 - all_fracs))
        results.update(specs)

        fake_family_meta = [
            ("mixup", "fake_mixup_EU", "fake mixup"),
            ("patchshuffle", "fake_patchshuffle_EU", "fake patchshuffle"),
            ("cutpaste", "fake_cutpaste_EU", "fake cutpaste"),
            ("masked", "fake_masked_EU", "fake masked"),
            ("heavy_noise", "fake_heavy_noise_EU", "fake heavy_noise"),
            ("multi_corrupt", "fake_multi_corrupt_EU", "fake multi_corrupt"),
            ("pixel_permute", "fake_pixel_permute_EU", "fake pixel_permute"),
        ]
        for family_name, eu_key, label in fake_family_meta:
            if family_name not in fake_datasets:
                continue
            loader = DataLoader(
                fake_datasets[family_name],
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            r = cache_from_loader(members, loader, device)
            results[eu_key] = r["EU"]
            print(f"  {label}: {len(r['EU'])} samples, EU mean={r['EU'].mean():.6f}")

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
