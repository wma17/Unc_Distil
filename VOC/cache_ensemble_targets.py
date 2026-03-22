"""
Cache per-pixel EU targets from the SegFormer ensemble for VOC 2012.

Stores per-pixel EU as float16 .npz shards (~5.5 GB for full train set).

Usage:
    python cache_ensemble_targets.py --save_dir ./checkpoints --gpu 0
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import create_teacher, NUM_CLASSES
from data import VOCSegDataset, SegTransformVal, IGNORE_INDEX


EPS = 1e-8
EU_MAP_SIZE = 128  # Downsample EU maps to this resolution for storage


def load_ensemble(save_dir, device):
    members = []
    for idx in range(20):
        path = os.path.join(save_dir, f"member_{idx}.pt")
        if not os.path.exists(path):
            break
        ckpt = torch.load(path, map_location=device, weights_only=False)
        mcfg = ckpt.get("member_config", {})
        model = create_teacher(
            num_classes=NUM_CLASSES,
            rank=mcfg.get("rank", 16),
            alpha=mcfg.get("alpha", 32.0),
            init_scale=1.0,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        print(f"  Loaded member_{idx}  mIoU={ckpt.get('val_miou', '?')}")
        members.append(model)
    print(f"Ensemble size: {len(members)}")
    return members


def entropy_per_pixel(probs):
    """Per-pixel entropy. probs: (B, C, H, W)"""
    return -(probs * torch.log(probs + EPS)).sum(dim=1)  # (B, H, W)


@torch.no_grad()
def compute_pixel_eu_batch(members, imgs, device, target_size=EU_MAP_SIZE):
    """Compute per-pixel EU for a batch. Returns EU map at target_size resolution."""
    imgs = imgs.to(device)
    all_probs = []
    for model in members:
        logits = model(imgs)
        # Resize to a common resolution
        logits_resized = F.interpolate(logits, size=(target_size, target_size),
                                        mode="bilinear", align_corners=False)
        probs = F.softmax(logits_resized, dim=1)
        all_probs.append(probs)

    # (K, B, C, H, W) -> stack
    stacked = torch.stack(all_probs, dim=0)  # (K, B, C, H, W)
    mean_probs = stacked.mean(dim=0)  # (B, C, H, W)

    tu = entropy_per_pixel(mean_probs)  # (B, H, W)
    au = torch.stack([entropy_per_pixel(p) for p in all_probs], dim=0).mean(dim=0)
    eu = tu - au  # (B, H, W)

    return eu.cpu().numpy(), mean_probs.cpu().numpy(), tu.cpu().numpy(), au.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Cache per-pixel ensemble targets for VOC")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--eu_map_size", type=int, default=EU_MAP_SIZE,
                        help="Resolution for cached EU maps (default 128)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading ensemble...")
    members = load_ensemble(args.save_dir, device)
    if args.subset_size is not None:
        members = members[:args.subset_size]
        print(f"Using subset of K={args.subset_size} members")

    val_transform = SegTransformVal(crop_size=512)

    # === Train set ===
    print("\n--- VOC 2012 train ---")
    train_ds = VOCSegDataset(args.data_dir, split="train", transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    train_eu_list, train_probs_list = [], []
    for batch_idx, (imgs, masks) in enumerate(train_loader):
        eu, mean_probs, tu, au = compute_pixel_eu_batch(
            members, imgs, device, target_size=args.eu_map_size)
        train_eu_list.append(eu.astype(np.float16))
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}  "
                  f"EU mean={eu.mean():.4f}")

    train_eu = np.concatenate(train_eu_list, axis=0)
    print(f"  Train EU shape: {train_eu.shape}  mean={train_eu.mean():.4f}")

    # === Val set ===
    print("\n--- VOC 2012 val ---")
    val_ds = VOCSegDataset(args.data_dir, split="val", transform=val_transform,
                            use_sbd=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    val_eu_list, val_probs_list = [], []
    val_tu_list, val_au_list = [], []
    for imgs, masks in val_loader:
        eu, mean_probs, tu, au = compute_pixel_eu_batch(
            members, imgs, device, target_size=args.eu_map_size)
        val_eu_list.append(eu.astype(np.float16))
        val_tu_list.append(tu.astype(np.float16))
        val_au_list.append(au.astype(np.float16))

    val_eu = np.concatenate(val_eu_list, axis=0)
    val_tu = np.concatenate(val_tu_list, axis=0)
    val_au = np.concatenate(val_au_list, axis=0)
    print(f"  Val EU shape: {val_eu.shape}  mean={val_eu.mean():.4f}")

    # === Corrupted train (for Phase 2 Tier 2) ===
    print("\n--- Corrupted VOC train ---")
    from data import CORRUPTIONS
    corrupt_eus = {}
    for cname, cfn in CORRUPTIONS.items():
        print(f"  Corruption: {cname}")
        c_eu_list = []
        for imgs, masks in train_loader:
            c_imgs = torch.stack([cfn(img) for img in imgs])
            eu, _, _, _ = compute_pixel_eu_batch(
                members, c_imgs, device, target_size=args.eu_map_size)
            c_eu_list.append(eu.astype(np.float16))
        corrupt_eus[cname] = np.concatenate(c_eu_list, axis=0)
        print(f"    EU mean={corrupt_eus[cname].mean():.4f}")

    # === Save ===
    results = {
        "train_eu": train_eu,
        "val_eu": val_eu,
        "val_tu": val_tu,
        "val_au": val_au,
        "eu_map_size": np.array(args.eu_map_size),
    }
    for cname, c_eu in corrupt_eus.items():
        results[f"corrupt_{cname}_eu"] = c_eu

    out_path = os.path.join(args.save_dir, "teacher_targets.npz")
    np.savez_compressed(out_path, **results)
    print(f"\nSaved to {out_path}")
    print(f"Keys: {sorted(results.keys())}")
    print(f"Size: {os.path.getsize(out_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
