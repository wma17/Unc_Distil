"""
Train a diverse LoRA ensemble of DeiT-Small models on Tiny-ImageNet-200.

Each member gets a stochastic combination of:
    - Moderate-rank LoRA adapters plus light partial backbone fine-tuning
    - Augmentation bundle with at least 3 of:
      RandAugment, MixUp, CutMix, random erasing, color jitter
    - Label smoothing ∈ [0.02, 0.05]
    - Data bagging (80-100 % subset)
    - Learning rate ∈ [5e-5, 2e-4], with very-low LR on unfrozen backbone
    - Weight decay fixed at 0.05
    - LR schedule (cosine with varying warmup)

Usage:
    python train_ensemble.py --num_members 5 --epochs 80 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from data import (
    TinyImageNetDataset,
    download_tiny_imagenet,
    get_train_transform,
    get_val_transform,
)
from models import create_ensemble_member, save_member


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr * pg.get("lr_scale", 1.0)
    return lr


def sample_augmentations(rng) -> list[str]:
    """Sample at least 3 augmentations from the main training pool."""
    pool = ["randaugment", "mixup", "cutmix", "erasing", "colorjitter"]
    n_aug = rng.choice([3, 4, 5])
    return rng.sample(pool, k=n_aug)


def mixup_batch(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def cutmix_batch(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, h, w = x.shape

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / max(1, h * w)
    return mixed_x, y, y[idx], lam


def sample_member_config(member_id: int, base_seed: int) -> dict:
    rng = random.Random(base_seed + member_id)

    rank = rng.choice([8, 16, 32])
    alpha = 8.0 if rank == 8 else 16.0
    lora_dropout = 0.1 if rank >= 32 else 0.05
    targets = rng.choice(["qkv+proj", "qkv+proj+mlp"])
    unfreeze_blocks = 2

    aug = sample_augmentations(rng)
    label_smooth = round(rng.uniform(0.02, 0.05), 3)
    bag_frac = round(rng.uniform(0.8, 1.0), 2)
    lr = round(rng.uniform(5e-5, 2e-4), 6)
    wd = 0.05
    warmup_epochs = rng.choice([5, 10])

    return {
        "member_id": member_id,
        "seed": base_seed + member_id,
        "rank": rank,
        "alpha": alpha,
        "lora_dropout": lora_dropout,
        "targets": targets,
        "unfreeze_blocks": unfreeze_blocks,
        "augmentations": aug,
        "label_smoothing": label_smooth,
        "bag_fraction": bag_frac,
        "lr": lr,
        "weight_decay": wd,
        "warmup_epochs": warmup_epochs,
    }


def build_param_groups(model: nn.Module, backbone_lr_factor: float):
    """Use full LR for head/LoRA and a much lower LR for unfrozen backbone weights."""
    fast_params = []
    slow_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head.") or "lora_A" in name or "lora_B" in name:
            fast_params.append(param)
        else:
            slow_params.append(param)

    param_groups = []
    if fast_params:
        param_groups.append({"params": fast_params, "lr_scale": 1.0, "group_name": "head+lora"})
    if slow_params:
        param_groups.append({
            "params": slow_params,
            "lr_scale": backbone_lr_factor,
            "group_name": "backbone",
        })

    return param_groups, len(fast_params), len(slow_params)


def train_one_member(cfg: dict, args):
    set_seed(cfg["seed"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Member {cfg['member_id']}  config: {json.dumps(cfg, indent=2)}")
    print(f"{'='*60}")

    root = download_tiny_imagenet(args.data_dir)

    train_tf = get_train_transform(cfg["augmentations"])
    val_tf = get_val_transform()
    train_ds = TinyImageNetDataset(root, split="train", transform=train_tf)
    val_ds = TinyImageNetDataset(root, split="val", transform=val_tf)

    if cfg["bag_fraction"] < 1.0:
        n_keep = int(len(train_ds) * cfg["bag_fraction"])
        indices = torch.randperm(len(train_ds))[:n_keep].tolist()
        train_ds = Subset(train_ds, indices)
        print(f"  Bagging: {n_keep}/{len(train_ds) + (len(train_ds.dataset) - n_keep)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = create_ensemble_member(
        rank=cfg["rank"], alpha=cfg["alpha"],
        lora_dropout=cfg["lora_dropout"], targets=cfg["targets"],
        unfreeze_blocks=cfg.get("unfreeze_blocks", 0),
    ).to(device)

    param_groups, n_fast, n_slow = build_param_groups(model, args.backbone_lr_factor)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler = GradScaler()
    print(f"  Optimizer groups: head+lora={n_fast} tensors @ x1.0 lr, "
          f"backbone={n_slow} tensors @ x{args.backbone_lr_factor:.4f} lr")
    print(f"  Augmentations: {cfg['augmentations']}")

    aug_set = set(cfg["augmentations"])
    batch_aug_ops = []
    if "mixup" in aug_set:
        batch_aug_ops.append("mixup")
    if "cutmix" in aug_set:
        batch_aug_ops.append("cutmix")

    best_acc = 0.0
    for epoch in range(args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs, cfg["warmup_epochs"], cfg["lr"])
        backbone_lr = lr * args.backbone_lr_factor if n_slow > 0 else 0.0
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            labels_a, labels_b, lam = labels, labels, 1.0
            if batch_aug_ops:
                aug_name = random.choice(batch_aug_ops)
                if aug_name == "mixup":
                    imgs, labels_a, labels_b, lam = mixup_batch(imgs, labels, alpha=0.2)
                else:
                    imgs, labels_a, labels_b, lam = cutmix_batch(imgs, labels, alpha=1.0)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(imgs)
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            correct += lam * (preds == labels_a).sum().item() + (1.0 - lam) * (preds == labels_b).sum().item()
            total += imgs.size(0)

        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast():
                    logits = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total

        print(f"  [{epoch+1:02d}/{args.epochs}] lr={lr:.6f}  backbone_lr={backbone_lr:.6f}  "
              f"train_loss={total_loss/total:.4f}  train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(args.save_dir, f"member_{cfg['member_id']}.pt")
            save_member(model, ckpt_path, extra={"config": cfg, "best_val_acc": best_acc})
            print(f"    -> saved (best val_acc={best_acc:.4f})")

    print(f"  Member {cfg['member_id']} done. Best val_acc = {best_acc:.4f}")
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Train LoRA ViT ensemble on Tiny-ImageNet")
    parser.add_argument("--num_members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--backbone_lr_factor", type=float, default=0.02,
                        help="Multiplier for unfrozen backbone LR relative to head/LoRA LR")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    configs = [sample_member_config(i, args.base_seed) for i in range(args.num_members)]
    with open(os.path.join(args.save_dir, "ensemble_configs.json"), "w") as f:
        json.dump(configs, f, indent=2)
    print(f"Saved ensemble configs to {args.save_dir}/ensemble_configs.json")

    results = {}
    for cfg in configs:
        acc = train_one_member(cfg, args)
        results[cfg["member_id"]] = acc

    print(f"\n{'='*60}")
    print("Ensemble training complete.")
    for mid, acc in results.items():
        print(f"  Member {mid}: best val_acc = {acc:.4f}")
    print(f"  Mean val_acc = {np.mean(list(results.values())):.4f}")


if __name__ == "__main__":
    main()
