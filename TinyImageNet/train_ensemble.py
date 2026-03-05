"""
Train a diverse LoRA ensemble of DeiT-Small models on Tiny-ImageNet-200.

Each member gets a stochastic combination of:
    - LoRA rank / alpha / dropout / target layers
    - Augmentation strategy (basic, RandAugment, AutoAugment, ColorJitter)
    - Label smoothing ∈ [0, 0.1]
    - Data bagging (80-100 % subset)
    - Learning rate ∈ [1e-4, 5e-4], weight decay ∈ {0.01, 0.05}
    - LR schedule (cosine with varying warmup)

Usage:
    python train_ensemble.py --num_members 5 --epochs 30 --gpu 0
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
        pg["lr"] = lr
    return lr


def sample_member_config(member_id: int, base_seed: int) -> dict:
    rng = random.Random(base_seed + member_id)

    rank = rng.choice([4, 8, 16])
    alpha = rng.choice([float(rank), float(2 * rank)])
    lora_dropout = rng.choice([0.0, 0.05, 0.1])
    targets = rng.choice(["qkv_only", "qkv+proj", "qkv+proj+mlp"])

    aug = rng.choice(["basic", "randaugment", "autoaugment", "colorjitter"])
    label_smooth = round(rng.uniform(0.0, 0.1), 3)
    bag_frac = round(rng.uniform(0.8, 1.0), 2)
    lr = round(rng.uniform(1e-4, 5e-4), 6)
    wd = rng.choice([0.01, 0.05])
    warmup_epochs = rng.choice([5, 10])

    return {
        "member_id": member_id,
        "seed": base_seed + member_id,
        "rank": rank,
        "alpha": alpha,
        "lora_dropout": lora_dropout,
        "targets": targets,
        "augmentation": aug,
        "label_smoothing": label_smooth,
        "bag_fraction": bag_frac,
        "lr": lr,
        "weight_decay": wd,
        "warmup_epochs": warmup_epochs,
    }


def train_one_member(cfg: dict, args):
    set_seed(cfg["seed"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Member {cfg['member_id']}  config: {json.dumps(cfg, indent=2)}")
    print(f"{'='*60}")

    root = download_tiny_imagenet(args.data_dir)

    train_tf = get_train_transform(cfg["augmentation"])
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
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler = GradScaler()

    best_acc = 0.0
    for epoch in range(args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs, cfg["warmup_epochs"], cfg["lr"])
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
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

        print(f"  [{epoch+1:02d}/{args.epochs}] lr={lr:.6f}  "
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
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
