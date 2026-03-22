"""
Train a 5-member LoRA SegFormer-B2 ensemble on Pascal VOC 2012.

Each member independently samples its own configuration:
    1. Random seed
    2. LoRA init scale (log-uniform [0.5, 1.5])
    3. Label smoothing (uniform [0.0, 0.05])
    4. Augmentation: one of {colorjitter, scale, rotation}
    5. Weight decay: one of {1e-4, 5e-4, 1e-3}

Usage:
    python train_ensemble.py --num_members 5 --iterations 40000 --gpu 0
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import create_teacher, NUM_CLASSES
from data import VOCSegDataset, SegTransformTrain, SegTransformVal, IGNORE_INDEX


@dataclass
class MemberConfig:
    member_id: int
    seed: int
    rank: int = 16
    alpha: float = 32.0
    init_scale: float = 1.0
    label_smoothing: float = 0.0
    aug_mode: str = "default"
    weight_decay: float = 0.01
    lr: float = 6e-5


def generate_diverse_configs(num_members: int, base_seed: int = 42):
    configs = []
    aug_modes = ["colorjitter", "scale", "rotation"]
    weight_decays = [1e-4, 5e-4, 1e-3]

    for i in range(num_members):
        seed = base_seed + i
        rng = random.Random(seed)
        configs.append(MemberConfig(
            member_id=i,
            seed=seed,
            init_scale=round(math.exp(rng.uniform(math.log(0.5), math.log(1.5))), 3),
            label_smoothing=round(rng.uniform(0.0, 0.05), 4),
            aug_mode=rng.choice(aug_modes),
            weight_decay=rng.choice(weight_decays),
        ))
    return configs


def poly_lr_lambda(step, max_steps, power=0.9):
    return (1 - step / max_steps) ** power


def compute_miou(pred, target, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    """Compute mean IoU."""
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0


def train_single_member(cfg: MemberConfig, args, device):
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    print(f"\n{'='*70}")
    print(f"  Member {cfg.member_id + 1}/{args.num_members}  (seed={cfg.seed})")
    print(f"  aug={cfg.aug_mode}  smooth={cfg.label_smoothing}  "
          f"wd={cfg.weight_decay}  init_scale={cfg.init_scale}")
    print(f"{'='*70}")

    model = create_teacher(
        num_classes=NUM_CLASSES, rank=cfg.rank, alpha=cfg.alpha,
        init_scale=cfg.init_scale,
    ).to(device)

    train_transform = SegTransformTrain(crop_size=512, aug_mode=cfg.aug_mode)
    val_transform = SegTransformVal(crop_size=512)

    train_ds = VOCSegDataset(args.data_dir, split="train", transform=train_transform)
    val_ds = VOCSegDataset(args.data_dir, split="val", transform=val_transform,
                            use_sbd=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX,
                                     label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.trainable_parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    max_iters = args.iterations
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: poly_lr_lambda(step, max_iters))

    best_miou = 0.0
    save_path = os.path.join(args.save_dir, f"member_{cfg.member_id}.pt")
    global_step = 0
    epoch = 0

    while global_step < max_iters:
        epoch += 1
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0, 0, 0

        for imgs, masks in train_loader:
            if global_step >= max_iters:
                break
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            # Upsample logits to mask resolution
            logits_up = nn.functional.interpolate(
                logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
            loss = criterion(logits_up, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item() * imgs.size(0)
            preds = logits_up.argmax(1)
            valid = masks != IGNORE_INDEX
            epoch_correct += (preds[valid] == masks[valid]).sum().item()
            epoch_total += valid.sum().item()

        # Evaluate every 5 epochs or at the end
        if epoch % 5 == 0 or global_step >= max_iters:
            model.eval()
            val_miou_list = []
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    logits = model(imgs)
                    logits_up = nn.functional.interpolate(
                        logits, size=masks.shape[1:], mode="bilinear",
                        align_corners=False)
                    preds = logits_up.argmax(1)
                    for b in range(preds.shape[0]):
                        miou = compute_miou(preds[b].cpu().numpy(),
                                           masks[b].cpu().numpy())
                        val_miou_list.append(miou)
            val_miou = np.mean(val_miou_list)
            pixel_acc = epoch_correct / max(1, epoch_total) * 100

            print(f"  Epoch {epoch} (step {global_step}/{max_iters}) | "
                  f"Loss {epoch_loss/max(1,len(train_ds)):.4f} | "
                  f"Pixel acc {pixel_acc:.2f}% | Val mIoU {val_miou:.4f}")

            if val_miou > best_miou:
                best_miou = val_miou
                torch.save({
                    "model_state_dict": model.trainable_state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "val_miou": val_miou,
                    "member_config": asdict(cfg),
                }, save_path)

    print(f"  Member {cfg.member_id} best mIoU: {best_miou:.4f}  ->  {save_path}")
    return best_miou


def main():
    parser = argparse.ArgumentParser(description="Train SegFormer ensemble on VOC 2012")
    parser.add_argument("--num_members", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=40000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    configs = generate_diverse_configs(args.num_members, args.base_seed)
    config_path = os.path.join(args.save_dir, "ensemble_configs.json")
    with open(config_path, "w") as f:
        json.dump([asdict(c) for c in configs], f, indent=2)

    results = []
    for cfg in configs:
        miou = train_single_member(cfg, args, device)
        results.append(miou)

    print(f"\n{'='*70}")
    print(f"  Ensemble training complete!")
    print(f"{'='*70}")
    for cfg, miou in zip(configs, results):
        print(f"  Member {cfg.member_id}: mIoU={miou:.4f}  "
              f"(aug={cfg.aug_mode}, smooth={cfg.label_smoothing:.4f})")
    print(f"\n  Mean mIoU: {sum(results)/len(results):.4f}")
    print(f"  Next: python cache_ensemble_targets.py --save_dir {args.save_dir}")


if __name__ == "__main__":
    main()
