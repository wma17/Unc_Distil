"""
Train a deep ensemble of compact CNNs on MNIST with randomly sampled
per-member diversity strategies for uncertainty quantification.

Each member independently samples its own configuration:
    1. Random seed              → different init + mini-batch order
    2. Random augmentation mix  → random subset of {rotation, affine, erasing}
    3. Label smoothing          → uniform [0, 0.05]
    4. Dropout before FC        → choice from {0, 0.05, 0.1}
    5. Bagging                  → random 80-100% data subset
    6. LR / weight decay        → perturbed around base values
    7. LR schedule              → choice from {cosine, step}
    8. Head init scale          → log-uniform [0.5, 1.5]

Usage:
    python train_ensemble.py --num_members 5 --epochs 50 --gpu 0
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import mnist_convnet


MNIST_MEAN = (0.1307, 0.1307, 0.1307)
MNIST_STD = (0.3081, 0.3081, 0.3081)


@dataclass
class MemberConfig:
    member_id: int
    seed: int
    augmentations: List[str] = field(default_factory=lambda: [])
    label_smoothing: float = 0.0
    dropout_rate: float = 0.0
    data_fraction: float = 1.0
    lr: float = 0.01
    weight_decay: float = 1e-4
    lr_schedule: str = "cosine"
    head_init_scale: float = 1.0


def sample_augmentations(rng):
    augs = []
    if rng.random() < 0.5:
        augs.append("rotation")
    if rng.random() < 0.4:
        augs.append("affine")
    if rng.random() < 0.3:
        augs.append("erasing")
    if rng.random() < 0.2:
        augs.append("perspective")
    return augs


def generate_diverse_configs(num_members, base_seed=42):
    configs = []
    for i in range(num_members):
        seed = base_seed + i
        member_rng = random.Random(seed)

        augs = sample_augmentations(member_rng)
        label_smoothing = round(member_rng.uniform(0.0, 0.05), 4)
        dropout_rate = member_rng.choice([0.0, 0.05, 0.1])
        data_fraction = round(member_rng.uniform(0.8, 1.0), 2)
        lr = round(member_rng.uniform(0.005, 0.02), 4)
        weight_decay = member_rng.choice([1e-4, 3e-4, 5e-4])
        lr_schedule = member_rng.choice(["cosine", "step"])
        head_init_scale = round(math.exp(member_rng.uniform(math.log(0.5), math.log(1.5))), 3)

        configs.append(MemberConfig(
            member_id=i, seed=seed, augmentations=augs,
            label_smoothing=label_smoothing, dropout_rate=dropout_rate,
            data_fraction=data_fraction, lr=lr, weight_decay=weight_decay,
            lr_schedule=lr_schedule, head_init_scale=head_init_scale,
        ))
    return configs


def generate_uniform_configs(num_members, base_seed=42):
    return [MemberConfig(member_id=i, seed=base_seed + i)
            for i in range(num_members)]


def build_train_transform(aug_list):
    pre = []
    if "rotation" in aug_list:
        pre.append(transforms.RandomRotation(15))
    if "affine" in aug_list:
        pre.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)))
    if "perspective" in aug_list:
        pre.append(transforms.RandomPerspective(distortion_scale=0.2, p=0.3))

    mid = [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ]

    post = []
    if "erasing" in aug_list:
        post.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)))

    return transforms.Compose(pre + mid + post)


def get_loaders(cfg, data_dir, batch_size=128, num_workers=4):
    train_transform = build_train_transform(cfg.augmentations)
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])

    train_set = datasets.MNIST(root=data_dir, train=True, download=False, transform=train_transform)
    test_set = datasets.MNIST(root=data_dir, train=False, download=False, transform=test_transform)

    if cfg.data_fraction < 1.0:
        rng = np.random.RandomState(cfg.seed)
        n = len(train_set)
        indices = rng.choice(n, size=int(n * cfg.data_fraction), replace=False)
        train_set = Subset(train_set, indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def build_scheduler(optimizer, cfg, epochs):
    if cfg.lr_schedule == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif cfg.lr_schedule == "step":
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[epochs // 2, 3 * epochs // 4], gamma=0.1)
    else:
        raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


def train_single_member(cfg, args, device):
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    aug_str = "+".join(cfg.augmentations) if cfg.augmentations else "base_only"
    print(f"\n{'='*70}")
    print(f"  Member {cfg.member_id + 1}/{args.num_members}  (seed={cfg.seed})")
    print(f"  augs=[{aug_str}]  smooth={cfg.label_smoothing}  "
          f"drop={cfg.dropout_rate}  data={cfg.data_fraction:.0%}")
    print(f"  lr={cfg.lr}  wd={cfg.weight_decay}  sched={cfg.lr_schedule}  "
          f"head_scale={cfg.head_init_scale}")
    print(f"{'='*70}")

    model = mnist_convnet(
        num_classes=10,
        dropout_rate=cfg.dropout_rate,
        head_init_scale=cfg.head_init_scale,
    ).to(device)

    train_loader, test_loader = get_loaders(cfg, args.data_dir, args.batch_size, args.num_workers)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg, args.epochs)

    best_acc = 0.0
    save_path = os.path.join(args.save_dir, f"member_{cfg.member_id}.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train {train_acc:5.2f}% (loss {train_loss:.4f}) | "
                  f"Test {test_acc:5.2f}% (loss {test_loss:.4f}) | "
                  f"LR {lr_now:.6f} | {elapsed:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "test_acc": test_acc,
                "member_config": asdict(cfg),
            }, save_path)

    print(f"  Member {cfg.member_id} best test accuracy: {best_acc:.2f}%  ->  {save_path}")
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Train diverse deep ensemble on MNIST")
    parser.add_argument("--num_members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no_diversity", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    if args.no_diversity:
        configs = generate_uniform_configs(args.num_members, args.base_seed)
        print("\n*** Ablation mode: seed-only diversity ***")
    else:
        configs = generate_diverse_configs(args.num_members, args.base_seed)

    config_path = os.path.join(args.save_dir, "ensemble_configs.json")
    with open(config_path, "w") as f:
        json.dump([asdict(c) for c in configs], f, indent=2)
    print(f"Configs saved to {config_path}")

    results = []
    for cfg in configs:
        acc = train_single_member(cfg, args, device)
        results.append(acc)

    print(f"\n{'='*70}")
    print("  Ensemble training complete!")
    print(f"{'='*70}")
    for cfg, acc in zip(configs, results):
        aug_str = "+".join(cfg.augmentations) if cfg.augmentations else "base"
        print(f"  Member {cfg.member_id}: {acc:.2f}%  "
              f"(augs=[{aug_str}], smooth={cfg.label_smoothing}, "
              f"drop={cfg.dropout_rate}, sched={cfg.lr_schedule})")
    print(f"\n  Mean accuracy: {sum(results)/len(results):.2f}%")
    print(f"  Next step:  python cache_ensemble_targets.py --save_dir {args.save_dir}")


if __name__ == "__main__":
    main()
