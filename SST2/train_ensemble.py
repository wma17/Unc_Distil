"""
Train a 5-member LoRA BERT ensemble on SST-2 for uncertainty quantification.

Each member independently samples its own configuration:
    1. Random seed
    2. LoRA init scale (log-uniform [0.5, 1.5])
    3. Label smoothing (from {0.0, 0.02, 0.05})
    4. Dropout on BERT attention (from {0.1, 0.15})
    5. Learning rate (from {1e-4, 2e-4, 3e-4})

Usage:
    python train_ensemble.py --num_members 5 --epochs 5 --gpu 0
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import create_teacher
from data import load_sst2, collate_fn


@dataclass
class MemberConfig:
    member_id: int
    seed: int
    rank: int = 8
    alpha: float = 16.0
    init_scale: float = 1.0
    label_smoothing: float = 0.0
    attention_dropout: float = 0.1
    lr: float = 2e-4
    weight_decay: float = 0.01


def generate_diverse_configs(num_members: int, base_seed: int = 42):
    configs = []
    label_smoothings = [0.0, 0.02, 0.05]
    attention_dropouts = [0.1, 0.15]
    learning_rates = [1e-4, 2e-4, 3e-4]

    for i in range(num_members):
        seed = base_seed + i
        rng = random.Random(seed)
        init_scale = round(math.exp(rng.uniform(math.log(0.5), math.log(1.5))), 3)
        configs.append(MemberConfig(
            member_id=i,
            seed=seed,
            init_scale=init_scale,
            label_smoothing=rng.choice(label_smoothings),
            attention_dropout=rng.choice(attention_dropouts),
            lr=rng.choice(learning_rates),
        ))
    return configs


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += input_ids.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * input_ids.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += input_ids.size(0)
    return total_loss / total, 100.0 * correct / total


def train_single_member(cfg: MemberConfig, args, device):
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    print(f"\n{'='*70}")
    print(f"  Member {cfg.member_id + 1}/{args.num_members}  (seed={cfg.seed})")
    print(f"  lr={cfg.lr}  smooth={cfg.label_smoothing}  attn_drop={cfg.attention_dropout}  "
          f"init_scale={cfg.init_scale}")
    print(f"{'='*70}")

    model = create_teacher(
        num_classes=2, rank=cfg.rank, alpha=cfg.alpha,
        attention_dropout=cfg.attention_dropout, init_scale=cfg.init_scale,
    ).to(device)

    train_ds, dev_ds, tokenizer = load_sst2("bert-base-uncased")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.trainable_parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0.0
    save_path = os.path.join(args.save_dir, f"member_{cfg.member_id}.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device)
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch}/{args.epochs} | "
              f"Train {train_acc:.2f}% (loss {train_loss:.4f}) | "
              f"Dev {dev_acc:.2f}% (loss {dev_loss:.4f}) | {elapsed:.1f}s")

        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save({
                "model_state_dict": model.trainable_state_dict(),
                "epoch": epoch,
                "dev_acc": dev_acc,
                "member_config": asdict(cfg),
            }, save_path)

    print(f"  Member {cfg.member_id} best dev accuracy: {best_acc:.2f}%  ->  {save_path}")
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Train BERT LoRA ensemble on SST-2")
    parser.add_argument("--num_members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_seed", type=int, default=42)
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
        acc = train_single_member(cfg, args, device)
        results.append(acc)

    print(f"\n{'='*70}")
    print(f"  Ensemble training complete!")
    print(f"{'='*70}")
    for cfg, acc in zip(configs, results):
        print(f"  Member {cfg.member_id}: {acc:.2f}%  "
              f"(lr={cfg.lr}, smooth={cfg.label_smoothing}, "
              f"attn_drop={cfg.attention_dropout})")
    print(f"\n  Mean accuracy: {sum(results)/len(results):.2f}%")
    print(f"  Next step: python cache_ensemble_targets.py --save_dir {args.save_dir}")


if __name__ == "__main__":
    main()
