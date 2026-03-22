"""
Two-phase distillation for SST-2: BERT teacher ensemble -> DistilBERT student.

Phase 1 — Classification KD (full student):
    L = (1-α) CE(y, z_S) + α τ² KL(softmax(z_T/τ) || softmax(z_S/τ))

Phase 2 — EU head (freeze backbone + classifier, train EU head):
    L = log1p_MSE(EU_S, EU_T) + β · PairwiseRankingLoss
    Three-tier curriculum: clean + perturbed + synthetic OOD (masked tokens)

Usage:
    python cache_ensemble_targets.py --save_dir ./checkpoints  # prerequisite
    python distill.py --save_dir ./checkpoints --gpu 0
    python distill.py --save_dir ./checkpoints --gpu 0 --phase2_only
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from models import create_student
from data import (load_sst2, SST2Dataset, EUTextDataset, PerturbedSST2Dataset,
                  TokenMaskedSST2Dataset, collate_fn, MAX_SEQ_LEN)


EPS = 1e-8


# ---------------------------------------------------------------------------
# Phase 1 Dataset
# ---------------------------------------------------------------------------

class DistillTextDataset:
    """SST-2 paired with teacher soft labels and EU."""

    def __init__(self, texts, labels, teacher_probs, teacher_eu, tokenizer,
                 max_len=MAX_SEQ_LEN):
        self.texts = texts
        self.labels = labels
        self.teacher_probs = torch.from_numpy(teacher_probs).float()
        self.teacher_eu = torch.from_numpy(teacher_eu).float()
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
            "teacher_probs": self.teacher_probs[idx],
            "teacher_eu": self.teacher_eu[idx],
        }


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def pearson_corr(a, b):
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def spearman_corr(a, b):
    def _rank(x):
        order = x.argsort()
        ranks = torch.empty_like(x)
        ranks[order] = torch.arange(len(x), dtype=x.dtype, device=x.device)
        return ranks
    return torch.corrcoef(torch.stack([_rank(a), _rank(b)]))[0, 1].item()


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def phase1_loss(student_logits, labels, teacher_probs, alpha, tau):
    loss_ce = F.cross_entropy(student_logits, labels)
    teacher_logits = torch.log(teacher_probs + EPS)
    teacher_soft = F.softmax(teacher_logits / tau, dim=-1)
    student_log_soft = F.log_softmax(student_logits / tau, dim=-1)
    loss_kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (tau ** 2)
    loss = (1 - alpha) * loss_ce + alpha * loss_kl
    return loss, loss_ce.detach().item(), loss_kl.detach().item()


def train_phase1_epoch(model, loader, optimizer, scheduler, alpha, tau, device):
    model.train()
    sum_ce, sum_kl, correct, total = 0, 0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        t_probs = batch["teacher_probs"].to(device)

        optimizer.zero_grad()
        logits, _eu = model(input_ids, attention_mask)
        loss, ce, kl = phase1_loss(logits, labels, t_probs, alpha, tau)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        bs = input_ids.size(0)
        sum_ce += ce * bs
        sum_kl += kl * bs
        correct += logits.argmax(1).eq(labels).sum().item()
        total += bs
    return sum_ce / total, sum_kl / total, 100.0 * correct / total


@torch.no_grad()
def eval_phase1(model, loader, alpha, tau, device):
    model.eval()
    sum_ce, sum_kl, correct, total = 0, 0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        t_probs = batch["teacher_probs"].to(device)

        logits, _eu = model(input_ids, attention_mask)
        _, ce, kl = phase1_loss(logits, labels, t_probs, alpha, tau)
        bs = input_ids.size(0)
        sum_ce += ce * bs
        sum_kl += kl * bs
        correct += logits.argmax(1).eq(labels).sum().item()
        total += bs
    return sum_ce / total, sum_kl / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def log1p_mse_loss(pred, target):
    return F.mse_loss(torch.log1p(pred), torch.log1p(target))


def pairwise_ranking_loss(pred, target, n_pairs=256, margin=0.05):
    bs = pred.size(0)
    if bs < 2:
        return pred.new_tensor(0.0)
    idx_i = torch.randint(0, bs, (n_pairs,), device=pred.device)
    idx_j = torch.randint(0, bs, (n_pairs,), device=pred.device)
    t_i, t_j = target[idx_i], target[idx_j]
    mask = t_i > t_j + EPS
    if mask.sum() < 1:
        return pred.new_tensor(0.0)
    p_i, p_j = pred[idx_i][mask], pred[idx_j][mask]
    return F.margin_ranking_loss(
        p_i, p_j,
        torch.ones(mask.sum(), device=pred.device),
        margin=margin,
    )


def train_phase2_epoch(model, loader, optimizer, rank_weight, device,
                       loss_mode="combined"):
    # Backbone eval, EU head train
    model.eval()
    model.eu_fc1.train()
    model.eu_fc2.train()

    sum_mse, sum_rank, total = 0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        t_eu = batch["eu_target"].to(device)

        optimizer.zero_grad()
        _logits, eu_pred = model(input_ids, attention_mask)

        if loss_mode == "mse":
            l_mse = F.mse_loss(eu_pred, t_eu)
            l_rank = eu_pred.new_tensor(0.0)
            loss = l_mse
        elif loss_mode == "log_mse":
            l_mse = log1p_mse_loss(eu_pred, t_eu)
            l_rank = eu_pred.new_tensor(0.0)
            loss = l_mse
        elif loss_mode == "ranking":
            l_mse = eu_pred.new_tensor(0.0)
            l_rank = pairwise_ranking_loss(eu_pred, t_eu)
            loss = l_rank
        else:  # combined
            l_mse = log1p_mse_loss(eu_pred, t_eu)
            l_rank = pairwise_ranking_loss(eu_pred, t_eu)
            loss = l_mse + rank_weight * l_rank

        loss.backward()
        optimizer.step()
        bs = input_ids.size(0)
        sum_mse += l_mse.detach().item() * bs
        sum_rank += l_rank.detach().item() * bs
        total += bs
    return sum_mse / total, sum_rank / total


@torch.no_grad()
def eval_phase2(model, loader, device):
    model.eval()
    sum_loss, total = 0, 0
    eu_preds, eu_targets = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        t_eu = batch["eu_target"].to(device)

        _logits, eu_pred = model(input_ids, attention_mask)
        loss = log1p_mse_loss(eu_pred, t_eu)
        sum_loss += loss.item() * input_ids.size(0)
        total += input_ids.size(0)
        eu_preds.append(eu_pred.cpu())
        eu_targets.append(t_eu.cpu())

    eu_preds = torch.cat(eu_preds)
    eu_targets = torch.cat(eu_targets)
    r_pearson = pearson_corr(eu_preds, eu_targets)
    r_spearman = spearman_corr(eu_preds, eu_targets)
    return sum_loss / total, r_pearson, r_spearman


# ---------------------------------------------------------------------------
# Phase 2 dataset builder
# ---------------------------------------------------------------------------

def build_phase2_datasets(data, tokenizer, curriculum="A3"):
    """Build mixed training dataset for Phase 2.

    A1: clean only
    A2: clean + perturbed
    A3: clean + perturbed + synthetic OOD (masked tokens)
    """
    from datasets import load_dataset
    raw_ds = load_dataset("glue", "sst2")
    train_texts = raw_ds["train"]["sentence"]
    dev_texts = raw_ds["validation"]["sentence"]
    n_train = len(train_texts)

    rng = np.random.RandomState(2026)

    datasets_list = []

    # Tier 1: clean
    clean_ds = EUTextDataset(train_texts, data["train_eu"], tokenizer)
    datasets_list.append(clean_ds)
    print(f"  Tier 1 (clean): {len(clean_ds)} samples")

    # Tier 2: perturbed
    if curriculum in ("A2", "A3"):
        n_perturbed = n_train // 4  # 25% total -> half char, half word
        n_char = n_perturbed // 2
        n_word = n_perturbed - n_char

        char_idx = rng.choice(n_train, size=n_char, replace=False)
        char_texts = [apply_char_perturbations_static(train_texts[i], seed=2026 + i)
                      for i in char_idx]
        char_eu = data["char_perturbed_eu"][char_idx]
        char_ds = EUTextDataset(char_texts, char_eu, tokenizer)
        datasets_list.append(char_ds)

        word_idx = rng.choice(n_train, size=n_word, replace=False)
        word_texts = [apply_word_perturbations_static(train_texts[i], seed=3026 + i)
                      for i in word_idx]
        word_eu = data["word_perturbed_eu"][word_idx]
        word_ds = EUTextDataset(word_texts, word_eu, tokenizer)
        datasets_list.append(word_ds)
        print(f"  Tier 2 (perturbed): {n_char} char + {n_word} word = {n_perturbed}")

    # Tier 3: synthetic OOD (masked tokens)
    if curriculum == "A3":
        n_ood = n_train // 4
        mask_rates = [0.3, 0.5, 0.7]
        n_per_rate = n_ood // len(mask_rates)
        for rate in mask_rates:
            key = f"masked_{rate}_eu"
            if key not in data:
                print(f"  Warning: {key} not in cache, skipping")
                continue
            idx = rng.choice(n_train, size=n_per_rate, replace=False)
            mask_texts = [train_texts[i] for i in idx]
            mask_eu = data[key][idx]
            mask_ds = TokenMaskedSST2Dataset(
                mask_texts, mask_eu, tokenizer,
                mask_rate=rate, seed=2027)
            datasets_list.append(mask_ds)
        print(f"  Tier 3 (masked OOD): ~{n_ood} samples")

    train_dataset = ConcatDataset(datasets_list)
    print(f"  Phase 2 total: {len(train_dataset)} samples")

    # Dev: clean only
    dev_dataset = EUTextDataset(dev_texts, data["dev_eu"], tokenizer)
    return train_dataset, dev_dataset


def apply_char_perturbations_static(text, seed):
    from data import apply_char_perturbations
    return apply_char_perturbations(text, seed=seed)


def apply_word_perturbations_static(text, seed):
    from data import apply_word_perturbations
    return apply_word_perturbations(text, seed=seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Two-phase distillation for SST-2")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--p1_epochs", type=int, default=20)
    parser.add_argument("--p1_lr", type=float, default=2e-4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=6.0)

    parser.add_argument("--p2_epochs", type=int, default=100)
    parser.add_argument("--p2_lr", type=float, default=0.001)
    parser.add_argument("--rank_weight", type=float, default=1.0)

    parser.add_argument("--phase2_only", action="store_true")

    parser.add_argument("--curriculum", type=str, default="A3",
                        choices=["A1", "A2", "A3"])
    parser.add_argument("--loss_mode", type=str, default="combined",
                        choices=["mse", "log_mse", "ranking", "combined"])
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    targets_path = os.path.join(args.save_dir, "teacher_targets.npz")
    if not os.path.exists(targets_path):
        raise FileNotFoundError(f"{targets_path} not found. Run cache_ensemble_targets.py first.")
    data = np.load(targets_path, allow_pickle=True)

    train_ds, dev_ds, tokenizer = load_sst2("distilbert-base-uncased")
    from datasets import load_dataset
    raw_ds = load_dataset("glue", "sst2")
    train_texts = raw_ds["train"]["sentence"]
    dev_texts = raw_ds["validation"]["sentence"]

    model = create_student(num_classes=2).to(device)
    p1_path = os.path.join(args.save_dir, "student_phase1.pt")
    final_path = os.path.join(args.save_dir, "student.pt")

    # ==================================================================
    # Phase 1: Classification KD
    # ==================================================================
    if not args.phase2_only:
        print(f"\n{'='*70}")
        print(f"  Phase 1: Classification distillation (α={args.alpha}, τ={args.tau})")
        print(f"  {args.p1_epochs} epochs, lr={args.p1_lr}")
        print(f"{'='*70}\n")

        p1_train_ds = DistillTextDataset(
            train_texts, raw_ds["train"]["label"],
            data["train_probs"], data["train_eu"], tokenizer)
        p1_dev_ds = DistillTextDataset(
            dev_texts, raw_ds["validation"]["label"],
            data["dev_probs"], data["dev_eu"], tokenizer)

        p1_train_loader = DataLoader(p1_train_ds, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers,
                                     pin_memory=True, collate_fn=collate_fn)
        p1_dev_loader = DataLoader(p1_dev_ds, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True, collate_fn=collate_fn)

        optimizer = optim.AdamW(model.parameters(), lr=args.p1_lr, weight_decay=0.01)
        total_steps = len(p1_train_loader) * args.p1_epochs
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        best_acc = 0.0

        for epoch in range(1, args.p1_epochs + 1):
            t0 = time.time()
            tr_ce, tr_kl, tr_acc = train_phase1_epoch(
                model, p1_train_loader, optimizer, scheduler, args.alpha, args.tau, device)
            te_ce, te_kl, te_acc = eval_phase1(
                model, p1_dev_loader, args.alpha, args.tau, device)
            elapsed = time.time() - t0

            print(f"  P1 Epoch {epoch:2d}/{args.p1_epochs} | "
                  f"Train {tr_acc:.2f}% (ce={tr_ce:.4f} kl={tr_kl:.4f}) | "
                  f"Dev {te_acc:.2f}% (ce={te_ce:.4f} kl={te_kl:.4f}) | {elapsed:.1f}s")

            if te_acc > best_acc:
                best_acc = te_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch, "dev_acc": te_acc, "phase": 1,
                }, p1_path)

        print(f"\n  Phase 1 best accuracy: {best_acc:.2f}%  ->  {p1_path}")
    else:
        print(f"\nSkipping Phase 1, loading {p1_path}")

    # Load best Phase 1 model
    ckpt = torch.load(p1_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  Loaded Phase 1 checkpoint (acc={ckpt.get('dev_acc', '?')}%)")

    # ==================================================================
    # Phase 2: EU head training
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  Phase 2: EU head (backbone frozen)")
    print(f"  Curriculum: {args.curriculum}  |  Loss: {args.loss_mode}")
    print(f"  {args.p2_epochs} epochs, lr={args.p2_lr}, β={args.rank_weight}")
    print(f"{'='*70}\n")

    # Freeze everything except EU head
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("eu_")

    model.reinit_eu_head()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total_params:,} (EU head only)")

    p2_train_ds, p2_dev_ds = build_phase2_datasets(
        data, tokenizer, curriculum=args.curriculum)

    p2_train_loader = DataLoader(p2_train_ds, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True, collate_fn=collate_fn)
    p2_dev_loader = DataLoader(p2_dev_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=True, collate_fn=collate_fn)

    optimizer = optim.Adam(model.eu_head_parameters, lr=args.p2_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.p2_epochs)
    best_spearman = -1.0

    for epoch in range(1, args.p2_epochs + 1):
        t0 = time.time()
        tr_mse, tr_rank = train_phase2_epoch(
            model, p2_train_loader, optimizer, args.rank_weight, device,
            loss_mode=args.loss_mode)
        te_loss, r_pear, r_spear = eval_phase2(model, p2_dev_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            print(f"  P2 Epoch {epoch:2d}/{args.p2_epochs} | "
                  f"Train mse={tr_mse:.6f} rank={tr_rank:.6f} | "
                  f"Dev mse={te_loss:.6f} | "
                  f"Pearson={r_pear:.4f} Spearman={r_spear:.4f} | {elapsed:.1f}s")

        if r_spear > best_spearman:
            best_spearman = r_spear
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "dev_acc": ckpt.get("dev_acc", 0),
                "eu_pearson": r_pear, "eu_spearman": r_spear, "phase": 2,
            }, final_path)

    print(f"\n  Phase 2 best Spearman: {best_spearman:.4f}")
    print(f"  Final student saved to: {final_path}")
    print(f"\n  Next: python evaluate_student.py --save_dir {args.save_dir}")


if __name__ == "__main__":
    main()
