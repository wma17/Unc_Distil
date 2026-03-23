"""
Train UQ baselines for SST-2 (E5), starting from ensemble member_0 (BERT LoRA teacher).

Methods (same family as MNIST/CIFAR/TinyImageNet):
  1. MC Dropout  — raise nn.Dropout p in BERT; short fine-tune; stochastic passes at test.
  2. EDL         — Dirichlet head on frozen [CLS] features; train head only.
  3. LLLA        — Laplace on last linear (768→2) over cached CLS features (laplace-torch).
  4. SGLD        — Last-layer SGLD on the same linear head.

Usage:
    python train_baselines.py --save_dir ./checkpoints --gpu 0
    python train_baselines.py --save_dir ./checkpoints --gpu 0 --methods mc_dropout edl
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models import create_teacher
from data import load_sst2, collate_fn


NUM_CLASSES = 2
EPS = 1e-8
CLS_DIM = 768


# ---------------------------------------------------------------------------
# Load member 0
# ---------------------------------------------------------------------------

def load_member0(save_dir: str, device: torch.device):
    path = os.path.join(save_dir, "member_0.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mcfg = ckpt.get("member_config", {})
    model = create_teacher(
        num_classes=NUM_CLASSES,
        rank=mcfg.get("rank", 8),
        alpha=mcfg.get("alpha", 16.0),
        attention_dropout=mcfg.get("attention_dropout", 0.1),
        init_scale=1.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model


# ---------------------------------------------------------------------------
# MC Dropout
# ---------------------------------------------------------------------------

def set_dropout_p(module: nn.Module, p: float) -> None:
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.p = p


def train_mc_dropout(model, train_loader, device, dropout_p: float = 0.2,
                     epochs: int = 2, lr: float = 2e-5):
    model.to(device)
    set_dropout_p(model, dropout_p)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * epochs)
    warmup = max(1, int(0.1 * total_steps))
    step = 0

    def lr_lambda(s):
        if s < warmup:
            return float(s) / warmup
        return max(0.0, float(total_steps - s) / max(1, total_steps - warmup))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    for epoch in range(1, epochs + 1):
        tot_loss, correct, tot = 0.0, 0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            tot_loss += loss.item() * input_ids.size(0)
            correct += logits.argmax(1).eq(labels).sum().item()
            tot += input_ids.size(0)
        print(f"  [MC Dropout] epoch {epoch}/{epochs}  "
              f"loss={tot_loss/tot:.4f}  acc={100*correct/tot:.2f}%")
    return model


# ---------------------------------------------------------------------------
# EDL
# ---------------------------------------------------------------------------

class EDLHead(nn.Module):
    def __init__(self, feat_dim: int = CLS_DIM, hidden: int = 256, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        evidence = F.softplus(self.fc2(F.relu(self.fc1(feat))))
        return evidence + 1.0


def edl_loss(alpha: torch.Tensor, labels: torch.Tensor, num_classes: int, lambda_kl: float = 0.1):
    """Sensoy et al. 2018 EDL loss (same form as CIFAR-10/train_baselines.py)."""
    S = alpha.sum(dim=-1, keepdim=True)
    p = alpha / S
    K = num_classes
    y = F.one_hot(labels, K).float()
    l_err = ((y - p) ** 2).sum(dim=-1)
    l_var = (p * (1.0 - p) / (S + 1.0)).sum(dim=-1)
    alpha_tilde = y + (1.0 - y) * alpha
    S_tilde = alpha_tilde.sum(dim=-1)
    kl = (torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(float(K), device=alpha.device))
          - torch.lgamma(alpha_tilde).sum(dim=-1)
          + ((alpha_tilde - 1.0) * (
              torch.digamma(alpha_tilde) - torch.digamma(S_tilde.unsqueeze(-1))
          )).sum(dim=-1))
    return (l_err + l_var + lambda_kl * kl).mean()


@torch.no_grad()
def bert_cls(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    out = model.bert(input_ids=input_ids, attention_mask=attention_mask)
    return out.last_hidden_state[:, 0]


def train_edl(model, train_loader, device, epochs: int = 50, lr: float = 1e-3,
              lambda_kl: float = 0.1, hidden: int = 256):
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    edl_head = EDLHead(feat_dim=CLS_DIM, hidden=hidden, num_classes=NUM_CLASSES).to(device)
    opt = optim.Adam(edl_head.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        edl_head.train()
        tot_loss, tot = 0.0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            feat = bert_cls(model, input_ids, attention_mask)
            alpha = edl_head(feat)
            loss = edl_loss(alpha, labels, NUM_CLASSES, lambda_kl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * input_ids.size(0)
            tot += input_ids.size(0)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [EDL] epoch {epoch}/{epochs}  loss={tot_loss/tot:.4f}")
    return edl_head


# ---------------------------------------------------------------------------
# CLS feature cache + last-layer linear clone
# ---------------------------------------------------------------------------

@torch.no_grad()
def cache_train_cls_labels(model, train_loader, device):
    model.eval()
    feats, labels = [], []
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lab = batch["label"]
        f = bert_cls(model, input_ids, attention_mask)
        feats.append(f.cpu())
        labels.append(lab)
    return torch.cat(feats, 0), torch.cat(labels, 0)


def clone_classifier_linear(classifier: nn.Linear) -> nn.Linear:
    lin = nn.Linear(classifier.in_features, classifier.out_features, bias=classifier.bias is not None)
    lin.load_state_dict(classifier.state_dict())
    return lin


# ---------------------------------------------------------------------------
# LLLA (standalone last linear; diag Laplace on small head)
# ---------------------------------------------------------------------------

def train_llla(linear_head: nn.Linear, X: torch.Tensor, y: torch.Tensor, device,
               batch_size: int = 512):
    try:
        from laplace import Laplace
    except ImportError:
        print("  [LLLA] laplace-torch not installed. pip install laplace-torch")
        return None

    linear_head = linear_head.to(device).eval()
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    la = Laplace(linear_head, "classification",
                 subset_of_weights="all",
                 hessian_structure="diag")
    print("  [LLLA] Fitting diagonal Laplace on 768→2 head...")
    la.fit(loader)
    print("  [LLLA] Optimizing prior precision...")
    la.optimize_prior_precision(method="marglik")
    print(f"  [LLLA] Prior precision: {float(la.prior_precision.detach().cpu()):.6f}")
    return la


# ---------------------------------------------------------------------------
# SGLD (last linear)
# ---------------------------------------------------------------------------

def train_sgld(linear_head: nn.Linear, X: torch.Tensor, y: torch.Tensor, device,
               step_size: float = 1e-6, burn_in: int = 200, n_samples: int = 16,
               thin: int = 10, prior_sigma: float = 1.0):
    N = X.size(0)
    linear_head = linear_head.to(device).eval()
    W = linear_head.weight.detach().clone().requires_grad_(True)
    B = linear_head.bias.detach().clone().requires_grad_(True)
    params = [W, B]

    total_steps = burn_in + n_samples * thin
    bs = min(256, N)

    samples = []
    t0 = time.time()
    for step in range(total_steps):
        perm = torch.randint(0, N, (bs,))
        xb = X[perm].to(device)
        yb = y[perm].to(device)

        logits = xb @ W.T + B
        loss_ce = F.cross_entropy(logits, yb)
        loss_reg = (W.pow(2).sum() + B.pow(2).sum()) / (2 * prior_sigma ** 2)
        U = N * loss_ce + loss_reg

        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        U.backward()

        with torch.no_grad():
            for p in params:
                noise = torch.randn_like(p) * (step_size ** 0.5)
                p.data -= 0.5 * step_size * p.grad + noise

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append({"weight": W.detach().cpu().clone(), "bias": B.detach().cpu().clone()})

        if step % 100 == 0:
            print(f"  [SGLD] step {step}/{total_steps}  U={U.item():.3f}  n={len(samples)}")

    print(f"  [SGLD] Done. {len(samples)} samples in {time.time()-t0:.1f}s")
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train SST-2 UQ baselines (E5)")
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--methods", nargs="+",
                   default=["mc_dropout", "edl", "llla", "sgld"],
                   choices=["mc_dropout", "edl", "llla", "sgld"])
    p.add_argument("--mc_epochs", type=int, default=2)
    p.add_argument("--mc_dropout_p", type=float, default=0.2)
    p.add_argument("--edl_epochs", type=int, default=50)
    p.add_argument("--edl_lr", type=float, default=1e-3)
    p.add_argument("--edl_lambda_kl", type=float, default=0.1)
    p.add_argument("--sgld_step", type=float, default=1e-6)
    p.add_argument("--sgld_burn_in", type=int, default=200)
    p.add_argument("--sgld_samples", type=int, default=16)
    p.add_argument("--sgld_thin", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    baseline_dir = os.path.join(args.save_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)

    train_ds, _, _ = load_sst2("bert-base-uncased")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    model = load_member0(args.save_dir, device)

    if "mc_dropout" in args.methods:
        print("\n=== MC Dropout ===")
        model = train_mc_dropout(model, train_loader, device,
                                 dropout_p=args.mc_dropout_p, epochs=args.mc_epochs)
        path = os.path.join(baseline_dir, "mc_dropout.pt")
        torch.save({
            "model_state_dict": model.cpu().trainable_state_dict(),
            "dropout_p": args.mc_dropout_p,
        }, path)
        print(f"  Saved -> {path}")

    # Reload fresh member for methods that expect original weights (EDL/LLLA/SGLD)
    model = load_member0(args.save_dir, device)

    if "edl" in args.methods:
        print("\n=== EDL ===")
        edl_head = train_edl(model, train_loader, device,
                             epochs=args.edl_epochs, lr=args.edl_lr,
                             lambda_kl=args.edl_lambda_kl)
        path = os.path.join(baseline_dir, "edl_head.pt")
        torch.save({
            "bert_trainable_state_dict": model.cpu().trainable_state_dict(),
            "edl_head_state_dict": edl_head.cpu().state_dict(),
        }, path)
        print(f"  Saved -> {path}")

    model = load_member0(args.save_dir, device)
    print("\n  Caching [CLS] features for LLLA / SGLD...")
    X, y = cache_train_cls_labels(model, train_loader, device)

    if "llla" in args.methods:
        print("\n=== LLLA ===")
        la = train_llla(clone_classifier_linear(model.classifier), X, y, device)
        if la is not None:
            path = os.path.join(baseline_dir, "llla.pt")
            torch.save({
                "laplace_state_dict": la.state_dict(),
                "linear_state_dict": la.model.cpu().state_dict(),
                "cls_dim": CLS_DIM,
                "num_classes": NUM_CLASSES,
            }, path)
            print(f"  Saved -> {path}")

    if "sgld" in args.methods:
        print("\n=== SGLD ===")
        samples = train_sgld(clone_classifier_linear(model.classifier), X, y, device,
                             step_size=args.sgld_step,
                             burn_in=args.sgld_burn_in,
                             n_samples=args.sgld_samples,
                             thin=args.sgld_thin)
        path = os.path.join(baseline_dir, "sgld_samples.pt")
        torch.save({
            "samples": samples,
            "linear_state_dict": clone_classifier_linear(model.classifier).state_dict(),
            "cls_dim": CLS_DIM,
            "num_classes": NUM_CLASSES,
        }, path)
        print(f"  Saved -> {path}")

    meta = {"methods": args.methods, "experiment": "E5_SST2"}
    with open(os.path.join(baseline_dir, "baselines_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. Artifacts in {baseline_dir}/")


if __name__ == "__main__":
    main()
