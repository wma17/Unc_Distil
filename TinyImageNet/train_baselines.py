"""
Train BNN baselines for TinyImageNet uncertainty quantification.

All baselines start from member_0 (first LoRA DeiT-S ensemble member).

Per user specification:
  1. MC Dropout  — Enable attention dropout in DeiT blocks; T=16 stochastic passes.
  2. EDL         — EDL head on frozen CLS features; fine-tune head only (50 epochs).
  3. LLLA        — Last-layer KFAC Laplace on prediction head only (no training).
  4. SGLD        — Last FC layer (model.head) only; backbone frozen; 200-step burn-in.

Usage:
    python train_baselines.py --save_dir ./checkpoints --gpu 0
    python train_baselines.py --save_dir ./checkpoints --gpu 0 --methods edl llla
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
from torch.amp import autocast
from torch.utils.data import DataLoader

from data import TinyImageNetDataset, get_train_transform, get_val_transform
from models import create_ensemble_member, load_saved_member_state, EMBED_DIM, NUM_CLASSES


EPS = 1e-8


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_train_loader(data_dir, batch_size, num_workers=4):
    ds = TinyImageNetDataset(data_dir, split="train",
                             transform=get_train_transform("basic"))
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)


def get_val_loader(data_dir, batch_size, num_workers=4):
    ds = TinyImageNetDataset(data_dir, split="val", transform=get_val_transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# Load member_0 helper
# ---------------------------------------------------------------------------

def load_member0(save_dir, device="cpu"):
    """Load first ensemble member using ensemble_configs.json if available."""
    cfg_path = os.path.join(save_dir, "ensemble_configs.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            configs = json.load(f)
        cfg = configs[0]
        model = create_ensemble_member(
            rank=cfg.get("rank", 8),
            alpha=cfg.get("alpha", None),
            lora_dropout=cfg.get("lora_dropout", 0.0),
            targets=cfg.get("targets", "qkv+proj"),
            unfreeze_blocks=cfg.get("unfreeze_blocks", 0),
            pretrained=True,
        )
    else:
        model = create_ensemble_member(pretrained=True)

    ckpt = torch.load(os.path.join(save_dir, "member_0.pt"),
                      map_location=device, weights_only=True)
    load_saved_member_state(model, ckpt, strict=False)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# 1. MC Dropout — enable attention dropout in DeiT blocks
# ---------------------------------------------------------------------------

def enable_attention_dropout(model, dropout_p=0.1):
    """Add dropout_p to every attention/proj layer in DeiT transformer blocks.

    Iterates over model.blocks; for each Attention sub-module sets attn_drop
    and proj_drop to dropout_p and creates Dropout modules if not present.
    This enables stochastic inference in train() mode.
    """
    for block in model.blocks:
        if hasattr(block, "attn"):
            attn = block.attn
            # timm attention has attn_drop and proj_drop as Dropout modules
            if hasattr(attn, "attn_drop"):
                attn.attn_drop = nn.Dropout(p=dropout_p)
            if hasattr(attn, "proj_drop"):
                attn.proj_drop = nn.Dropout(p=dropout_p)
    return model


def prepare_mc_dropout(save_dir, dropout_p=0.1, device="cpu"):
    model = load_member0(save_dir, device)
    model = enable_attention_dropout(model, dropout_p)
    # MC Dropout needs no fine-tuning — the attention dropout changes the
    # effective model distribution at test time.
    # We DO run a short fine-tune (5 epochs) to let BN-equivalent layers adapt.
    return model


def finetune_mc_dropout(model, train_loader, device, epochs=10, lr=5e-5):
    """Short fine-tune so the model adapts to the active dropout."""
    model.to(device).train()
    # Only fine-tune head and LoRA adapters; keep frozen backbone frozen
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        correct, total = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(imgs)
            F.cross_entropy(logits.float(), labels).backward()
            optimizer.step()
            with torch.no_grad():
                correct += logits.argmax(1).eq(labels).sum().item()
                total   += imgs.size(0)
        scheduler.step()
        if epoch % 2 == 0 or epoch == 1:
            print(f"  [MC Dropout] epoch {epoch}/{epochs}  acc={100*correct/total:.1f}%")
    return model


# ---------------------------------------------------------------------------
# 2. EDL head
# ---------------------------------------------------------------------------

class EDLHead(nn.Module):
    """Dirichlet evidential head on frozen CLS features."""

    def __init__(self, feat_dim=EMBED_DIM, hidden=256, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, feat):
        evidence = F.softplus(self.fc2(F.relu(self.fc1(feat))))
        return evidence + 1.0  # α_k ≥ 1


def edl_loss(alpha, labels, num_classes, lambda_kl=0.1):
    K   = num_classes
    S   = alpha.sum(dim=-1, keepdim=True)
    p   = alpha / S
    y   = F.one_hot(labels, K).float()

    l_err = ((y - p) ** 2).sum(dim=-1)
    l_var = (p * (1.0 - p) / (S + 1.0)).sum(dim=-1)

    alpha_tilde = y + (1.0 - y) * alpha
    S_tilde = alpha_tilde.sum(dim=-1)
    kl = (torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(float(K)))
          - torch.lgamma(alpha_tilde).sum(dim=-1)
          + ((alpha_tilde - 1.0) * (
              torch.digamma(alpha_tilde)
              - torch.digamma(S_tilde.unsqueeze(-1))
          )).sum(dim=-1))

    return (l_err + l_var + lambda_kl * kl).mean()


def _extract_cls_features(backbone, loader, device, max_batches=None):
    """Run backbone in eval/no_grad; return (CLS features, labels)."""
    backbone.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            imgs = imgs.to(device)
            with autocast("cuda"):
                # DeiT forward up to CLS token
                x = backbone.patch_embed(imgs)
                cls = backbone.cls_token.expand(x.size(0), -1, -1)
                x = torch.cat([cls, x], dim=1)
                x = backbone.pos_drop(x + backbone.pos_embed)
                x = backbone.blocks(x)
                x = backbone.norm(x)
                feat = x[:, 0]  # CLS token  (B, 384)
            all_feats.append(feat.float().cpu())
            all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


def train_edl(backbone, train_loader, device, epochs=50,
              lr=1e-3, lambda_kl=0.1, hidden=256):
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    edl_head = EDLHead(feat_dim=EMBED_DIM, hidden=hidden,
                       num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(edl_head.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("  [EDL] Extracting CLS features (this may take a few minutes)...")
    feats, labels_all = _extract_cls_features(backbone, train_loader, device)
    feat_loader = DataLoader(
        torch.utils.data.TensorDataset(feats, labels_all),
        batch_size=256, shuffle=True, num_workers=0)

    for epoch in range(1, epochs + 1):
        edl_head.train()
        total_loss, correct, total = 0.0, 0, 0
        for feat, labels in feat_loader:
            feat, labels = feat.to(device), labels.to(device)
            optimizer.zero_grad()
            alpha = edl_head(feat)
            loss  = edl_loss(alpha, labels, NUM_CLASSES, lambda_kl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feat.size(0)
            correct    += alpha.argmax(1).eq(labels).sum().item()
            total      += feat.size(0)
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [EDL] epoch {epoch}/{epochs}  loss={total_loss/total:.4f}"
                  f"  acc={100*correct/total:.1f}%")
    return edl_head


# ---------------------------------------------------------------------------
# 3. LLLA — KFAC on prediction head only
# ---------------------------------------------------------------------------

def train_llla(backbone, train_loader, device):
    try:
        from laplace import Laplace
    except ImportError:
        print("  [LLLA] laplace-torch not installed.")
        return None

    backbone.to(device).eval()

    # Wrap backbone so that only head is exposed to Laplace
    class DeiTWithHead(nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b
        def forward(self, x):
            x_e = self.b.patch_embed(x)
            cls = self.b.cls_token.expand(x_e.size(0), -1, -1)
            x_e = torch.cat([cls, x_e], dim=1)
            x_e = self.b.pos_drop(x_e + self.b.pos_embed)
            x_e = self.b.blocks(x_e)
            x_e = self.b.norm(x_e)
            return self.b.head(x_e[:, 0])

    wrapped = DeiTWithHead(backbone)
    print("  [LLLA] Fitting KFAC Laplace on last layer (prediction head)...")
    la = Laplace(wrapped, "classification",
                 subset_of_weights="last_layer",
                 hessian_structure="kron")
    la.fit(train_loader)
    print("  [LLLA] Optimizing prior precision...")
    la.optimize_prior_precision(method="marglik")
    print(f"  [LLLA] Prior precision: {la.prior_precision.item():.4f}")
    return la


# ---------------------------------------------------------------------------
# 4. SGLD — last FC layer only (backbone frozen)
# ---------------------------------------------------------------------------

def train_sgld(backbone, train_loader, device,
               step_size=1e-6, burn_in=200, n_samples=16, thin=10,
               prior_sigma=1.0):
    """SGLD on backbone.head (nn.Linear(384, 200)) only; backbone frozen."""
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    fc_weight = backbone.head.weight.detach().clone().requires_grad_(True)
    fc_bias   = backbone.head.bias.detach().clone().requires_grad_(True)
    params    = [fc_weight, fc_bias]

    N = len(train_loader.dataset)
    total_steps = burn_in + n_samples * thin
    data_iter   = iter(train_loader)
    samples     = []
    t0 = time.time()

    for step in range(total_steps):
        try:
            imgs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            imgs, labels = next(data_iter)
        imgs, labels = imgs.to(device), labels.to(device)

        # Extract CLS features (frozen backbone)
        with torch.no_grad():
            with autocast("cuda"):
                x_e = backbone.patch_embed(imgs)
                cls = backbone.cls_token.expand(x_e.size(0), -1, -1)
                x_e = torch.cat([cls, x_e], dim=1)
                x_e = backbone.pos_drop(x_e + backbone.pos_embed)
                x_e = backbone.blocks(x_e)
                x_e = backbone.norm(x_e)
                feat = x_e[:, 0].float()  # (B, 384)

        logits   = feat @ fc_weight.T + fc_bias
        loss_ce  = F.cross_entropy(logits, labels)
        loss_reg = (fc_weight.pow(2).sum() + fc_bias.pow(2).sum()) / (2 * prior_sigma ** 2)
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
            samples.append({
                "weight": fc_weight.detach().cpu().clone(),
                "bias":   fc_bias.detach().cpu().clone(),
            })

        if step % 50 == 0:
            print(f"  [SGLD] step {step}/{total_steps}  "
                  f"U={U.item():.3f}  samples_collected={len(samples)}")

    print(f"  [SGLD] Done. {len(samples)} samples in {time.time()-t0:.1f}s")
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir",     default="./checkpoints")
    p.add_argument("--data_dir",     default="/home/maw6/maw6/unc_regression/data")
    p.add_argument("--gpu",          type=int,   default=0)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--methods", nargs="+",
                   default=["mc_dropout", "edl", "llla", "sgld"],
                   choices=["mc_dropout", "edl", "llla", "sgld"])
    # MC Dropout
    p.add_argument("--mc_epochs",    type=int,   default=10)
    p.add_argument("--mc_lr",        type=float, default=5e-5)
    p.add_argument("--mc_dropout_p", type=float, default=0.1)
    # EDL
    p.add_argument("--edl_epochs",   type=int,   default=50)
    p.add_argument("--edl_lr",       type=float, default=1e-3)
    p.add_argument("--edl_lambda_kl",type=float, default=0.1)
    # SGLD
    p.add_argument("--sgld_step",    type=float, default=1e-6)
    p.add_argument("--sgld_burn_in", type=int,   default=200)
    p.add_argument("--sgld_samples", type=int,   default=16)
    p.add_argument("--sgld_thin",    type=int,   default=10)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    baseline_dir = os.path.join(args.save_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)

    member0_path = os.path.join(args.save_dir, "member_0.pt")
    if not os.path.isfile(member0_path):
        raise FileNotFoundError(f"member_0.pt not found in {args.save_dir}")

    train_loader = get_train_loader(args.data_dir, args.batch_size, args.num_workers)

    # ---- MC Dropout ----
    if "mc_dropout" in args.methods:
        print("\n=== MC Dropout ===")
        mc_model = prepare_mc_dropout(args.save_dir, args.mc_dropout_p, device)
        mc_model  = finetune_mc_dropout(mc_model, train_loader, device,
                                        epochs=args.mc_epochs, lr=args.mc_lr)
        out_path = os.path.join(baseline_dir, "mc_dropout.pt")
        # Save LoRA-head state (same format as members)
        from models import member_state_dict
        torch.save({
            "member_state":  member_state_dict(mc_model.cpu()),
            "dropout_p": args.mc_dropout_p,
        }, out_path)
        print(f"  Saved -> {out_path}")

    # ---- EDL ----
    if "edl" in args.methods:
        print("\n=== Evidential Deep Learning ===")
        backbone = load_member0(args.save_dir, device)
        edl_head = train_edl(backbone, train_loader, device,
                             epochs=args.edl_epochs, lr=args.edl_lr,
                             lambda_kl=args.edl_lambda_kl)
        out_path = os.path.join(baseline_dir, "edl_head.pt")
        from models import member_state_dict
        torch.save({
            "backbone_member_state": member_state_dict(backbone.cpu()),
            "edl_head_state_dict":   edl_head.cpu().state_dict(),
        }, out_path)
        print(f"  Saved -> {out_path}")

    # ---- LLLA ----
    if "llla" in args.methods:
        print("\n=== Last-Layer Laplace (KFAC) ===")
        backbone = load_member0(args.save_dir, device)
        la = train_llla(backbone, train_loader, device)
        if la is not None:
            out_path = os.path.join(baseline_dir, "llla.pt")
            torch.save(la.state_dict(), out_path)
            from models import member_state_dict
            torch.save({"backbone_member_state": member_state_dict(backbone.cpu())},
                       os.path.join(baseline_dir, "llla_backbone.pt"))
            print(f"  Saved -> {out_path}")

    # ---- SGLD ----
    if "sgld" in args.methods:
        print("\n=== SGLD (last layer) ===")
        backbone = load_member0(args.save_dir, device)
        samples  = train_sgld(backbone, train_loader, device,
                              step_size=args.sgld_step,
                              burn_in=args.sgld_burn_in,
                              n_samples=args.sgld_samples,
                              thin=args.sgld_thin)
        out_path = os.path.join(baseline_dir, "sgld_samples.pt")
        from models import member_state_dict
        torch.save({
            "backbone_member_state": member_state_dict(backbone.cpu()),
            "samples": samples,
            "n_samples": len(samples),
        }, out_path)
        print(f"  Saved -> {out_path}  ({len(samples)} samples)")

    print(f"\nAll done. Baselines saved in {baseline_dir}/")


if __name__ == "__main__":
    main()
