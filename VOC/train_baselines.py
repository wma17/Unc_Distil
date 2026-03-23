"""
Train UQ baselines for Pascal VOC (E4), starting from ensemble member_0 (SegFormer LoRA).

  1. MC Dropout  — increase nn.Dropout p across the SegFormer; short fine-tune.
  2. EDL         — 1×1 evidential head on frozen decode features (Dirichlet α per class).
  3. LLLA        — diagonal Laplace on the last 1×1 conv (equivalent linear 512→21).
  4. SGLD        — last-layer SGLD on that same linear head.

The decode head’s final 1×1 conv is treated as a shared linear over spatial positions,
matching the CIFAR “last layer” Laplace / SGLD setup.

Usage:
    python train_baselines.py --save_dir ./checkpoints --data_dir ../data --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models import create_teacher, NUM_CLASSES
from data import VOCSegDataset, SegTransformTrain, SegTransformVal, IGNORE_INDEX


EPS = 1e-8


# ---------------------------------------------------------------------------
# Fused features (input to decode_head.classifier)
# ---------------------------------------------------------------------------

def forward_fused_and_logits(model: nn.Module, pixel_values: torch.Tensor):
    """Returns (fused, logits) with fused (B, C, h, w) and logits (B, num_classes, H', W')."""
    fused_box: list = []

    def _pre_hook(_m, inputs):
        fused_box.append(inputs[0])

    h = model.segformer.decode_head.classifier.register_forward_pre_hook(_pre_hook)
    logits = model(pixel_values)
    h.remove()
    fused = fused_box[0]
    return fused, logits


# ---------------------------------------------------------------------------
# Load member 0
# ---------------------------------------------------------------------------

def load_member0(save_dir: str, device: torch.device) -> nn.Module:
    path = os.path.join(save_dir, "member_0.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mcfg = ckpt.get("member_config", {})
    model = create_teacher(
        num_classes=NUM_CLASSES,
        rank=mcfg.get("rank", 16),
        alpha=mcfg.get("alpha", 32.0),
        init_scale=1.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model


def set_dropout_p(module: nn.Module, p: float) -> None:
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.p = p


# ---------------------------------------------------------------------------
# MC Dropout fine-tune
# ---------------------------------------------------------------------------

def train_mc_dropout(model, train_loader, device, dropout_p: float = 0.1,
                     iterations: int = 4000, lr: float = 6e-5):
    model.to(device)
    set_dropout_p(model, dropout_p)
    params = list(model.trainable_parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    model.train()
    it = 0
    t0 = time.time()
    while it < iterations:
        for imgs, masks in train_loader:
            if it >= iterations:
                break
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            lab = F.interpolate(masks.float().unsqueeze(1), size=logits.shape[2:],
                                mode="nearest").long().squeeze(1)
            loss = F.cross_entropy(logits, lab, ignore_index=IGNORE_INDEX)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            it += 1
            if it % 500 == 0:
                print(f"  [MC Dropout] iter {it}/{iterations}  loss={loss.item():.4f}")
    print(f"  [MC Dropout] done in {time.time()-t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# EDL (1×1 conv head on fused features)
# ---------------------------------------------------------------------------

def edl_loss_seg(alpha: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
                 num_classes: int, lambda_kl: float):
    """alpha (B,K,H,W), labels (B,H,W) int64, mask bool (B,H,W) valid pixels."""
    B, K, H, W = alpha.shape
    alpha_f = alpha.permute(0, 2, 3, 1).reshape(-1, K)
    y_f = labels.reshape(-1)
    m_f = mask.reshape(-1)
    if m_f.sum() == 0:
        return alpha.sum() * 0.0
    alpha_v = alpha_f[m_f]
    y_v = y_f[m_f]

    S = alpha_v.sum(dim=-1, keepdim=True)
    p = alpha_v / S
    y_oh = F.one_hot(y_v, num_classes).float()
    l_err = ((y_oh - p) ** 2).sum(dim=-1)
    l_var = (p * (1.0 - p) / (S + 1.0)).sum(dim=-1)
    alpha_tilde = y_oh + (1.0 - y_oh) * alpha_v
    S_tilde = alpha_tilde.sum(dim=-1)
    kl = (torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(float(num_classes), device=alpha.device))
          - torch.lgamma(alpha_tilde).sum(dim=-1)
          + ((alpha_tilde - 1.0) * (
              torch.digamma(alpha_tilde) - torch.digamma(S_tilde.unsqueeze(-1))
          )).sum(dim=-1))
    return (l_err + l_var + lambda_kl * kl).mean()


def train_edl(model, train_loader, device, in_ch: int, epochs: int = 15,
              lr: float = 1e-3, lambda_kl: float = 0.1):
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    edl_conv = nn.Conv2d(in_ch, NUM_CLASSES, kernel_size=1, bias=True).to(device)
    opt = optim.Adam(edl_conv.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        edl_conv.train()
        tot, n = 0.0, 0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                fused, _ = forward_fused_and_logits(model, imgs)
            fused = fused.detach()
            lab = F.interpolate(masks.float().unsqueeze(1), size=fused.shape[2:],
                                mode="nearest").long().squeeze(1)
            valid = lab != IGNORE_INDEX
            alpha = F.softplus(edl_conv(fused)) + 1.0
            loss = edl_loss_seg(alpha, lab, valid, NUM_CLASSES, lambda_kl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * imgs.size(0)
            n += imgs.size(0)
        print(f"  [EDL] epoch {epoch}/{epochs}  loss={tot/max(n,1):.4f}")
    return edl_conv


# ---------------------------------------------------------------------------
# Last conv ↔ Linear + pixel dataset
# ---------------------------------------------------------------------------

def conv1x1_to_linear(conv: nn.Conv2d) -> nn.Linear:
    assert conv.kernel_size == (1, 1)
    in_ch = conv.in_channels
    lin = nn.Linear(in_ch, conv.out_channels, bias=conv.bias is not None)
    lin.weight.data.copy_(conv.weight.data.squeeze(-1).squeeze(-1))
    if conv.bias is not None:
        lin.bias.data.copy_(conv.bias.data)
    return lin


@torch.no_grad()
def collect_decode_pixels(model, loader, device, max_pixels: int = 200_000):
    model.eval()
    feats_all, y_all = [], []
    count = 0
    for imgs, masks in loader:
        imgs = imgs.to(device)
        fused, _ = forward_fused_and_logits(model, imgs)
        B, C, H, W = fused.shape
        fused_flat = fused.permute(0, 2, 3, 1).reshape(-1, C)
        lab = F.interpolate(masks.float().unsqueeze(1).to(device), size=(H, W),
                            mode="nearest").long().squeeze(1).reshape(-1)
        valid = lab != IGNORE_INDEX
        ff = fused_flat[valid].cpu()
        yy = lab[valid].cpu()
        feats_all.append(ff)
        y_all.append(yy)
        count += ff.size(0)
        if count >= max_pixels:
            break
    X = torch.cat(feats_all, 0)[:max_pixels]
    y = torch.cat(y_all, 0)[:max_pixels]
    return X, y


# ---------------------------------------------------------------------------
# LLLA / SGLD on linear head
# ---------------------------------------------------------------------------

def train_llla(linear_head: nn.Linear, X: torch.Tensor, y: torch.Tensor, device,
               batch_size: int = 4096):
    try:
        from laplace import Laplace
    except ImportError:
        print("  [LLLA] laplace-torch not installed. pip install laplace-torch")
        return None

    linear_head = linear_head.to(device).eval()
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    la = Laplace(linear_head, "classification", subset_of_weights="all",
                 hessian_structure="diag")
    print("  [LLLA] Fitting diagonal Laplace on decode 1×1 conv (as linear)...")
    la.fit(loader)
    print("  [LLLA] Optimizing prior precision...")
    la.optimize_prior_precision(method="marglik")
    print(f"  [LLLA] Prior precision: {float(la.prior_precision.detach().cpu()):.6f}")
    return la


def train_sgld(linear_head: nn.Linear, X: torch.Tensor, y: torch.Tensor, device,
               step_size: float = 1e-7, burn_in: int = 200, n_samples: int = 16,
               thin: int = 10, prior_sigma: float = 1.0):
    N = X.size(0)
    W = linear_head.weight.detach().clone().requires_grad_(True)
    B = linear_head.bias.detach().clone().requires_grad_(True)
    params = [W, B]
    bs = min(4096, N)
    total_steps = burn_in + n_samples * thin
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
            print(f"  [SGLD] step {step}/{total_steps}  U={U.item():.2f}  n={len(samples)}")
    print(f"  [SGLD] Done. {len(samples)} samples in {time.time()-t0:.1f}s")
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train VOC SegFormer baselines (E4)")
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--data_dir", type=str, default="../data")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--methods", nargs="+",
                   default=["mc_dropout", "edl", "llla", "sgld"],
                   choices=["mc_dropout", "edl", "llla", "sgld"])
    p.add_argument("--mc_dropout_p", type=float, default=0.1)
    p.add_argument("--mc_iters", type=int, default=4000)
    p.add_argument("--edl_epochs", type=int, default=15)
    p.add_argument("--edl_lr", type=float, default=1e-3)
    p.add_argument("--edl_lambda_kl", type=float, default=0.1)
    p.add_argument("--max_llla_pixels", type=int, default=200_000)
    p.add_argument("--sgld_step", type=float, default=1e-7)
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

    train_tf = SegTransformTrain(crop_size=512, aug_mode="scale")
    train_ds = VOCSegDataset(args.data_dir, split="train", transform=train_tf, use_sbd=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model = load_member0(args.save_dir, device)
    clf = model.segformer.decode_head.classifier
    in_ch = clf.in_channels

    if "mc_dropout" in args.methods:
        print("\n=== MC Dropout ===")
        model = train_mc_dropout(model, train_loader, device,
                                 dropout_p=args.mc_dropout_p, iterations=args.mc_iters)
        path = os.path.join(baseline_dir, "mc_dropout.pt")
        torch.save({
            "model_state_dict": model.cpu().trainable_state_dict(),
            "dropout_p": args.mc_dropout_p,
        }, path)
        print(f"  Saved -> {path}")

    model = load_member0(args.save_dir, device)

    if "edl" in args.methods:
        print("\n=== EDL ===")
        edl_conv = train_edl(model, train_loader, device, in_ch,
                             epochs=args.edl_epochs, lr=args.edl_lr,
                             lambda_kl=args.edl_lambda_kl)
        path = os.path.join(baseline_dir, "edl_head.pt")
        torch.save({
            "edl_conv_state_dict": edl_conv.cpu().state_dict(),
            "in_channels": in_ch,
        }, path)
        print(f"  Saved -> {path}")

    model = load_member0(args.save_dir, device)
    print(f"\n  Collecting up to {args.max_llla_pixels} labeled decode pixels...")
    X, y = collect_decode_pixels(model, train_loader, device, max_pixels=args.max_llla_pixels)
    lin = conv1x1_to_linear(model.segformer.decode_head.classifier)

    if "llla" in args.methods:
        print("\n=== LLLA ===")
        la = train_llla(lin, X, y, device)
        if la is not None:
            path = os.path.join(baseline_dir, "llla.pt")
            torch.save({
                "laplace_state_dict": la.state_dict(),
                "linear_state_dict": la.model.cpu().state_dict(),
                "in_channels": in_ch,
                "num_classes": NUM_CLASSES,
            }, path)
            print(f"  Saved -> {path}")

    if "sgld" in args.methods:
        print("\n=== SGLD ===")
        lin_sg = conv1x1_to_linear(load_member0(args.save_dir, device).segformer.decode_head.classifier)
        samples = train_sgld(lin_sg, X, y, device,
                             step_size=args.sgld_step,
                             burn_in=args.sgld_burn_in,
                             n_samples=args.sgld_samples,
                             thin=args.sgld_thin)
        path = os.path.join(baseline_dir, "sgld_samples.pt")
        torch.save({
            "samples": samples,
            "linear_state_dict": lin_sg.state_dict(),
            "in_channels": in_ch,
            "num_classes": NUM_CLASSES,
        }, path)
        print(f"  Saved -> {path}")

    with open(os.path.join(baseline_dir, "baselines_meta.json"), "w") as f:
        json.dump({"methods": args.methods, "experiment": "E4_VOC"}, f, indent=2)
    print(f"\nDone. Artifacts in {baseline_dir}/")


if __name__ == "__main__":
    main()
