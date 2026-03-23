"""
Evaluate VOC (E4) SegFormer baselines against cached teacher ensemble EU maps.

Requires:
    save_dir/teacher_targets.npz  (val_eu, etc.)
    save_dir/baselines/*.pt       from train_baselines.py

Usage:
    python evaluate_baselines.py --save_dir ./checkpoints --data_dir ../data --gpu 0
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import create_teacher, NUM_CLASSES
from data import VOCSegDataset, SegTransformVal, IGNORE_INDEX
from train_baselines import forward_fused_and_logits, set_dropout_p
from evaluate_student import (
    pearson_corr, spearman_corr, compute_miou, compute_pixel_ece, compute_pixel_nll,
)


EPS = 1e-8


def load_member0(save_dir: str, device: torch.device) -> nn.Module:
    path = os.path.join(save_dir, "member_0.pt")
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


def pixel_mi_from_prob_stack(probs_stack: torch.Tensor) -> torch.Tensor:
    """probs_stack (T, N, K) -> EU (N,)"""
    mean_p = probs_stack.mean(dim=0)
    h_mean = -(mean_p * torch.log(mean_p.clamp_min(EPS))).sum(dim=-1)
    h_each = -(probs_stack * torch.log(probs_stack.clamp_min(EPS))).sum(dim=-1)
    mean_h = h_each.mean(dim=0)
    return (h_mean - mean_h).clamp(min=0.0)


@torch.no_grad()
def predict_mc_maps(model, loader, device, T: int, dropout_p: float, eu_map_size: int):
    model.to(device)
    set_dropout_p(model, dropout_p)
    model.train()
    all_eu, all_probs, all_masks = [], [], []
    for imgs, masks in loader:
        imgs = imgs.to(device)
        B = imgs.size(0)
        probs_samples = []
        for _ in range(T):
            logits = model(imgs)
            logits_s = F.interpolate(logits, size=(eu_map_size, eu_map_size),
                                      mode="bilinear", align_corners=False)
            probs_samples.append(F.softmax(logits_s, dim=1))
        stacked = torch.stack(probs_samples, dim=0)  # T,B,C,H,W
        Tt, Bb, Cc, Hh, Ww = stacked.shape
        x = stacked.permute(0, 1, 3, 4, 2).reshape(Tt, Bb * Hh * Ww, Cc)
        eu_flat = pixel_mi_from_prob_stack(x)
        eu_maps = eu_flat.view(Bb, Hh, Ww)
        mean_probs = x.mean(dim=0).view(Bb, Hh, Ww, Cc).permute(0, 3, 1, 2)
        all_eu.append(eu_maps.cpu())
        all_probs.append(mean_probs.cpu())
        all_masks.append(masks.numpy())
    return all_eu, all_probs, all_masks


@torch.no_grad()
def predict_edl_maps(model, edl_conv, loader, device, eu_map_size: int):
    model.eval()
    edl_conv.eval()
    model.to(device)
    edl_conv.to(device)
    all_eu, all_probs, all_masks = [], [], []
    for imgs, masks in loader:
        imgs = imgs.to(device)
        fused, _ = forward_fused_and_logits(model, imgs)
        alpha = F.softplus(edl_conv(fused)) + 1.0
        alpha_s = F.interpolate(alpha, size=(eu_map_size, eu_map_size),
                                mode="bilinear", align_corners=False)
        S = alpha_s.sum(dim=1, keepdim=True)
        p_hat = alpha_s / S
        u_ale = (torch.digamma(S + 1).squeeze(1)
                 - (alpha_s / S * torch.digamma(alpha_s + 1)).sum(dim=1))
        h_p = -(p_hat * torch.log(p_hat.clamp_min(EPS))).sum(dim=1)
        u_epi = (h_p - u_ale).clamp(min=0.0)
        all_eu.append(u_epi.cpu())
        all_probs.append(p_hat.cpu())
        all_masks.append(masks.numpy())
    return all_eu, all_probs, all_masks


@torch.no_grad()
def predict_sgld_maps(model, samples, loader, device, eu_map_size: int):
    model.eval()
    model.to(device)
    all_eu, all_probs, all_masks = [], [], []
    for imgs, masks in loader:
        imgs = imgs.to(device)
        fused, _ = forward_fused_and_logits(model, imgs)
        B, C, H, W = fused.shape
        fused_s = F.interpolate(fused, size=(eu_map_size, eu_map_size),
                                mode="bilinear", align_corners=False)
        Hs, Ws = fused_s.shape[2], fused_s.shape[3]
        ff = fused_s.permute(0, 2, 3, 1).reshape(-1, C)
        prob_samples = []
        for s in samples:
            w = s["weight"].to(device)
            b = s["bias"].to(device)
            logits_f = ff @ w.T + b
            prob_samples.append(F.softmax(logits_f, dim=-1))
        stacked = torch.stack(prob_samples, dim=0)
        eu_flat = pixel_mi_from_prob_stack(stacked)
        eu_maps = eu_flat.view(B, Hs, Ws)
        mean_probs = stacked.mean(dim=0).view(B, Hs, Ws, NUM_CLASSES).permute(0, 3, 1, 2)
        all_eu.append(eu_maps.cpu())
        all_probs.append(mean_probs.cpu())
        all_masks.append(masks.numpy())
    return all_eu, all_probs, all_masks


@torch.no_grad()
def predict_llla_maps(model, la, loader, device, eu_map_size: int, T: int):
    try:
        from laplace import Laplace
    except ImportError:
        print("  [LLLA] laplace-torch not installed.")
        return None, None, None

    model.eval()
    la.model.to(device)
    all_eu, all_probs, all_masks = [], [], []
    for imgs, masks in loader:
        imgs = imgs.to(device)
        fused, _ = forward_fused_and_logits(model, imgs)
        B, C, H, W = fused.shape
        fused_s = F.interpolate(fused, size=(eu_map_size, eu_map_size),
                                mode="bilinear", align_corners=False)
        Hs, Ws = fused_s.shape[2], fused_s.shape[3]
        ff = fused_s.permute(0, 2, 3, 1).reshape(-1, C)
        bs = 8192
        logit_chunks = []
        for i in range(0, ff.size(0), bs):
            batch = ff[i:i + bs].to(device)
            logit_samps = la.predictive_samples(batch, pred_type="glm", n_samples=T)
            logit_chunks.append(logit_samps.cpu())
        stacked_logits = torch.cat(logit_chunks, dim=1)
        stacked = F.softmax(stacked_logits, dim=-1)
        eu_flat = pixel_mi_from_prob_stack(stacked)
        eu_maps = eu_flat.view(B, Hs, Ws)
        mean_probs = stacked.mean(dim=0).view(B, Hs, Ws, NUM_CLASSES).permute(0, 3, 1, 2)
        all_eu.append(eu_maps.cpu())
        all_probs.append(mean_probs.cpu())
        all_masks.append(masks.numpy())
    return all_eu, all_probs, all_masks


def summarize_run(name, all_eu, all_probs, all_masks, teacher_val_eu_np: np.ndarray):
    """Print mIoU, pixel ECE/NLL, EU correlation vs teacher (flattened maps)."""
    pred_mask_pairs = []
    prob_rows = []
    label_rows = []
    for pr, m in zip(all_probs, all_masks):
        pn = pr.numpy()
        m_np = m
        Hm, Wm = m_np.shape[1], m_np.shape[2]
        for bi in range(pn.shape[0]):
            pr_b = pn[bi]
            p_up = F.interpolate(torch.from_numpy(pr_b).unsqueeze(0), size=(Hm, Wm),
                                 mode="bilinear", align_corners=False).squeeze(0).numpy()
            pred_mask_pairs.append((p_up.argmax(0), m_np[bi]))
            prob_rows.append(p_up.reshape(NUM_CLASSES, -1).T)
            label_rows.append(m_np[bi].reshape(-1))

    mious = [compute_miou(p, mm) for p, mm in pred_mask_pairs]
    mean_miou = float(np.mean(mious))

    probs_flat = np.concatenate(prob_rows, axis=0)
    labels_flat = np.concatenate(label_rows, axis=0)
    ece = compute_pixel_ece(probs_flat, labels_flat)
    nll = compute_pixel_nll(probs_flat, labels_flat)

    stu_eu_flat = torch.from_numpy(
        np.concatenate([e.numpy().reshape(-1) for e in all_eu], axis=0).astype(np.float32))
    tea_eu_flat = torch.from_numpy(teacher_val_eu_np.astype(np.float32).reshape(-1))
    n_min = min(stu_eu_flat.numel(), tea_eu_flat.numel())
    rp = pearson_corr(stu_eu_flat[:n_min], tea_eu_flat[:n_min])
    rs = spearman_corr(stu_eu_flat[:n_min], tea_eu_flat[:n_min])

    print(f"  {name:<18} mIoU={mean_miou:.4f}  pix_ECE={ece:.4f}  pix_NLL={nll:.4f}  "
          f"EU_P={rp:.4f}  EU_S={rs:.4f}")
    return mean_miou


def main():
    p = argparse.ArgumentParser(description="Evaluate VOC baselines (E4)")
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--data_dir", type=str, default="../data")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--methods", nargs="+",
                   default=["mc_dropout", "edl", "llla", "sgld"],
                   choices=["mc_dropout", "edl", "llla", "sgld"])
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    targets_path = os.path.join(args.save_dir, "teacher_targets.npz")
    if not os.path.isfile(targets_path):
        raise FileNotFoundError(targets_path)

    data = np.load(targets_path, allow_pickle=True)
    eu_map_size = int(data["eu_map_size"]) if "eu_map_size" in data else 128
    teacher_val_eu = data["val_eu"]

    val_tf = SegTransformVal(crop_size=512)
    val_ds = VOCSegDataset(args.data_dir, split="val", transform=val_tf, use_sbd=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    baseline_dir = os.path.join(args.save_dir, "baselines")

    print(f"Device: {device}   EU map size: {eu_map_size}")

    if "mc_dropout" in args.methods:
        path = os.path.join(baseline_dir, "mc_dropout.pt")
        if os.path.isfile(path):
            print("\n[MC Dropout]")
            ckpt = torch.load(path, map_location=device, weights_only=False)
            model = load_member0(args.save_dir, device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            dp = ckpt.get("dropout_p", 0.1)
            eu, pr, ms = predict_mc_maps(model, val_loader, device, args.T, dp, eu_map_size)
            summarize_run("MC Dropout", eu, pr, ms, teacher_val_eu)
        else:
            print(f"\n[MC Dropout] missing {path}")

    if "edl" in args.methods:
        path = os.path.join(baseline_dir, "edl_head.pt")
        if os.path.isfile(path):
            print("\n[EDL]")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = load_member0(args.save_dir, device)
            in_ch = ckpt.get("in_channels", model.segformer.decode_head.classifier.in_channels)
            edl_conv = nn.Conv2d(in_ch, NUM_CLASSES, kernel_size=1, bias=True)
            edl_conv.load_state_dict(ckpt["edl_conv_state_dict"])
            eu, pr, ms = predict_edl_maps(model, edl_conv, val_loader, device, eu_map_size)
            summarize_run("EDL", eu, pr, ms, teacher_val_eu)
        else:
            print(f"\n[EDL] missing {path}")

    if "llla" in args.methods:
        path = os.path.join(baseline_dir, "llla.pt")
        if os.path.isfile(path):
            print("\n[LLLA]")
            try:
                from laplace import Laplace
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                lin = nn.Linear(ckpt["in_channels"], NUM_CLASSES)
                lin.load_state_dict(ckpt["linear_state_dict"])
                la = Laplace(lin, "classification", subset_of_weights="all",
                             hessian_structure="diag")
                la.load_state_dict(ckpt["laplace_state_dict"])
                model = load_member0(args.save_dir, device)
                out = predict_llla_maps(model, la, val_loader, device, eu_map_size, args.T)
                if out[0] is not None:
                    summarize_run("LLLA", *out, teacher_val_eu)
            except Exception as e:
                print(f"  [LLLA] failed: {e}")
        else:
            print(f"\n[LLLA] missing {path}")

    if "sgld" in args.methods:
        path = os.path.join(baseline_dir, "sgld_samples.pt")
        if os.path.isfile(path):
            print("\n[SGLD]")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = load_member0(args.save_dir, device)
            eu, pr, ms = predict_sgld_maps(model, ckpt["samples"], val_loader, device, eu_map_size)
            summarize_run("SGLD", eu, pr, ms, teacher_val_eu)
        else:
            print(f"\n[SGLD] missing {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
