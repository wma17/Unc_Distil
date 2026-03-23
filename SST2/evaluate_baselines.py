"""
Evaluate SST-2 (E5) UQ baselines vs cached teacher ensemble targets.

Expects:
    save_dir/teacher_targets.npz   (dev_eu, dev_probs, dev_labels, ...)
    save_dir/baselines/*.pt        from train_baselines.py

Usage:
    python evaluate_baselines.py --save_dir ./checkpoints --gpu 0
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import create_teacher
from data import load_sst2, collate_fn
from train_baselines import EDLHead, CLS_DIM, NUM_CLASSES, set_dropout_p


EPS = 1e-8


def mutual_information(probs_stack: np.ndarray):
    mean_probs = probs_stack.mean(axis=0)
    h_mean = -(mean_probs * np.log(mean_probs + EPS)).sum(axis=-1)
    h_each = -(probs_stack * np.log(probs_stack + EPS)).sum(axis=-1)
    mean_h = h_each.mean(axis=0)
    eu = np.maximum(h_mean - mean_h, 0.0)
    return eu, mean_probs


def pearson_corr(a, b):
    a, b = a.float(), b.float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def spearman_corr(a, b):
    def _rank(x):
        order = x.argsort()
        ranks = torch.empty_like(x)
        ranks[order] = torch.arange(len(x), dtype=x.dtype)
        return ranks
    a, b = a.float(), b.float()
    return torch.corrcoef(torch.stack([_rank(a), _rank(b)]))[0, 1].item()


def compute_ece(probs_np, labels_np, n_bins=15):
    confidences = probs_np.max(axis=-1)
    predictions = probs_np.argmax(axis=-1)
    accuracies = (predictions == labels_np).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() > 0:
            ece += (in_bin.sum() / len(labels_np)) * abs(
                accuracies[in_bin].mean() - confidences[in_bin].mean())
    return ece


def compute_nll(probs_np, labels_np):
    n = len(labels_np)
    return -np.log(probs_np[np.arange(n), labels_np] + EPS).mean()


def compute_brier(probs_np, labels_np):
    n = len(labels_np)
    one_hot = np.zeros_like(probs_np)
    one_hot[np.arange(n), labels_np] = 1.0
    return ((probs_np - one_hot) ** 2).sum(axis=-1).mean()


def load_teacher_from_ckpt(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ckpt.get("member_config", {})
    model = create_teacher(
        num_classes=NUM_CLASSES,
        rank=mcfg.get("rank", 8),
        alpha=mcfg.get("alpha", 16.0),
        attention_dropout=mcfg.get("attention_dropout", 0.1),
        init_scale=1.0,
    ).to(device)
    key = "model_state_dict" if "model_state_dict" in ckpt else None
    if key is None and "trainable_state_dict" in ckpt:
        key = "trainable_state_dict"
    model.load_state_dict(ckpt[key], strict=False)
    return model


@torch.no_grad()
def bert_cls(model, input_ids, attention_mask):
    out = model.bert(input_ids=input_ids, attention_mask=attention_mask)
    return out.last_hidden_state[:, 0]


@torch.no_grad()
def predict_mc_dropout(model, loader, device, T: int, dropout_p: float):
    model.to(device)
    set_dropout_p(model, dropout_p)
    model.train()
    all_ids, all_mask, all_labels = [], [], []
    for batch in loader:
        all_ids.append(batch["input_ids"])
        all_mask.append(batch["attention_mask"])
        all_labels.append(batch["label"])
    input_ids = torch.cat(all_ids, 0).to(device)
    attention_mask = torch.cat(all_mask, 0).to(device)
    labels = torch.cat(all_labels, 0).numpy()

    sample_probs = []
    for _ in range(T):
        probs_b = []
        bs = 64
        for i in range(0, input_ids.size(0), bs):
            logits = model(input_ids[i:i + bs], attention_mask[i:i + bs])
            probs_b.append(F.softmax(logits, dim=-1).cpu().numpy())
        sample_probs.append(np.concatenate(probs_b, axis=0))
    probs_stack = np.stack(sample_probs, axis=0)
    eu, mean_probs = mutual_information(probs_stack)
    return eu, mean_probs, labels


@torch.no_grad()
def predict_edl(model, edl_head, loader, device):
    model.eval()
    edl_head.eval()
    model.to(device)
    edl_head.to(device)
    eu_epi_l, eu_ale_l, probs_l, labels_l = [], [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]
        feat = bert_cls(model, input_ids, attention_mask)
        alpha = edl_head(feat)
        S = alpha.sum(dim=-1, keepdim=True)
        p_hat = (alpha / S).cpu()
        u_ale = (torch.digamma(S + 1)
                 - (alpha / S * torch.digamma(alpha + 1)).sum(dim=-1, keepdim=True)
                 ).squeeze(-1).cpu()
        h_p = -(p_hat * torch.log(p_hat + EPS)).sum(dim=-1)
        u_epi = (h_p - u_ale).clamp(min=0.0)
        eu_epi_l.append(u_epi.numpy())
        eu_ale_l.append(u_ale.numpy())
        probs_l.append(p_hat.numpy())
        labels_l.append(labels.numpy())
    return (np.concatenate(eu_epi), np.concatenate(eu_ale),
            np.concatenate(probs_l), np.concatenate(labels_l))


@torch.no_grad()
def predict_llla(model, la, loader, device, T: int):
    try:
        from laplace import Laplace
    except ImportError:
        print("  [LLLA] laplace-torch not installed.")
        return None, None, None

    model.to(device).eval()
    la.model.to(device)

    all_ids, all_mask, all_labels = [], [], []
    for batch in loader:
        all_ids.append(batch["input_ids"])
        all_mask.append(batch["attention_mask"])
        all_labels.append(batch["label"])
    input_ids = torch.cat(all_ids, 0).to(device)
    attention_mask = torch.cat(all_mask, 0).to(device)
    labels = torch.cat(all_labels, 0).numpy()

    feat_chunks = []
    bs = 64
    for i in range(0, input_ids.size(0), bs):
        feat_chunks.append(
            bert_cls(model, input_ids[i:i + bs], attention_mask[i:i + bs]).cpu())
    feats = torch.cat(feat_chunks, dim=0)

    sample_probs = []
    for i in range(0, feats.size(0), bs):
        batch = feats[i:i + bs].to(device)
        logit_samps = la.predictive_samples(batch, pred_type="glm", n_samples=T)
        sample_probs.append(F.softmax(logit_samps, dim=-1).cpu().numpy())
    probs_stack = np.concatenate(sample_probs, axis=1)
    eu, mean_probs = mutual_information(probs_stack)
    return eu, mean_probs, labels


@torch.no_grad()
def predict_sgld(model, samples, loader, device):
    model.to(device).eval()
    all_ids, all_mask, all_labels = [], [], []
    for batch in loader:
        all_ids.append(batch["input_ids"])
        all_mask.append(batch["attention_mask"])
        all_labels.append(batch["label"])
    input_ids = torch.cat(all_ids, 0).to(device)
    attention_mask = torch.cat(all_mask, 0).to(device)
    labels = torch.cat(all_labels, 0).numpy()

    feat_chunks = []
    bs = 64
    for i in range(0, input_ids.size(0), bs):
        feat_chunks.append(
            bert_cls(model, input_ids[i:i + bs], attention_mask[i:i + bs]).cpu())
    feats = torch.cat(feat_chunks, dim=0)

    sample_probs = []
    for s in samples:
        w = s["weight"].to(device)
        b = s["bias"].to(device)
        probs_b = []
        for i in range(0, feats.size(0), bs):
            f = feats[i:i + bs].to(device)
            logits = f @ w.T + b
            probs_b.append(F.softmax(logits, dim=-1).cpu().numpy())
        sample_probs.append(np.concatenate(probs_b, axis=0))
    probs_stack = np.stack(sample_probs, axis=0)
    eu, mean_probs = mutual_information(probs_stack)
    return eu, mean_probs, labels


def main():
    p = argparse.ArgumentParser(description="Evaluate SST-2 baselines (E5)")
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--methods", nargs="+",
                   default=["mc_dropout", "edl", "llla", "sgld"],
                   choices=["mc_dropout", "edl", "llla", "sgld"])
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    baseline_dir = os.path.join(args.save_dir, "baselines")
    targets_path = os.path.join(args.save_dir, "teacher_targets.npz")
    if not os.path.isfile(targets_path):
        raise FileNotFoundError(targets_path)

    data = np.load(targets_path, allow_pickle=True)
    teacher_eu = torch.from_numpy(data["dev_eu"].astype(np.float32))
    teacher_probs = data["dev_probs"]
    true_labels = data["dev_labels"]

    _, dev_ds, _ = load_sst2("bert-base-uncased")
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    print(f"Device: {device}")
    print(f"Teacher dev EU mean: {teacher_eu.mean().item():.4f}")

    def report(name, eu_np, probs_np):
        acc = 100.0 * (probs_np.argmax(-1) == true_labels).mean()
        ece = compute_ece(probs_np, true_labels)
        nll = compute_nll(probs_np, true_labels)
        br = compute_brier(probs_np, true_labels)
        eu_t = torch.from_numpy(eu_np.astype(np.float32))
        rp = pearson_corr(eu_t, teacher_eu)
        rs = spearman_corr(eu_t, teacher_eu)
        print(f"  {name:<22} acc={acc:5.2f}%  ECE={ece:.4f}  NLL={nll:.4f}  "
              f"Brier={br:.4f}  EU_P={rp:.4f}  EU_S={rs:.4f}")

    member0 = os.path.join(args.save_dir, "member_0.pt")

    if "mc_dropout" in args.methods:
        path = os.path.join(baseline_dir, "mc_dropout.pt")
        if os.path.isfile(path):
            print("\n[MC Dropout]")
            ckpt = torch.load(path, map_location=device, weights_only=False)
            model = load_teacher_from_ckpt(member0, device)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("trainable_state_dict")), strict=False)
            dp = ckpt.get("dropout_p", 0.2)
            eu, probs, _ = predict_mc_dropout(model, dev_loader, device, args.T, dp)
            report("MC Dropout", eu, probs)
        else:
            print(f"\n[MC Dropout] missing {path}")

    if "edl" in args.methods:
        path = os.path.join(baseline_dir, "edl_head.pt")
        if os.path.isfile(path):
            print("\n[EDL]")
            ckpt = torch.load(path, map_location=device, weights_only=False)
            model = load_teacher_from_ckpt(member0, device)
            model.load_state_dict(ckpt["bert_trainable_state_dict"], strict=False)
            edl_head = EDLHead(feat_dim=CLS_DIM, num_classes=NUM_CLASSES)
            edl_head.load_state_dict(ckpt["edl_head_state_dict"])
            eu_epi, _, probs, _ = predict_edl(model, edl_head, dev_loader, device)
            report("EDL (epistemic)", eu_epi, probs)
        else:
            print(f"\n[EDL] missing {path}")

    if "llla" in args.methods:
        path = os.path.join(baseline_dir, "llla.pt")
        if os.path.isfile(path):
            print("\n[LLLA]")
            try:
                from laplace import Laplace
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                model = load_teacher_from_ckpt(member0, device)
                lin = nn.Linear(CLS_DIM, NUM_CLASSES)
                lin.load_state_dict(ckpt["linear_state_dict"])
                la = Laplace(lin, "classification", subset_of_weights="all",
                             hessian_structure="diag")
                la.load_state_dict(ckpt["laplace_state_dict"])
                la.model.to(device)
                eu, probs, _ = predict_llla(model, la, dev_loader, device, args.T)
                if eu is not None:
                    report("LLLA", eu, probs)
            except Exception as e:
                print(f"  [LLLA] failed: {e}")
        else:
            print(f"\n[LLLA] missing {path}")

    if "sgld" in args.methods:
        path = os.path.join(baseline_dir, "sgld_samples.pt")
        if os.path.isfile(path):
            print("\n[SGLD]")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = load_teacher_from_ckpt(member0, device)
            eu, probs, _ = predict_sgld(model, ckpt["samples"], dev_loader, device)
            report("SGLD", eu, probs)
        else:
            print(f"\n[SGLD] missing {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
