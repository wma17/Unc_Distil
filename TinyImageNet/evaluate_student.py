"""
Comprehensive evaluation of the distilled Tiny-ImageNet student.

Outputs structured results suitable for result.md.

Usage:
    python evaluate_student.py --save_dir ./checkpoints --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset

from data import (
    CORRUPTIONS,
    TinyImageNetDataset,
    apply_corruption,
    download_tiny_imagenet,
    get_ood_loaders,
    get_val_transform,
)
from models import create_student, create_ensemble_member, load_saved_member_state, NUM_CLASSES

SEEN_OOD = {"SVHN", "CIFAR_100"}
UNSEEN_OOD = {"CIFAR_10", "LSUN", "iSUN", "Places365", "STL10", "DTD",
              "FashionMNIST", "MNIST"}


@torch.no_grad()
def predict_student(model, loader, device):
    all_probs, all_eu, all_labels = [], [], []
    model.eval()
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast("cuda"):
            logits, eu = model(imgs)
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        eu = eu.float().clamp(min=0).cpu().numpy()
        all_probs.append(probs)
        all_eu.append(eu)
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_eu), np.concatenate(all_labels)


@torch.no_grad()
def predict_member(model, loader, device):
    all_probs, all_labels = [], []
    model.eval()
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast("cuda"):
            logits = model(imgs)
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def entropy(p, eps=1e-10):
    return -np.sum(p * np.log(p + eps), axis=-1)


def auroc_safe(id_scores, ood_scores):
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def ood_type(name):
    safe = name.replace("-", "_").replace(" ", "_")
    return "seen" if safe in SEEN_OOD else "unseen"


def sec(title, f):
    f.write(f"\n{'='*60}\n  {title}\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    root = download_tiny_imagenet(args.data_dir)
    val_tf = get_val_transform()

    targets = dict(np.load(os.path.join(args.save_dir, "teacher_targets.npz"),
                           allow_pickle=True))

    model = create_student().to(device)
    ckpt = torch.load(os.path.join(args.save_dir, "student.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_ds = TinyImageNetDataset(root, split="val", transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=4, pin_memory=True)

    result_path = os.path.join(args.save_dir, "..", "result.md")
    out = open(result_path, "w")

    def w(s=""):
        print(s)
        out.write(s + "\n")

    w(f"Loaded student (spearman={ckpt.get('spearman','?'):.4f}, "
      f"pearson={ckpt.get('pearson','?'):.4f})")

    # ── 1. Accuracy ──
    sec("1. Accuracy", out)
    print(f"\n{'='*60}\n  1. Accuracy\n{'='*60}")
    s_probs, s_eu, s_labels = predict_student(model, val_loader, device)
    s_preds = s_probs.argmax(1)
    s_acc = (s_preds == s_labels).mean()

    t_probs_val = targets["val_mean_probs"]
    t_preds = t_probs_val.argmax(1)
    t_acc = (t_preds == targets["val_labels"]).mean()

    w(f"  Teacher (ensemble):  {t_acc*100:.2f}%")
    w(f"  Student (distilled): {s_acc*100:.2f}%")

    # ── 2. Correctness Agreement ──
    sec("2. Correctness Agreement", out)
    print(f"\n{'='*60}\n  2. Correctness Agreement\n{'='*60}")
    n = min(len(s_preds), len(t_preds))
    s_correct = s_preds[:n] == s_labels[:n]
    t_correct = t_preds[:n] == targets["val_labels"][:n]
    both_c = (s_correct & t_correct).sum()
    both_w = (~s_correct & ~t_correct).sum()
    s_only = (s_correct & ~t_correct).sum()
    t_only = (~s_correct & t_correct).sum()
    agreement = (s_correct == t_correct).mean()

    w(f"  Both correct:           {both_c:5d}  ({both_c/n*100:.1f}%)")
    w(f"  Both wrong:              {both_w:4d}  ({both_w/n*100:.1f}%)")
    w(f"  Student correct only:    {s_only:4d}  ({s_only/n*100:.1f}%)")
    w(f"  Teacher correct only:    {t_only:4d}  ({t_only/n*100:.1f}%)")
    w(f"  Total agreement:       {agreement*100:.2f}%")

    # ── 3. EU Correlation ──
    sec("3. EU Correlation (student EU vs teacher EU)", out)
    print(f"\n{'='*60}\n  3. EU Correlation\n{'='*60}")
    corr_results = []

    eu_tgt_val = targets["val_EU"][:len(s_eu)]
    pear = np.corrcoef(s_eu, eu_tgt_val)[0, 1]
    spear = sp_stats.spearmanr(s_eu, eu_tgt_val).correlation
    corr_results.append(("Clean TinyImageNet val", pear, spear, s_eu.mean(), eu_tgt_val.mean()))

    for cname in CORRUPTIONS:
        key = f"corrupt_{cname}_EU"
        if key not in targets:
            continue
        c_imgs, c_labels = apply_corruption(val_ds, CORRUPTIONS[cname], max_samples=5000)
        c_ds = TensorDataset(c_imgs, c_labels)
        c_loader = DataLoader(c_ds, batch_size=args.batch_size, num_workers=2)
        _, c_eu, _ = predict_student(model, c_loader, device)
        c_tgt = targets[key][:len(c_eu)]
        if c_eu.std() > 1e-8 and c_tgt.std() > 1e-8:
            p = np.corrcoef(c_eu, c_tgt)[0, 1]
            s = sp_stats.spearmanr(c_eu, c_tgt).correlation
            corr_results.append((f"Corrupted: {cname}", p, s, c_eu.mean(), c_tgt.mean()))

    ood_loaders = get_ood_loaders(args.data_dir, args.batch_size)
    ood_cache = {}
    for name, loader in ood_loaders.items():
        safe = name.replace("-", "_").replace(" ", "_")
        key = f"ood_{safe}_EU"
        o_probs, o_eu, o_labels = predict_student(model, loader, device)
        ood_cache[name] = (o_probs, o_eu, o_labels)
        if key in targets:
            o_tgt = targets[key][:len(o_eu)]
            if o_eu.std() > 1e-8 and o_tgt.std() > 1e-8:
                p = np.corrcoef(o_eu, o_tgt)[0, 1]
                s = sp_stats.spearmanr(o_eu, o_tgt).correlation
                corr_results.append((f"OOD: {name}", p, s, o_eu.mean(), o_tgt.mean()))

    w(f"  {'Dataset':<30} {'Pearson':>10} {'Spearman':>10} {'Stu mean':>10} {'Tea mean':>10}")
    w(f"  {'-'*70}")
    for dname, p, s, sm, tm in corr_results:
        w(f"  {dname:<30} {p:>10.4f} {s:>10.4f} {sm:>10.4f} {tm:>10.4f}")

    # ── 4a. OOD Detection — SEEN ──
    sec("4a. OOD Detection — SEEN OOD (used in Phase 2 training)", out)
    print(f"\n{'='*60}\n  4a. OOD Detection — SEEN\n{'='*60}")

    s_tu = entropy(s_probs)
    s_au = np.maximum(s_tu - s_eu, 0)
    t_id_eu = targets["val_EU"]
    t_id_tu = targets.get("val_TU", entropy(t_probs_val))

    for name in ood_loaders:
        if ood_type(name) != "seen":
            continue
        o_probs, o_eu, _ = ood_cache[name]
        o_tu = entropy(o_probs)
        safe = name.replace("-", "_").replace(" ", "_")
        t_eu_key = f"ood_{safe}_EU"
        t_tu_key = f"ood_{safe}_TU"
        t_ood_eu = targets.get(t_eu_key)
        t_ood_tu = targets.get(t_tu_key)

        w(f"\n  Clean TinyImageNet (neg) vs {name} (pos)")
        w(f"  {'Method':<40} {'AUROC':>8}")
        w(f"  {'-'*48}")
        if t_ood_eu is not None:
            w(f"  {'Teacher EU (ensemble MI)':<40} {auroc_safe(t_id_eu, t_ood_eu):>8.4f}")
        if t_ood_tu is not None:
            w(f"  {'Teacher TU (ensemble entropy)':<40} {auroc_safe(t_id_tu, t_ood_tu):>8.4f}")
        w(f"  {'Student EU (learned)':<40} {auroc_safe(s_eu, o_eu):>8.4f}")
        w(f"  {'Student TU (entropy)':<40} {auroc_safe(s_tu, o_tu):>8.4f}")
        w(f"  {'1 - max softmax prob':<40} {auroc_safe(1 - s_probs.max(1), 1 - o_probs.max(1)):>8.4f}")

    # ── 4b. OOD Detection — UNSEEN ──
    sec("4b. OOD Detection — UNSEEN OOD (not in training)", out)
    print(f"\n{'='*60}\n  4b. OOD Detection — UNSEEN\n{'='*60}")

    for name in ood_loaders:
        if ood_type(name) != "unseen":
            continue
        o_probs, o_eu, _ = ood_cache[name]
        o_tu = entropy(o_probs)
        safe = name.replace("-", "_").replace(" ", "_")
        t_eu_key = f"ood_{safe}_EU"
        t_tu_key = f"ood_{safe}_TU"
        t_ood_eu = targets.get(t_eu_key)
        t_ood_tu = targets.get(t_tu_key)

        w(f"\n  Clean TinyImageNet (neg) vs {name} (pos)")
        w(f"  {'Method':<40} {'AUROC':>8}")
        w(f"  {'-'*48}")
        if t_ood_eu is not None:
            w(f"  {'Teacher EU (ensemble MI)':<40} {auroc_safe(t_id_eu, t_ood_eu):>8.4f}")
        if t_ood_tu is not None:
            w(f"  {'Teacher TU (ensemble entropy)':<40} {auroc_safe(t_id_tu, t_ood_tu):>8.4f}")
        w(f"  {'Student EU (learned)':<40} {auroc_safe(s_eu, o_eu):>8.4f}")
        w(f"  {'Student TU (entropy)':<40} {auroc_safe(s_tu, o_tu):>8.4f}")
        w(f"  {'1 - max softmax prob':<40} {auroc_safe(1 - s_probs.max(1), 1 - o_probs.max(1)):>8.4f}")

    # ── 4c. Shifted ID vs OOD ──
    sec("4c. OOD Detection — Shifted TinyImageNet vs OOD", out)
    print(f"\n{'='*60}\n  4c. Shifted ID vs OOD\n{'='*60}")

    shift_imgs_all, shift_labels_all = [], []
    for cname in CORRUPTIONS:
        ci, cl = apply_corruption(val_ds, CORRUPTIONS[cname], max_samples=1000)
        shift_imgs_all.append(ci)
        shift_labels_all.append(cl)
    shift_imgs = torch.cat(shift_imgs_all)
    shift_labels = torch.cat(shift_labels_all)
    shift_ds = TensorDataset(shift_imgs, shift_labels)
    shift_loader = DataLoader(shift_ds, batch_size=args.batch_size, num_workers=2)
    shift_probs, shift_eu, shift_lab = predict_student(model, shift_loader, device)
    shift_tu = entropy(shift_probs)

    for name in ood_loaders:
        o_probs, o_eu, _ = ood_cache[name]
        o_tu = entropy(o_probs)
        typ = ood_type(name)
        w(f"\n  Shifted TinyImageNet (neg) vs {name} ({typ}) (pos)")
        w(f"  {'Method':<40} {'AUROC':>8}")
        w(f"  {'-'*48}")
        w(f"  {'Student EU (learned)':<40} {auroc_safe(shift_eu, o_eu):>8.4f}")
        w(f"  {'Student TU (entropy)':<40} {auroc_safe(shift_tu, o_tu):>8.4f}")

    # ── 4d. Distribution Shift Detection ──
    sec("4d. Distribution Shift Detection", out)
    print(f"\n{'='*60}\n  4d. Distribution Shift Detection\n{'='*60}")

    w(f"\n  Clean TinyImageNet (neg) vs Shifted TinyImageNet (pos)")
    w(f"  {'Method':<40} {'AUROC':>8}")
    w(f"  {'-'*48}")
    t_shift_eu = np.concatenate([targets.get(f"corrupt_{c}_EU", np.array([]))
                                 for c in CORRUPTIONS])
    if len(t_shift_eu) > 0:
        w(f"  {'Teacher EU (ensemble)':<40} {auroc_safe(t_id_eu, t_shift_eu):>8.4f}")
    w(f"  {'Student EU (learned)':<40} {auroc_safe(s_eu, shift_eu):>8.4f}")
    w(f"  {'Student TU (entropy)':<40} {auroc_safe(s_tu, shift_tu):>8.4f}")

    # ── 5. Uncertainty Decomposition ──
    sec("5. Uncertainty Decomposition (TinyImageNet val)", out)
    print(f"\n{'='*60}\n  5. Uncertainty Decomposition\n{'='*60}")

    t_tu = targets.get("val_TU", entropy(t_probs_val))
    t_au = targets.get("val_AU", np.zeros_like(t_tu))
    t_eu_v = targets["val_EU"]

    w(f"  {'Metric':<10} {'Teacher mean':>14} {'Student mean':>14} {'Pearson':>10} {'Spearman':>10}")
    w(f"  {'-'*58}")
    for mname, sv, tv in [("TU", s_tu, t_tu), ("AU", s_au, t_au), ("EU", s_eu, t_eu_v)]:
        p = np.corrcoef(sv[:len(tv)], tv[:len(sv)])[0, 1] if sv.std() > 1e-8 else 0
        s = sp_stats.spearmanr(sv[:len(tv)], tv[:len(sv)]).correlation if sv.std() > 1e-8 else 0
        w(f"  {mname:<10} {tv.mean():>14.6f} {sv.mean():>14.6f} {p:>10.4f} {s:>10.4f}")

    # ── 6. Decomposed OOD AUROC summary table ──
    sec("6. OOD Detection AUROC — Full Comparison", out)
    print(f"\n{'='*60}\n  6. Full OOD Comparison Table\n{'='*60}")

    t_id_tu = targets.get("val_TU", entropy(t_probs_val))

    member0 = None
    try:
        cfg_path = os.path.join(args.save_dir, "ensemble_configs.json")
        with open(cfg_path) as f:
            configs = json.load(f)
        cfg0 = configs[0]
        member0 = create_ensemble_member(
            rank=cfg0["rank"], alpha=cfg0["alpha"],
            lora_dropout=cfg0["lora_dropout"], targets=cfg0["targets"],
            unfreeze_blocks=cfg0.get("unfreeze_blocks", 0),
        )
        ckpt0 = torch.load(
            os.path.join(args.save_dir, f"member_{cfg0['member_id']}.pt"),
            map_location=device, weights_only=True,
        )
        load_saved_member_state(member0, ckpt0, strict=False)
        member0.to(device).eval()
        m_probs, _ = predict_member(member0, val_loader, device)
        m_ent = entropy(m_probs)
    except Exception as e:
        print(f"  [skip] single member baseline: {e}")

    w(f"  {'Dataset':<14} {'Type':<7} {'Tea EU':>7} {'Tea TU':>7} | {'Stu EU':>7} {'Stu TU':>7} {'Stu AU':>7} | {'Sgl(H)':>7}")
    w(f"  {'-'*78}")

    for name in ood_loaders:
        o_probs, o_eu, _ = ood_cache[name]
        o_tu = entropy(o_probs)
        o_au = np.maximum(o_tu - o_eu, 0)
        typ = ood_type(name)

        safe = name.replace("-", "_").replace(" ", "_")
        t_eu_key = f"ood_{safe}_EU"
        t_tu_key = f"ood_{safe}_TU"
        t_auc_eu = auroc_safe(t_id_eu, targets[t_eu_key]) if t_eu_key in targets else float("nan")
        t_ood_tu = targets.get(t_tu_key)
        t_auc_tu = auroc_safe(t_id_tu, t_ood_tu) if t_ood_tu is not None else float("nan")
        s_auc_eu = auroc_safe(s_eu, o_eu)
        s_auc_tu = auroc_safe(s_tu, o_tu)
        s_auc_au = auroc_safe(s_au, o_au)

        single_auc = "    N/A"
        if member0 is not None:
            try:
                mp, _ = predict_member(member0, ood_loaders[name], device)
                me = entropy(mp)
                single_auc = f"{auroc_safe(m_ent, me):>7.4f}"
            except:
                pass

        w(f"  {name:<14} {typ:<7} {t_auc_eu:>7.4f} {t_auc_tu:>7.4f} | {s_auc_eu:>7.4f} {s_auc_tu:>7.4f} {s_auc_au:>7.4f} | {single_auc}")

    w(f"\n  Tea EU = I[y;θ|x] (ensemble MI),  Tea TU = H[Ē[p]] (ensemble entropy)")
    w(f"  Stu EU = EU head,  Stu TU = H[softmax(logits)],  Stu AU = TU - EU")
    w(f"  Sgl(H) = entropy of one LoRA member (member_0)")
    w(f"  Fair comparison: Tea EU↔Stu EU (epistemic); Tea TU↔Stu TU↔Sgl(H) (entropy)")

    # ── 7. Single Member Baseline ──
    sec("7. Baseline: Single Ensemble Member", out)
    print(f"\n{'='*60}\n  7. Single Member Baseline\n{'='*60}")

    if member0 is not None:
        m_acc = (m_probs.argmax(1) == s_labels[:len(m_probs)]).mean()
        w(f"  Single member test accuracy: {m_acc*100:.2f}%")
        w(f"\n  {'Dataset':<16} {'Type':<8} {'Entropy':>10} {'1-MaxProb':>10}")
        w(f"  {'-'*48}")
        for name in ood_loaders:
            typ = ood_type(name)
            try:
                mp, _ = predict_member(member0, ood_loaders[name], device)
                me = entropy(mp)
                mm = 1.0 - mp.max(1)
                w(f"  {name:<16} {typ:<8} {auroc_safe(m_ent, me):>10.4f} {auroc_safe(1-m_probs.max(1), mm):>10.4f}")
            except:
                w(f"  {name:<16} {typ:<8} {'N/A':>10} {'N/A':>10}")
    else:
        w("  [skip] no single member available")

    w(f"\n{'='*60}")
    w("Evaluation complete.")
    w(f"{'='*60}")

    out.close()
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
