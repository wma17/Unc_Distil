"""
Uncertainty visualization for the Tiny-ImageNet pipeline.

Figures:
    1. EU scatter: student vs teacher per dataset (subplots)
    2. EU histograms: teacher — ID / Shifted ID / OOD
    3. EU histograms: student — ID / Shifted ID / OOD
    4. Violin: teacher (blue) vs student (red) side-by-side
    5. TU/AU/EU decomposition comparison scatter (val)
    6. Decomposition stacked bars: teacher vs student
    7. Student TU/EU/AU density overlays by dataset
    8. Teacher TU/EU/AU density overlays by dataset

Usage:
    python plot_eu.py --save_dir ./checkpoints --out_dir ./figures --gpu 0
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F
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
from models import create_student

COLORS = {
    "Clean ID": "#2ecc71",
    "Shifted ID": "#f39c12",
    "SVHN": "#3498db",
    "CIFAR-10": "#e91e63",
    "CIFAR-100": "#9c27b0",
    "STL10": "#00bcd4",
    "DTD": "#ff5722",
    "FashionMNIST": "#795548",
    "MNIST": "#e74c3c",
    "LSUN": "#4caf50",
    "iSUN": "#607d8b",
    "Places365": "#8bc34a",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def entropy(p, eps=1e-10):
    return -np.sum(p * np.log(p + eps), axis=-1)


@torch.no_grad()
def predict_student(model, loader, device):
    all_probs, all_eu = [], []
    model.eval()
    for imgs, *rest in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast("cuda"):
            logits, eu = model(imgs)
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        eu = eu.float().clamp(min=0).cpu().numpy()
        all_probs.append(probs)
        all_eu.append(eu)
    return np.concatenate(all_probs), np.concatenate(all_eu)


def collect_all_eu(model, targets, val_ds, args, device):
    """Collect teacher+student EU for: Clean ID, Shifted ID (all corruptions merged), and each OOD."""
    result = {}

    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=4, pin_memory=True)
    s_probs, s_eu = predict_student(model, val_loader, device)
    t_eu = targets["val_EU"][:len(s_eu)]
    result["Clean ID"] = {
        "student_eu": s_eu, "teacher_eu": t_eu, "student_probs": s_probs,
        "teacher_TU": targets.get("val_TU", entropy(targets["val_mean_probs"]))[:len(s_eu)],
        "teacher_AU": targets.get("val_AU", np.zeros(len(s_eu)))[:len(s_eu)],
    }

    shift_s_eu, shift_t_eu, shift_s_probs = [], [], []
    shift_t_tu, shift_t_au = [], []
    for cname in CORRUPTIONS:
        key = f"corrupt_{cname}_EU"
        if key not in targets:
            continue
        c_imgs, c_labels = apply_corruption(val_ds, CORRUPTIONS[cname], max_samples=2000)
        c_ds = TensorDataset(c_imgs, c_labels)
        c_loader = DataLoader(c_ds, batch_size=args.batch_size, num_workers=2)
        cp, ce = predict_student(model, c_loader, device)
        n = min(len(ce), len(targets[key]))
        shift_s_eu.append(ce[:n])
        shift_t_eu.append(targets[key][:n])
        shift_s_probs.append(cp[:n])
        shift_t_tu.append(targets.get(f"corrupt_{cname}_TU", np.zeros(n))[:n])
        shift_t_au.append(targets.get(f"corrupt_{cname}_AU", np.zeros(n))[:n])

    if shift_s_eu:
        result["Shifted ID"] = {
            "student_eu": np.concatenate(shift_s_eu),
            "teacher_eu": np.concatenate(shift_t_eu),
            "student_probs": np.concatenate(shift_s_probs),
            "teacher_TU": np.concatenate(shift_t_tu),
            "teacher_AU": np.concatenate(shift_t_au),
        }

    ood_loaders = get_ood_loaders(args.data_dir, args.batch_size)
    for name, loader in ood_loaders.items():
        safe = name.replace("-", "_").replace(" ", "_")
        key = f"ood_{safe}_EU"
        op, oe = predict_student(model, loader, device)
        t_eu_ood = targets.get(key)
        if t_eu_ood is not None:
            n = min(len(oe), len(t_eu_ood))
            result[name] = {
                "student_eu": oe[:n], "teacher_eu": t_eu_ood[:n],
                "student_probs": op[:n],
                "teacher_TU": targets.get(f"ood_{safe}_TU", np.zeros(n))[:n],
                "teacher_AU": targets.get(f"ood_{safe}_AU", np.zeros(n))[:n],
            }
        else:
            result[name] = {
                "student_eu": oe, "teacher_eu": np.full(len(oe), np.nan),
                "student_probs": op,
                "teacher_TU": np.full(len(oe), np.nan),
                "teacher_AU": np.full(len(oe), np.nan),
            }

    return result


# ── Fig 1: EU scatter per dataset ──

def fig1_eu_scatter(data, out_dir):
    has_teacher = {k: v for k, v in data.items() if not np.isnan(v["teacher_eu"]).any()}
    names = list(has_teacher.keys())
    n = len(names)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)
    for idx, name in enumerate(names):
        ax = axes[idx // cols][idx % cols]
        d = has_teacher[name]
        c = COLORS.get(name, "#7f8c8d")
        ax.scatter(d["teacher_eu"], d["student_eu"], alpha=0.15, s=4, c=c, rasterized=True)
        lim = max(d["teacher_eu"].max(), d["student_eu"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
        r = np.corrcoef(d["teacher_eu"], d["student_eu"])[0, 1]
        ax.set_title(f"{name}", fontweight="bold")
        ax.set_xlabel("Teacher EU")
        ax.set_ylabel("Student EU")
        ax.text(0.05, 0.92, f"r={r:.3f}", transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    fig.suptitle("Teacher EU vs Student EU", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_eu_scatter.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_eu_scatter.png")


# ── Fig 2 & 3: EU histograms ──

def _plot_eu_hist(data, key, title, filename, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    all_vals = np.concatenate([d[key] for d in data.values() if not np.isnan(d[key]).any()])
    bins = np.linspace(0, np.percentile(all_vals, 99.5), 60)
    for name, d in data.items():
        vals = d[key]
        if np.isnan(vals).any():
            continue
        c = COLORS.get(name, "#7f8c8d")
        ax.hist(vals, bins=bins, alpha=0.45, label=f"{name} (μ={vals.mean():.3f})",
                color=c, density=True, histtype="stepfilled", edgecolor=c, lw=1.2)
    ax.set_xlabel("Epistemic Uncertainty")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


# ── Fig 4: Teacher+Student violin side-by-side ──

def fig4_violin_comparison(data, out_dir):
    names = list(data.keys())
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(12, n * 1.8), 5.5))
    positions = np.arange(n)
    width = 0.35

    teacher_data = []
    student_data = []
    for name in names:
        t = data[name]["teacher_eu"]
        s = data[name]["student_eu"]
        teacher_data.append(t if not np.isnan(t).any() else np.array([0.0]))
        student_data.append(s)

    clip_max = np.percentile(np.concatenate(student_data), 99.5)
    teacher_clipped = [np.clip(d, 0, clip_max) for d in teacher_data]
    student_clipped = [np.clip(d, 0, clip_max) for d in student_data]

    vp_t = ax.violinplot(teacher_clipped, positions=positions - width / 2,
                         widths=width, showmeans=True, showmedians=True)
    vp_s = ax.violinplot(student_clipped, positions=positions + width / 2,
                         widths=width, showmeans=True, showmedians=True)
    for body in vp_t["bodies"]:
        body.set_facecolor("#3498db")
        body.set_alpha(0.6)
    for k in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if k in vp_t:
            vp_t[k].set_color("#2c3e50")
    for body in vp_s["bodies"]:
        body.set_facecolor("#e74c3c")
        body.set_alpha(0.6)
    for k in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if k in vp_s:
            vp_s[k].set_color("#922b21")

    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Epistemic Uncertainty")
    ax.set_title("EU Distribution: Teacher (blue) vs Student (red)", fontsize=14)
    ax.legend(handles=[Patch(fc="#3498db", alpha=0.6, label="Teacher (ensemble)"),
                       Patch(fc="#e74c3c", alpha=0.6, label="Student (distilled)")],
              loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_violin_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_violin_comparison.png")


# ── Fig 5: Decomposition scatter (val) ──

def fig5_decomposition_scatter(data, out_dir):
    d = data["Clean ID"]
    s_tu = entropy(d["student_probs"])
    s_eu = d["student_eu"]
    s_au = np.maximum(s_tu - s_eu, 0)
    t_tu = d["teacher_TU"]
    t_au = d["teacher_AU"]
    t_eu = d["teacher_eu"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (label, sv, tv) in zip(axes, [("TU", s_tu, t_tu), ("AU", s_au, t_au), ("EU", s_eu, t_eu)]):
        n = min(len(sv), len(tv))
        ax.scatter(tv[:n], sv[:n], alpha=0.1, s=3, c="#2196F3", rasterized=True)
        lo = min(tv[:n].min(), sv[:n].min())
        hi = max(tv[:n].max(), sv[:n].max())
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)
        r = np.corrcoef(tv[:n], sv[:n])[0, 1]
        ax.set_xlabel(f"Teacher {label}")
        ax.set_ylabel(f"Student {label}")
        ax.set_title(f"{label} (r={r:.3f})")
    fig.suptitle("Uncertainty Decomposition: Teacher vs Student (Clean ID)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_decomposition_scatter.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_decomposition_scatter.png")


# ── Fig 6: Stacked bars — Teacher vs Student decomposition ──

def fig6_decomposition_bars(data, out_dir):
    names = list(data.keys())
    n = len(names)

    t_au_m, t_eu_m, s_au_m, s_eu_m = [], [], [], []
    for name in names:
        d = data[name]
        s_tu = entropy(d["student_probs"])
        s_eu = d["student_eu"]
        s_au_m.append(np.maximum(s_tu - s_eu, 0).mean())
        s_eu_m.append(s_eu.mean())
        t_eu_m.append(d["teacher_eu"].mean() if not np.isnan(d["teacher_eu"]).any() else 0)
        t_au_m.append(d["teacher_AU"].mean() if not np.isnan(d["teacher_AU"]).any() else 0)

    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(12, n * 2), 6))

    ax.bar(x - width / 2, t_au_m, width, color="#85c1e9", label="Teacher AU")
    ax.bar(x - width / 2, t_eu_m, width, bottom=t_au_m, color="#2980b9", label="Teacher EU")
    ax.bar(x + width / 2, s_au_m, width, color="#f5b7b1", label="Student AU")
    ax.bar(x + width / 2, s_eu_m, width, bottom=s_au_m, color="#e74c3c", label="Student EU")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Mean Uncertainty")
    ax.set_title("Uncertainty Decomposition: TU = AU + EU\nTeacher (left) vs Student (right)", fontsize=13)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_decomposition_bars.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig6_decomposition_bars.png")


# ── Fig 7: Student TU/EU/AU density overlays ──

def fig7_student_density(data, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (utype, label) in zip(axes, [("tu", "TU"), ("eu", "EU"), ("au", "AU")]):
        all_vals = []
        for name, d in data.items():
            s_tu = entropy(d["student_probs"])
            s_eu = d["student_eu"]
            s_au = np.maximum(s_tu - s_eu, 0)
            vals = {"tu": s_tu, "eu": s_eu, "au": s_au}[utype]
            all_vals.append((name, vals))
        combined = np.concatenate([v for _, v in all_vals])
        bins = np.linspace(0, np.percentile(combined, 99.5), 50)
        for name, vals in all_vals:
            c = COLORS.get(name, "#7f8c8d")
            ax.hist(vals, bins=bins, alpha=0.4, color=c, density=True,
                    histtype="stepfilled", edgecolor=c, lw=1,
                    label=f"{name} (μ={vals.mean():.3f})")
        ax.set_xlabel("Uncertainty")
        ax.set_ylabel("Density")
        ax.set_title(f"Student {label}", fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(left=0)
    fig.suptitle("Student Model — TU / EU / AU by Dataset", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig7_student_density.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig7_student_density.png")


# ── Fig 8: Teacher TU/EU/AU density overlays ──

def fig8_teacher_density(data, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (utype, label) in zip(axes, [("tu", "TU"), ("eu", "EU"), ("au", "AU")]):
        all_vals = []
        for name, d in data.items():
            if np.isnan(d["teacher_eu"]).any():
                continue
            vals = {"tu": d["teacher_TU"], "eu": d["teacher_eu"], "au": d["teacher_AU"]}[utype]
            all_vals.append((name, vals))
        if not all_vals:
            continue
        combined = np.concatenate([v for _, v in all_vals])
        bins = np.linspace(0, np.percentile(combined, 99.5), 50)
        for name, vals in all_vals:
            c = COLORS.get(name, "#7f8c8d")
            ax.hist(vals, bins=bins, alpha=0.4, color=c, density=True,
                    histtype="stepfilled", edgecolor=c, lw=1,
                    label=f"{name} (μ={vals.mean():.3f})")
        ax.set_xlabel("Uncertainty")
        ax.set_ylabel("Density")
        ax.set_title(f"Teacher {label}", fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(left=0)
    fig.suptitle("BNN Teacher — TU / EU / AU by Dataset", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig8_teacher_density.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig8_teacher_density.png")


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--out_dir", type=str, default="./figures")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
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
    print("Loaded student model.")

    val_ds = TinyImageNetDataset(root, split="val", transform=val_tf)
    print("Collecting predictions for all datasets ...")
    data = collect_all_eu(model, targets, val_ds, args, device)
    print(f"  Datasets: {list(data.keys())}")

    print("\nGenerating figures ...")
    fig1_eu_scatter(data, args.out_dir)
    _plot_eu_hist(data, "teacher_eu", "BNN Teacher — EU Distribution", "fig2_teacher_eu_hist.png", args.out_dir)
    _plot_eu_hist(data, "student_eu", "Student — EU Distribution", "fig3_student_eu_hist.png", args.out_dir)
    fig4_violin_comparison(data, args.out_dir)
    fig5_decomposition_scatter(data, args.out_dir)
    fig6_decomposition_bars(data, args.out_dir)
    fig7_student_density(data, args.out_dir)
    fig8_teacher_density(data, args.out_dir)

    print(f"\nAll 8 figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
