"""
Visualize epistemic uncertainty distributions, teacher-student correlations,
and full uncertainty decomposition (TU / EU / AU) for the MNIST pipeline.

Figures:
    1. Teacher vs Student EU scatter + correlation per dataset
    2. BNN teacher EU distribution across dataset types
    3. Student EU distribution across dataset types
    4. Teacher vs Student violin comparison
    5. Uncertainty decomposition: stacked bars (AU + EU = TU) for both models
    6. Student TU / EU / AU distribution overlays per dataset
    7. Teacher TU / EU / AU distribution overlays per dataset
    8. Teacher vs Student decomposition on clean MNIST

Usage:
    python plot_eu.py --save_dir ./checkpoints --gpu 0 --out_dir ./figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

from evaluate_student import (
    SEEN_OOD, UNSEEN_OOD, MNIST_MEAN, MNIST_STD, EPS,
    load_student, load_all_targets, get_test_loader, get_ood_loader,
    get_corrupted_test_loader, predict_student, spearman_corr, pearson_corr,
    CORRUPTION_TYPES,
)

COLORS = {
    "Clean MNIST": "#2ecc71",
    "Shifted MNIST": "#f39c12",
    "FashionMNIST": "#3498db",
    "Omniglot": "#2980b9",
    "EMNIST-Letters": "#e74c3c",
    "CIFAR-10": "#9b59b6",
    "SVHN": "#e67e22",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def entropy_np(p):
    return -(p * np.log(p + EPS)).sum(axis=-1)


def collect_all_eu(model, data, args, device):
    results = {}

    def _get_teacher_decomp(prefix):
        tu = data.get(f"{prefix}_tu")
        au = data.get(f"{prefix}_au")
        return tu, au

    # Clean MNIST test
    test_loader = get_test_loader(args.data_dir, args.batch_size, args.num_workers)
    s_probs, s_eu = predict_student(model, test_loader, device)
    t_tu, t_au = _get_teacher_decomp("test")
    if t_tu is None and "test_probs" in data:
        t_tu = entropy_np(data["test_probs"])
        t_au = np.maximum(t_tu - data["test_eu"], 0.0)
    results["Clean MNIST"] = {
        "teacher_eu": data["test_eu"],
        "teacher_tu": t_tu,
        "teacher_au": t_au,
        "student_eu": s_eu.numpy(),
        "student_probs": s_probs.numpy(),
    }

    # Corrupted MNIST test (aggregate)
    shifted_tea_eu, shifted_tea_tu, shifted_tea_au = [], [], []
    shifted_stu_eu, shifted_stu_probs = [], []
    for ctype in CORRUPTION_TYPES:
        key = f"corrupt_{ctype}_test_eu"
        if key not in data:
            continue
        c_loader = get_corrupted_test_loader(args.data_dir, ctype, args.batch_size)
        c_probs, c_eu = predict_student(model, c_loader, device)
        shifted_tea_eu.append(data[key])
        shifted_stu_eu.append(c_eu.numpy())
        shifted_stu_probs.append(c_probs.numpy())
        c_tu, c_au = _get_teacher_decomp(f"corrupt_{ctype}_test")
        shifted_tea_tu.append(c_tu)
        shifted_tea_au.append(c_au)
    if shifted_tea_eu:
        has_all_tu = all(x is not None for x in shifted_tea_tu)
        results["Shifted MNIST"] = {
            "teacher_eu": np.concatenate(shifted_tea_eu),
            "teacher_tu": np.concatenate(shifted_tea_tu) if has_all_tu else None,
            "teacher_au": np.concatenate(shifted_tea_au) if has_all_tu else None,
            "student_eu": np.concatenate(shifted_stu_eu),
            "student_probs": np.concatenate(shifted_stu_probs),
        }

    # All OOD datasets
    for ood_items in [SEEN_OOD, UNSEEN_OOD]:
        for ood_id, cache_key, display_name in ood_items:
            if cache_key not in data:
                continue
            try:
                ood_loader = get_ood_loader(ood_id, args.data_dir,
                                            args.batch_size, args.num_workers)
                ood_probs, ood_eu = predict_student(model, ood_loader, device)
            except Exception as e:
                print(f"  Skipping {display_name}: {e}")
                continue
            t_eu = data[cache_key]
            tu_key = cache_key.replace("_eu", "_tu")
            au_key = cache_key.replace("_eu", "_au")
            t_tu = data.get(tu_key)
            t_au = data.get(au_key)
            min_n = min(len(ood_eu), len(t_eu))
            results[display_name] = {
                "teacher_eu": t_eu[:min_n],
                "teacher_tu": t_tu[:min_n] if t_tu is not None else None,
                "teacher_au": t_au[:min_n] if t_au is not None else None,
                "student_eu": ood_eu[:min_n].numpy(),
                "student_probs": ood_probs[:min_n].numpy(),
            }

    return results


def plot_correlation(all_eu, out_dir):
    names = list(all_eu.keys())
    n = len(names)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)

    for idx, name in enumerate(names):
        ax = axes[idx // cols][idx % cols]
        t_eu = all_eu[name]["teacher_eu"]
        s_eu = all_eu[name]["student_eu"]
        color = COLORS.get(name, "#7f8c8d")
        ax.scatter(t_eu, s_eu, alpha=0.15, s=4, color=color, rasterized=True)
        lim = max(t_eu.max(), s_eu.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
        t_t = torch.from_numpy(t_eu).float()
        s_t = torch.from_numpy(s_eu).float()
        r_p = pearson_corr(t_t, s_t)
        r_s = spearman_corr(t_t, s_t)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Teacher EU (nats)")
        ax.set_ylabel("Student EU (nats)")
        ax.text(0.05, 0.92, f"Pearson={r_p:.3f}\nSpearman={r_s:.3f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    fig.suptitle("Teacher EU vs Student EU — per dataset", fontsize=15, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "eu_correlation.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_eu_distribution(all_eu, key, title, filename, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    all_vals = np.concatenate([all_eu[n][key] for n in all_eu])
    bins = np.linspace(0, np.percentile(all_vals, 99.5), 60)
    for name in all_eu:
        vals = all_eu[name][key]
        color = COLORS.get(name, "#7f8c8d")
        ax.hist(vals, bins=bins, alpha=0.5, label=f"{name} (μ={vals.mean():.3f})",
                color=color, density=True, histtype="stepfilled", edgecolor=color, linewidth=1.2)
    ax.set_xlabel("Epistemic Uncertainty (nats)")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_violin_comparison(all_eu, out_dir):
    names = list(all_eu.keys())
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 5))
    positions = np.arange(n)
    teacher_data = [all_eu[name]["teacher_eu"] for name in names]
    student_data = [all_eu[name]["student_eu"] for name in names]
    clip_max = max(np.percentile(np.concatenate(teacher_data), 99),
                   np.percentile(np.concatenate(student_data), 99))
    teacher_clipped = [np.clip(d, 0, clip_max) for d in teacher_data]
    student_clipped = [np.clip(d, 0, clip_max) for d in student_data]
    width = 0.35
    vp_t = ax.violinplot(teacher_clipped, positions=positions - width/2,
                         widths=width, showmeans=True, showmedians=True)
    vp_s = ax.violinplot(student_clipped, positions=positions + width/2,
                         widths=width, showmeans=True, showmedians=True)
    for body in vp_t["bodies"]:
        body.set_facecolor("#3498db"); body.set_alpha(0.6)
    for k in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        if k in vp_t: vp_t[k].set_color("#2c3e50")
    for body in vp_s["bodies"]:
        body.set_facecolor("#e74c3c"); body.set_alpha(0.6)
    for k in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        if k in vp_s: vp_s[k].set_color("#c0392b")
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Epistemic Uncertainty (nats)")
    ax.set_title("EU Distribution: Teacher (blue) vs Student (red)", fontsize=14)
    ax.set_xlim(-0.5, n - 0.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#3498db", alpha=0.6, label="Teacher (ensemble)"),
                       Patch(facecolor="#e74c3c", alpha=0.6, label="Student (distilled)")],
              loc="upper left")
    fig.tight_layout()
    path = os.path.join(out_dir, "eu_violin_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_decomposition_bars(all_eu, out_dir):
    names = list(all_eu.keys())
    n = len(names)

    teacher_au, teacher_eu, teacher_tu = [], [], []
    student_au, student_eu, student_tu = [], [], []

    for name in names:
        d = all_eu[name]
        s_tu = entropy_np(d["student_probs"])
        s_eu = d["student_eu"]
        s_au = np.maximum(s_tu - s_eu, 0.0)
        student_tu.append(s_tu.mean())
        student_eu.append(s_eu.mean())
        student_au.append(s_au.mean())

        t_eu = d["teacher_eu"]
        if d["teacher_tu"] is not None:
            teacher_tu.append(d["teacher_tu"].mean())
            teacher_eu.append(t_eu.mean())
            teacher_au.append(d["teacher_au"].mean())
        else:
            teacher_tu.append(None)
            teacher_eu.append(t_eu.mean())
            teacher_au.append(None)

    fig, ax = plt.subplots(figsize=(max(12, n * 2.2), 6))
    x = np.arange(n)
    width = 0.35

    for i in range(n):
        if teacher_au[i] is not None:
            ax.bar(x[i] - width/2, teacher_au[i], width, color="#85c1e9",
                   label="Teacher AU" if i == 0 else "")
            ax.bar(x[i] - width/2, teacher_eu[i], width, bottom=teacher_au[i],
                   color="#2980b9", label="Teacher EU" if i == 0 else "")
        else:
            ax.bar(x[i] - width/2, teacher_eu[i], width, color="#2980b9",
                   label="Teacher EU" if i == 0 else "")

    s_au_arr = np.array(student_au)
    s_eu_arr = np.array(student_eu)
    ax.bar(x + width/2, s_au_arr, width, color="#f5b7b1", label="Student AU")
    ax.bar(x + width/2, s_eu_arr, width, bottom=s_au_arr, color="#e74c3c", label="Student EU")

    for i in range(n):
        if teacher_tu[i] is not None:
            ax.plot(x[i] - width/2, teacher_tu[i], "v", color="#1a5276", ms=6)
        ax.plot(x[i] + width/2, student_tu[i], "v", color="#922b21", ms=6)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Uncertainty (nats)")
    ax.set_title("Uncertainty Decomposition: TU = AU + EU\n"
                 "Teacher (left, blue) vs Student (right, red)", fontsize=13)
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.set_xlim(-0.5, n - 0.5)
    fig.tight_layout()
    path = os.path.join(out_dir, "uncertainty_decomposition.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_student_decomposition_dists(all_eu, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (utype, label) in zip(axes, [("tu", "Total Uncertainty (TU)"),
                                          ("eu", "Epistemic Uncertainty (EU)"),
                                          ("au", "Aleatoric Uncertainty (AU)")]):
        all_vals = []
        for name, d in all_eu.items():
            s_tu = entropy_np(d["student_probs"])
            s_eu = d["student_eu"]
            s_au = np.maximum(s_tu - s_eu, 0.0)
            vals = {"tu": s_tu, "eu": s_eu, "au": s_au}[utype]
            all_vals.append(vals)
        combined = np.concatenate(all_vals)
        bins = np.linspace(0, np.percentile(combined, 99.5), 50)
        for name, vals in zip(all_eu.keys(), all_vals):
            color = COLORS.get(name, "#7f8c8d")
            ax.hist(vals, bins=bins, alpha=0.45, color=color, density=True,
                    histtype="stepfilled", edgecolor=color, linewidth=1.0,
                    label=f"{name} (μ={vals.mean():.3f})")
        ax.set_xlabel("Uncertainty (nats)")
        ax.set_ylabel("Density")
        ax.set_title(label, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
        ax.set_xlim(left=0)
    fig.suptitle("Student Model — Uncertainty Decomposition by Dataset", fontsize=15, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "student_tu_eu_au_dist.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_teacher_decomposition_dists(all_eu, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (utype, label) in zip(axes, [("tu", "Total Uncertainty (TU)"),
                                          ("eu", "Epistemic Uncertainty (EU)"),
                                          ("au", "Aleatoric Uncertainty (AU)")]):
        all_vals, valid_names = [], []
        for name, d in all_eu.items():
            if utype == "eu":
                vals = d["teacher_eu"]
            elif d["teacher_tu"] is not None:
                vals = {"tu": d["teacher_tu"], "au": d["teacher_au"]}[utype]
            else:
                continue
            all_vals.append(vals)
            valid_names.append(name)
        if not all_vals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, fontweight="bold")
            continue
        combined = np.concatenate(all_vals)
        bins = np.linspace(0, np.percentile(combined, 99.5), 50)
        for name, vals in zip(valid_names, all_vals):
            color = COLORS.get(name, "#7f8c8d")
            ax.hist(vals, bins=bins, alpha=0.45, color=color, density=True,
                    histtype="stepfilled", edgecolor=color, linewidth=1.0,
                    label=f"{name} (μ={vals.mean():.3f})")
        ax.set_xlabel("Uncertainty (nats)")
        ax.set_ylabel("Density")
        ax.set_title(label, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
        ax.set_xlim(left=0)
    fig.suptitle("BNN Teacher — Uncertainty Decomposition by Dataset", fontsize=15, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "teacher_tu_eu_au_dist.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_teacher_vs_student_decomposition(all_eu, out_dir):
    d = all_eu.get("Clean MNIST")
    if d is None or d["teacher_tu"] is None:
        print("  Skipping teacher-vs-student decomposition (no teacher TU)")
        return

    t_tu, t_eu, t_au = d["teacher_tu"], d["teacher_eu"], d["teacher_au"]
    s_tu = entropy_np(d["student_probs"])
    s_eu = d["student_eu"]
    s_au = np.maximum(s_tu - s_eu, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    bins_tu = np.linspace(0, np.percentile(np.concatenate([t_tu, s_tu]), 99), 50)
    bins_eu = np.linspace(0, np.percentile(np.concatenate([t_eu, s_eu]), 99), 50)
    bins_au = np.linspace(0, np.percentile(np.concatenate([t_au, s_au]), 99), 50)

    for ax, t_vals, s_vals, bins, label in zip(
            axes, [t_tu, t_eu, t_au], [s_tu, s_eu, s_au],
            [bins_tu, bins_eu, bins_au], ["TU", "EU", "AU"]):
        ax.hist(t_vals, bins=bins, alpha=0.55, color="#3498db", density=True,
                histtype="stepfilled", edgecolor="#3498db", linewidth=1.2,
                label=f"Teacher (μ={t_vals.mean():.4f})")
        ax.hist(s_vals, bins=bins, alpha=0.55, color="#e74c3c", density=True,
                histtype="stepfilled", edgecolor="#e74c3c", linewidth=1.2,
                label=f"Student (μ={s_vals.mean():.4f})")
        ax.set_xlabel(f"{label} (nats)")
        ax.set_ylabel("Density")
        ax.set_title(label, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(left=0)

    fig.suptitle("Clean MNIST Test — Teacher vs Student Decomposition", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "decomposition_teacher_vs_student.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot EU distributions and decomposition (MNIST)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--out_dir", type=str, default="./figures")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    print("Loading model and data...")
    model = load_student(args.save_dir, device)
    data = load_all_targets(args.save_dir)

    print("Collecting EU for all datasets...")
    all_eu = collect_all_eu(model, data, args, device)
    print(f"  Datasets: {list(all_eu.keys())}")

    print("\nGenerating figures...")
    plot_correlation(all_eu, args.out_dir)
    plot_eu_distribution(all_eu, "teacher_eu",
                         "BNN Teacher — EU Distribution by Dataset",
                         "eu_dist_teacher.png", args.out_dir)
    plot_eu_distribution(all_eu, "student_eu",
                         "Student Model — EU Distribution by Dataset",
                         "eu_dist_student.png", args.out_dir)
    plot_violin_comparison(all_eu, args.out_dir)
    plot_decomposition_bars(all_eu, args.out_dir)
    plot_student_decomposition_dists(all_eu, args.out_dir)
    plot_teacher_decomposition_dists(all_eu, args.out_dir)
    plot_teacher_vs_student_decomposition(all_eu, args.out_dir)

    print("\nDone! Figures saved to:", args.out_dir)


if __name__ == "__main__":
    main()
