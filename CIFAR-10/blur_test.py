"""
Blur Test: Iterative blur by uncertainty contribution — uncertainty reduction curve.

Workflow:
1. For each test image: use student EU + IG to get an uncertainty attribution map
2. Rank pixels by their contribution to uncertainty (IG value, highest first)
3. Iteratively blur pixels by contribution: at step k, blur the top k% of pixels
   using a Gaussian filter (mean 0, std σ)
4. Refeed the blurred image and evaluate uncertainty at each step
5. Record the uncertainty reduction curve (EU vs % pixels blurred)

This demonstrates effectiveness: if IG correctly identifies uncertainty-contributing
regions, blurring them in order of contribution should yield a steep EU reduction.

Usage:
    python blur_test.py --save_dir ./checkpoints --out_dir ./blur_test_results --gpu 0
    python blur_test.py --save_dir ./checkpoints --max_samples 500 --ig_steps 20 --blur_levels 0,1,2,5,10,20,50,100
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import gaussian_blur

from evaluate_student import load_student
from uncertainty_cam import (
    compute_student_eu_ig,
    CIFAR10_MEAN,
    CIFAR10_STD,
)


def apply_blur_to_top_uncertain(image_tensor, ig_map, top_percent, kernel_size=5, sigma=2.0):
    """
    Blur only the top `top_percent` most uncertain pixels (by IG attribution).

    Uses a Gaussian filter with mean 0 and standard deviation σ.

    Args:
        image_tensor: (1, 3, H, W) normalized CIFAR image
        ig_map: (H, W) attribution map from student EU IG
        top_percent: float in (0, 100], e.g. 2 or 5
        kernel_size: spatial kernel size
        sigma: Gaussian std (mean=0)
    Returns:
        blurred_image: (1, 3, H, W) same shape, only top uncertain region blurred
    """
    h, w = ig_map.shape
    n_pixels = h * w
    k = max(1, int(n_pixels * top_percent / 100))

    # Flatten and get indices of top-k highest attribution
    flat = ig_map.flatten()
    top_idx_flat = np.argsort(-flat)[:k]

    # Create binary mask (1 = top uncertain, 0 = keep original)
    mask = np.zeros((h, w), dtype=np.float32)
    for idx in top_idx_flat:
        i, j = idx // w, idx % w
        mask[i, j] = 1.0

    # Expand mask to (1, 3, H, W)
    mask_t = torch.from_numpy(mask).to(
        device=image_tensor.device, dtype=image_tensor.dtype
    ).view(1, 1, h, w).expand_as(image_tensor)

    # Blur the full image
    blurred = gaussian_blur(image_tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

    # Blend: original where mask=0, blurred where mask=1
    out = image_tensor * (1 - mask_t) + blurred * mask_t
    return out


def run_blur_test(student, test_loader, device, out_dir, ig_steps=50, blur_levels=(0, 1, 2, 5, 10, 20, 50, 100),
                  max_samples=None, blur_kernel=5, blur_sigma=2.0):
    """
    Run iterative blur test: blur pixels by uncertainty contribution (IG rank).

    At each blur level (e.g. 0%, 1%, 2%, 5%, ...), we blur the top X% of pixels
    (by IG attribution) with a Gaussian filter (mean 0, std σ), then evaluate EU.

    Returns:
        results: dict with curve data (pct_blurred, eu_mean, eu_std, acc per level)
    """
    blur_levels = tuple(sorted(set(blur_levels)))
    all_images = []
    all_labels = []
    for batch_x, batch_y in test_loader:
        all_images.append(batch_x)
        all_labels.append(batch_y)
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    n_total = len(all_images)
    if max_samples is not None:
        n_total = min(n_total, max_samples)
    all_images = all_images[:n_total]
    all_labels = all_labels[:n_total]

    print(f"Blur test on {n_total} samples, blur_levels={blur_levels}%")
    print(f"  Gaussian blur: kernel={blur_kernel}, σ={blur_sigma} (mean=0)")

    # Per-level storage: (n_levels, n_samples)
    n_levels = len(blur_levels)
    eu_at_level = np.zeros((n_levels, n_total), dtype=np.float32)
    pred_at_level = np.zeros((n_levels, n_total), dtype=np.int64)

    # Pre-process: level 0 = no blur
    with torch.no_grad():
        for i in range(0, n_total, 256):
            batch = all_images[i:i + 256].to(device)
            logits, eu = student(batch)
            eu_at_level[0, i:i + batch.size(0)] = eu.cpu().numpy()
            pred_at_level[0, i:i + batch.size(0)] = logits.argmax(dim=1).cpu().numpy()

    student.eval()
    for idx in range(n_total):
        x = all_images[idx:idx + 1].to(device)
        ig_map = compute_student_eu_ig(student, x, device, steps=ig_steps)

        for level_idx, pct in enumerate(blur_levels):
            if pct == 0:
                continue  # already filled above
            x_blur = apply_blur_to_top_uncertain(
                x.clone(), ig_map, top_percent=float(pct),
                kernel_size=blur_kernel, sigma=blur_sigma
            )
            with torch.no_grad():
                logits, eu = student(x_blur)
            eu_at_level[level_idx, idx] = eu.item()
            pred_at_level[level_idx, idx] = logits.argmax(dim=1).item()

        if (idx + 1) % 500 == 0:
            print(f"    Processed {idx + 1}/{n_total}")

    # Aggregate curve
    labels_np = all_labels.numpy()
    curve = []
    for level_idx, pct in enumerate(blur_levels):
        eu_mean = float(eu_at_level[level_idx].mean())
        eu_std = float(eu_at_level[level_idx].std())
        acc = (pred_at_level[level_idx] == labels_np).mean() * 100
        curve.append({
            "pct_blurred": pct,
            "eu_mean": eu_mean,
            "eu_std": eu_std,
            "acc": acc,
        })

    results = {
        "n_samples": n_total,
        "blur_levels": blur_levels,
        "blur_sigma": blur_sigma,
        "curve": curve,
        "eu_at_level": eu_at_level,
        "pred_at_level": pred_at_level,
    }

    # Save curve CSV
    os.makedirs(out_dir, exist_ok=True)
    curve_path = os.path.join(out_dir, "uncertainty_reduction_curve.csv")
    with open(curve_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pct_blurred", "eu_mean", "eu_std", "acc"])
        for row in curve:
            writer.writerow([row["pct_blurred"], f"{row['eu_mean']:.6f}", f"{row['eu_std']:.6f}", f"{row['acc']:.2f}"])

    # Per-sample CSV (one row per sample, columns for each level)
    per_sample_path = os.path.join(out_dir, "blur_test_per_sample.csv")
    with open(per_sample_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["idx", "true_label"] + [f"eu_pct{p}" for p in blur_levels] + [f"correct_pct{p}" for p in blur_levels]
        writer.writerow(header)
        for i in range(n_total):
            correct_at_level = (pred_at_level[:, i] == labels_np[i]).astype(int)
            row = [i, int(labels_np[i])] + [f"{eu_at_level[level_idx, i]:.6f}" for level_idx in range(n_levels)] + list(correct_at_level)
            writer.writerow(row)

    # Plot uncertainty reduction curve
    fig, ax = plt.subplots(figsize=(8, 5))
    pcts = [r["pct_blurred"] for r in curve]
    eu_means = [r["eu_mean"] for r in curve]
    eu_stds = [r["eu_std"] for r in curve]
    ax.plot(pcts, eu_means, "o-", color="#2ecc71", linewidth=2, markersize=8, label="EU mean")
    ax.fill_between(pcts, np.array(eu_means) - np.array(eu_stds), np.array(eu_means) + np.array(eu_stds), alpha=0.3, color="#2ecc71")
    ax.set_xlabel("% of pixels blurred (by IG contribution)", fontsize=12)
    ax.set_ylabel("Epistemic uncertainty (EU)", fontsize=12)
    ax.set_title("Uncertainty reduction curve — iterative blur by contribution", fontsize=13)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "uncertainty_reduction_curve.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results


def print_results(results):
    print("\n" + "=" * 70)
    print("  UNCERTAINTY REDUCTION CURVE (iterative blur by IG contribution)")
    print("=" * 70)
    print(f"  Samples: {results['n_samples']}   |   Gaussian σ={results['blur_sigma']}")
    print()
    print(f"  {'% blurred':>10} {'EU mean':>12} {'EU std':>10} {'Acc (%)':>10}")
    print(f"  {'-'*44}")
    for row in results["curve"]:
        print(f"  {row['pct_blurred']:>10} {row['eu_mean']:>12.6f} {row['eu_std']:>10.6f} {row['acc']:>10.2f}")
    print("=" * 70)
    print(f"  Curve saved to uncertainty_reduction_curve.png")


def main():
    parser = argparse.ArgumentParser(description="Blur test: iterative blur by uncertainty contribution")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--out_dir", type=str, default="./blur_test_results")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ig_steps", type=int, default=50, help="IG integration steps (higher=slower, more accurate)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of test samples (None = full test set)")
    parser.add_argument("--blur_levels", type=str, default="0,1,2,5,10,20,50,100",
                        help="Comma-separated % of pixels to blur at each step (e.g. 0,1,2,5,10,20,50,100)")
    parser.add_argument("--blur_kernel", type=int, default=5)
    parser.add_argument("--blur_sigma", type=float, default=2.0,
                        help="Gaussian blur std (mean=0)")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    blur_levels = tuple(int(x.strip()) for x in args.blur_levels.split(",") if x.strip())
    if not blur_levels:
        blur_levels = (0, 1, 2, 5, 10, 20, 50, 100)

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=args.num_workers)

    print("Loading student...")
    student = load_student(args.save_dir, device)

    results = run_blur_test(
        student, test_loader, device, args.out_dir,
        ig_steps=args.ig_steps,
        blur_levels=blur_levels,
        max_samples=args.max_samples,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
    )

    print_results(results)
    print(f"\nResults saved to {args.out_dir}/")
    print(f"  - uncertainty_reduction_curve.csv")
    print(f"  - uncertainty_reduction_curve.png")
    print(f"  - blur_test_per_sample.csv")


if __name__ == "__main__":
    main()
