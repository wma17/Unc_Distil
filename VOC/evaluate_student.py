"""
Comprehensive evaluation of the distilled SegFormer student on Pascal VOC 2012.

Reports:
    1. Segmentation quality (mIoU, pixel accuracy)
    2. Calibration (pixel-level ECE-15, NLL)
    3. Per-pixel EU correlation (Pearson, Spearman)
    4. Boundary vs interior EU ratio
    5. Coverage-mIoU curve (selective segmentation)
    6. Image-level OOD detection AUROC
    7. Inference throughput

Usage:
    python evaluate_student.py --save_dir ./checkpoints --gpu 0
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import create_student, create_teacher, NUM_CLASSES
from data import (VOCSegDataset, SegTransformVal, IGNORE_INDEX,
                  get_dtd_loader, get_coco_loader)


EPS = 1e-8


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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


def compute_miou(pred, target, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0


def compute_pixel_ece(probs_np, labels_np, n_bins=15, ignore_index=IGNORE_INDEX):
    """Pixel-level ECE."""
    valid = labels_np != ignore_index
    if valid.sum() == 0:
        return 0.0
    probs_flat = probs_np[valid]
    labels_flat = labels_np[valid]
    confidences = probs_flat.max(axis=-1)
    predictions = probs_flat.argmax(axis=-1)
    accuracies = (predictions == labels_flat).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels_flat)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() > 0:
            ece += (in_bin.sum() / n) * abs(
                accuracies[in_bin].mean() - confidences[in_bin].mean())
    return ece


def compute_pixel_nll(probs_np, labels_np, ignore_index=IGNORE_INDEX):
    valid = labels_np != ignore_index
    if valid.sum() == 0:
        return 0.0
    probs_flat = probs_np[valid]
    labels_flat = labels_np[valid]
    n = len(labels_flat)
    return -np.log(probs_flat[np.arange(n), labels_flat] + EPS).mean()


def auroc(scores_neg, scores_pos):
    labels = np.concatenate([np.zeros(len(scores_neg)), np.ones(len(scores_pos))])
    scores = np.concatenate([scores_neg, scores_pos])
    order = np.argsort(-scores)
    labels = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp, tpr_sum = 0, 0.0
    for lab in labels:
        if lab == 1:
            tp += 1
        else:
            tpr_sum += tp / n_pos
    return tpr_sum / n_neg


def find_boundary_pixels(mask, distance=3):
    """Find pixels within `distance` of a class boundary."""
    from scipy.ndimage import binary_dilation
    boundary = np.zeros_like(mask, dtype=bool)
    for c in range(NUM_CLASSES):
        c_mask = mask == c
        dilated = binary_dilation(c_mask, iterations=distance)
        boundary |= (dilated & ~c_mask)
    # Exclude ignore pixels
    boundary &= (mask != IGNORE_INDEX)
    return boundary


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_student_batch(model, imgs, device):
    """Returns logits, eu_map for a batch."""
    imgs = imgs.to(device)
    logits, eu = model(imgs, return_eu=True)
    return logits.cpu(), eu.cpu()


@torch.no_grad()
def measure_throughput(model, device, img_shape=(3, 512, 512), batch_size=4,
                       n_batches=50, n_warmup=5):
    dummy = torch.randn(batch_size, *img_shape, device=device)
    for _ in range(n_warmup):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_batches):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (batch_size * n_batches) / (time.time() - t0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate VOC segmentation student")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    # Load student
    model = create_student(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(os.path.join(args.save_dir, "student.pt"),
                       map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded student (mIoU={ckpt.get('val_miou', '?')}, "
          f"EU Spearman={ckpt.get('eu_spearman', '?')})")

    # Load cached targets
    data = np.load(os.path.join(args.save_dir, "teacher_targets.npz"), allow_pickle=True)
    eu_map_size = int(data["eu_map_size"]) if "eu_map_size" in data else 128

    # Load val dataset
    val_transform = SegTransformVal(crop_size=512)
    val_ds = VOCSegDataset(args.data_dir, split="val", transform=val_transform,
                            use_sbd=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Run inference on val
    all_preds, all_masks, all_probs = [], [], []
    all_student_eu, all_teacher_eu = [], []

    idx = 0
    for imgs, masks in val_loader:
        logits, eu = predict_student_batch(model, imgs, device)
        # Upsample logits to mask resolution
        logits_up = F.interpolate(logits, size=masks.shape[1:],
                                   mode="bilinear", align_corners=False)
        probs = F.softmax(logits_up, dim=1)
        preds = logits_up.argmax(1)

        bs = imgs.size(0)
        for b in range(bs):
            all_preds.append(preds[b].numpy())
            all_masks.append(masks[b].numpy())
            all_probs.append(probs[b].numpy())

            # Resize EU to eu_map_size for comparison with teacher
            eu_resized = F.interpolate(eu[b:b+1].unsqueeze(0),
                                        size=(eu_map_size, eu_map_size),
                                        mode="bilinear", align_corners=False)
            all_student_eu.append(eu_resized.squeeze().numpy())

            if idx + b < len(data["val_eu"]):
                all_teacher_eu.append(data["val_eu"][idx + b].astype(np.float32))
        idx += bs

    # ======================================================================
    # 1. Segmentation Quality
    # ======================================================================
    mious = [compute_miou(p, m) for p, m in zip(all_preds, all_masks)]
    mean_miou = np.mean(mious)

    total_correct, total_valid = 0, 0
    for p, m in zip(all_preds, all_masks):
        valid = m != IGNORE_INDEX
        total_correct += (p[valid] == m[valid]).sum()
        total_valid += valid.sum()
    pixel_acc = total_correct / max(1, total_valid) * 100

    print(f"\n{'='*60}")
    print(f"  1. Segmentation Quality (VOC val)")
    print(f"{'='*60}")
    print(f"  Student mIoU: {mean_miou:.4f}")
    print(f"  Pixel accuracy: {pixel_acc:.2f}%")

    # ======================================================================
    # 2. Calibration
    # ======================================================================
    # Stack probs and labels for pixel-level calibration
    all_probs_flat = np.concatenate([p.reshape(NUM_CLASSES, -1).T for p in all_probs], axis=0)
    all_labels_flat = np.concatenate([m.reshape(-1) for m in all_masks], axis=0)

    ece = compute_pixel_ece(all_probs_flat, all_labels_flat)
    nll = compute_pixel_nll(all_probs_flat, all_labels_flat)

    print(f"\n{'='*60}")
    print(f"  2. Pixel-level Calibration")
    print(f"{'='*60}")
    print(f"  ECE-15: {ece:.4f}")
    print(f"  NLL:    {nll:.4f}")

    # ======================================================================
    # 3. Per-pixel EU Correlation
    # ======================================================================
    n_common = min(len(all_student_eu), len(all_teacher_eu))
    if n_common > 0:
        stu_eu_flat = torch.from_numpy(
            np.concatenate([e.reshape(-1) for e in all_student_eu[:n_common]]))
        tea_eu_flat = torch.from_numpy(
            np.concatenate([e.reshape(-1) for e in all_teacher_eu[:n_common]]))

        r_p = pearson_corr(stu_eu_flat, tea_eu_flat)
        r_s = spearman_corr(stu_eu_flat, tea_eu_flat)

        print(f"\n{'='*60}")
        print(f"  3. Per-pixel EU Correlation (VOC val)")
        print(f"{'='*60}")
        print(f"  Pearson:  {r_p:.4f}")
        print(f"  Spearman: {r_s:.4f}")

    # ======================================================================
    # 4. Boundary vs Interior EU Ratio
    # ======================================================================
    try:
        boundary_eus, interior_eus = [], []
        for i in range(min(n_common, len(all_masks))):
            mask = all_masks[i]
            eu_map = all_student_eu[i]
            # Resize mask to eu_map size
            mask_resized = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            mask_resized = F.interpolate(mask_resized,
                                          size=(eu_map_size, eu_map_size),
                                          mode="nearest").squeeze().numpy().astype(int)
            boundary = find_boundary_pixels(mask_resized, distance=3)
            interior = (mask_resized != IGNORE_INDEX) & ~boundary
            if boundary.sum() > 0:
                boundary_eus.append(eu_map[boundary].mean())
            if interior.sum() > 0:
                interior_eus.append(eu_map[interior].mean())

        if boundary_eus and interior_eus:
            mean_boundary = np.mean(boundary_eus)
            mean_interior = np.mean(interior_eus)
            ratio = mean_boundary / max(mean_interior, EPS)
            print(f"\n{'='*60}")
            print(f"  4. Boundary vs Interior EU")
            print(f"{'='*60}")
            print(f"  Boundary mean EU: {mean_boundary:.4f}")
            print(f"  Interior mean EU: {mean_interior:.4f}")
            print(f"  Ratio (boundary/interior): {ratio:.2f}")
    except ImportError:
        print("\n  [skip] Boundary analysis requires scipy")

    # ======================================================================
    # 5. Coverage-mIoU curve
    # ======================================================================
    print(f"\n{'='*60}")
    print(f"  5. Selective Segmentation (Coverage-mIoU)")
    print(f"{'='*60}")

    # Flatten per-pixel errors and EU scores
    errors_flat = []
    eu_flat = []
    for i in range(len(all_preds)):
        mask = all_masks[i]
        pred = all_preds[i]
        valid = mask != IGNORE_INDEX
        pixel_errors = (pred[valid] != mask[valid]).astype(float)
        errors_flat.append(pixel_errors)
        # Resize student EU to mask resolution
        eu_up = F.interpolate(
            torch.from_numpy(all_student_eu[i]).unsqueeze(0).unsqueeze(0).float(),
            size=mask.shape, mode="bilinear", align_corners=False
        ).squeeze().numpy()
        eu_flat.append(eu_up[valid])

    errors_all = np.concatenate(errors_flat)
    eu_all = np.concatenate(eu_flat)

    # Sort by EU ascending
    order = np.argsort(eu_all)
    sorted_errors = errors_all[order]
    n = len(sorted_errors)

    print(f"  {'Coverage':>10} {'Accuracy':>10}")
    print(f"  {'-'*22}")
    for cov in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k = max(1, int(cov * n))
        acc = 1.0 - sorted_errors[:k].mean()
        print(f"  {cov:>10.1%} {acc:>10.4f}")

    # AURC
    coverages = np.arange(1, n + 1) / n
    risks = np.cumsum(sorted_errors) / np.arange(1, n + 1)
    aurc = float(np.trapz(risks, coverages))
    print(f"  AURC: {aurc:.6f}")

    # ======================================================================
    # 6. Image-level OOD Detection
    # ======================================================================
    print(f"\n{'='*60}")
    print(f"  6. Image-level OOD Detection")
    print(f"{'='*60}")

    # ID scores: mean per-pixel EU per image
    id_scores = np.array([eu.mean() for eu in all_student_eu])
    id_tea_scores = np.array([eu.mean() for eu in all_teacher_eu[:n_common]])

    # DTD (far-OOD)
    dtd_loader = get_dtd_loader(args.data_dir, batch_size=args.batch_size)
    if dtd_loader is not None:
        dtd_scores = []
        with torch.no_grad():
            for imgs, _ in dtd_loader:
                imgs = imgs.to(device)
                _, eu = model(imgs, return_eu=True)
                for b in range(eu.shape[0]):
                    dtd_scores.append(eu[b].mean().item())
        dtd_scores = np.array(dtd_scores)
        a = auroc(id_scores, dtd_scores)
        print(f"  VOC vs DTD (far-OOD):  Student EU AUROC = {a:.4f}")

    # COCO (near-OOD)
    coco_loader = get_coco_loader(args.data_dir, batch_size=args.batch_size)
    if coco_loader is not None:
        coco_scores = []
        with torch.no_grad():
            for imgs, _ in coco_loader:
                imgs = imgs.to(device)
                _, eu = model(imgs, return_eu=True)
                for b in range(eu.shape[0]):
                    coco_scores.append(eu[b].mean().item())
        coco_scores = np.array(coco_scores)
        a = auroc(id_scores, coco_scores)
        print(f"  VOC vs COCO (near-OOD): Student EU AUROC = {a:.4f}")

    # ======================================================================
    # 7. Throughput
    # ======================================================================
    print(f"\n{'='*60}")
    print(f"  7. Inference Throughput (device={device})")
    print(f"{'='*60}")

    model.eval()
    stu_tp = measure_throughput(model, device, img_shape=(3, 512, 512),
                                 batch_size=args.batch_size)
    print(f"  Student (single pass): {stu_tp:.1f} images/sec")

    # ======================================================================
    # 8. Summary
    # ======================================================================
    summary = {
        "experiment": "E4_VOC",
        "student_miou": float(mean_miou),
        "pixel_accuracy": float(pixel_acc),
        "pixel_ece": float(ece),
        "pixel_nll": float(nll),
        "eu_pearson": float(r_p) if n_common > 0 else None,
        "eu_spearman": float(r_s) if n_common > 0 else None,
    }
    summary_path = os.path.join(args.save_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
