"""
Comprehensive evaluation of the distilled student model.

Reports:
    1. Accuracy comparison     — student vs teacher (Bayesian ensemble)
    2. Correctness agreement   — fraction of samples where both agree
    3. EU correlation          — Pearson + Spearman per dataset (clean, corrupted, OOD)
    4. OOD detection AUROC     — teacher EU vs student EU vs baselines
       Seen OOD  (used in Phase 2): SVHN, CIFAR-100
       Unseen OOD (held out):       MNIST, FashionMNIST, STL10, DTD

Usage:
    python evaluate_student.py --save_dir ./checkpoints --gpu 0
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from models import cifar_resnet18, cifar_resnet18_student
from cache_ensemble_targets import apply_corruption, CORRUPTION_TYPES, CORRUPTION_SEED


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
EPS = 1e-8

SEEN_OOD = [
    ("svhn", "svhn_eu", "SVHN"),
    ("cifar100", "cifar100_eu", "CIFAR-100"),
]
UNSEEN_OOD = [
    ("mnist", "mnist_eu", "MNIST"),
    ("fashionmnist", "fashionmnist_eu", "FashionMNIST"),
    ("stl10", "stl10_eu", "STL10"),
    ("dtd", "dtd_eu", "DTD"),
]


# ---------------------------------------------------------------------------
# Correlation & AUROC
# ---------------------------------------------------------------------------

def pearson_corr(a, b):
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def spearman_corr(a, b):
    def _rank(x):
        order = x.argsort()
        ranks = torch.empty_like(x)
        ranks[order] = torch.arange(len(x), dtype=x.dtype)
        return ranks
    return torch.corrcoef(torch.stack([_rank(a), _rank(b)]))[0, 1].item()


def compute_ece(probs_np, labels_np, n_bins=15):
    """15-bin Expected Calibration Error."""
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
                accuracies[in_bin].mean() - confidences[in_bin].mean()
            )
    return ece


def compute_nll(probs_np, labels_np):
    """Mean negative log-likelihood."""
    n = len(labels_np)
    return -np.log(probs_np[np.arange(n), labels_np] + EPS).mean()


def compute_brier(probs_np, labels_np):
    """Mean Brier score (sum of squared differences from one-hot)."""
    n = len(labels_np)
    one_hot = np.zeros_like(probs_np)
    one_hot[np.arange(n), labels_np] = 1.0
    return ((probs_np - one_hot) ** 2).sum(axis=-1).mean()


def compute_aurc(errors_np, scores_np):
    """AURC: sort samples by scores ascending (low score = most certain = included first).

    Returns (aurc, oracle_aurc, oracle_gap, acc_at_90, acc_at_80).
    Lower AURC is better.
    """
    n = len(errors_np)
    order = np.argsort(scores_np)
    sorted_errors = errors_np[order]
    coverages = np.arange(1, n + 1) / n
    risks = np.cumsum(sorted_errors) / np.arange(1, n + 1)
    aurc = float(np.trapz(risks, coverages))
    oracle_risks = np.cumsum(errors_np[np.argsort(errors_np)]) / np.arange(1, n + 1)
    oracle_aurc = float(np.trapz(oracle_risks, coverages))
    k90 = max(1, int(0.9 * n))
    k80 = max(1, int(0.8 * n))
    acc_at_90 = 1.0 - sorted_errors[:k90].mean()
    acc_at_80 = 1.0 - sorted_errors[:k80].mean()
    return aurc, oracle_aurc, aurc - oracle_aurc, acc_at_90, acc_at_80


@torch.no_grad()
def measure_throughput(fn, img_shape, device, batch_size=256, n_batches=100, n_warmup=10):
    """Measure forward-pass throughput in samples/sec."""
    dummy = torch.randn(batch_size, *img_shape, device=device)
    for _ in range(n_warmup):
        fn(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_batches):
        fn(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (batch_size * n_batches) / (time.time() - t0)


def auroc(scores_neg, scores_pos):
    """AUROC = P(score_pos > score_neg) via Wilcoxon-Mann-Whitney."""
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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_student(save_dir, device):
    path = os.path.join(save_dir, "student.pt")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = cifar_resnet18_student(num_classes=10).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded student from {path}")
    print(f"  Phase 1 acc={ckpt.get('test_acc', '?')}%  "
          f"EU Pearson={ckpt.get('eu_pearson', '?')}  Spearman={ckpt.get('eu_spearman', '?')}")
    return model


def load_all_targets(save_dir):
    return np.load(os.path.join(save_dir, "teacher_targets.npz"))


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def get_test_loader(data_dir, batch_size=256, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def get_ood_loader(ood_name, data_dir, batch_size=256, num_workers=4):
    """Load any supported OOD dataset."""
    rgb_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    gray_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    loaders = {
        "svhn": lambda: datasets.SVHN(
            root=os.path.join(data_dir, "svhn"), split="test",
            download=True, transform=rgb_transform),
        "cifar100": lambda: datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=rgb_transform),
        "mnist": lambda: datasets.MNIST(
            root=data_dir, train=False, download=True, transform=gray_transform),
        "fashionmnist": lambda: datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=gray_transform),
        "stl10": lambda: datasets.STL10(
            root=data_dir, split="test", download=True, transform=rgb_transform),
        "dtd": lambda: datasets.DTD(
            root=data_dir, split="test", download=True, transform=rgb_transform),
    }
    if ood_name not in loaders:
        raise ValueError(f"Unknown OOD dataset: {ood_name}. Available: {list(loaders.keys())}")
    ds = loaders[ood_name]()
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def get_corrupted_test_loader(data_dir, corruption_type, batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    imgs = torch.cat([x for x, _ in loader], 0)
    corrupted = apply_corruption(imgs, corruption_type, seed=CORRUPTION_SEED)
    ds = TensorDataset(corrupted, torch.zeros(len(corrupted), dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_student(model, loader, device):
    all_probs, all_eu = [], []
    for batch in loader:
        imgs = batch[0].to(device)
        logits, eu = model(imgs)
        all_probs.append(F.softmax(logits, dim=-1).cpu())
        all_eu.append(eu.cpu())
    return torch.cat(all_probs), torch.cat(all_eu)


# ---------------------------------------------------------------------------
# OOD detection table
# ---------------------------------------------------------------------------

def print_ood_table(id_name, ood_name,
                    id_student_eu, ood_student_eu,
                    id_teacher_eu, ood_teacher_eu,
                    id_student_probs, ood_student_probs):
    print(f"\n  {id_name} (neg) vs {ood_name} (pos)")
    print(f"  {'Method':<35} {'AUROC':>10}")
    print(f"  {'-'*45}")

    a = auroc(id_teacher_eu, ood_teacher_eu)
    print(f"  {'Teacher EU (ensemble)':<35} {a:>10.4f}")

    a = auroc(id_student_eu, ood_student_eu)
    print(f"  {'Student EU (learned)':<35} {a:>10.4f}")

    id_ent = -(id_student_probs * np.log(id_student_probs + EPS)).sum(axis=-1)
    ood_ent = -(ood_student_probs * np.log(ood_student_probs + EPS)).sum(axis=-1)
    a = auroc(id_ent, ood_ent)
    print(f"  {'Student entropy (softmax)':<35} {a:>10.4f}")

    id_mp = 1.0 - id_student_probs.max(axis=-1)
    ood_mp = 1.0 - ood_student_probs.max(axis=-1)
    a = auroc(id_mp, ood_mp)
    print(f"  {'1 - max softmax prob':<35} {a:>10.4f}")

    print(f"  ── EU stats ──")
    print(f"  {id_name:>20}  teacher={id_teacher_eu.mean():.4f}±{id_teacher_eu.std():.4f}  "
          f"student={id_student_eu.mean():.4f}±{id_student_eu.std():.4f}")
    print(f"  {ood_name:>20}  teacher={ood_teacher_eu.mean():.4f}±{ood_teacher_eu.std():.4f}  "
          f"student={ood_student_eu.mean():.4f}±{ood_student_eu.std():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate distilled student model")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    model = load_student(args.save_dir, device)
    data = load_all_targets(args.save_dir)

    teacher_probs_t = torch.from_numpy(data["test_probs"]).float()
    teacher_eu_t = torch.from_numpy(data["test_eu"]).float()
    true_labels_t = torch.from_numpy(data["test_labels"]).long()

    test_loader = get_test_loader(args.data_dir, args.batch_size, args.num_workers)
    student_probs, student_eu = predict_student(model, test_loader, device)
    student_preds = student_probs.argmax(dim=1)
    teacher_preds = teacher_probs_t.argmax(dim=1)

    # ======================================================================
    # 1. Accuracy
    # ======================================================================
    student_acc = student_preds.eq(true_labels_t).float().mean().item() * 100
    teacher_acc = teacher_preds.eq(true_labels_t).float().mean().item() * 100

    print(f"\n{'='*60}")
    print(f"  1. Accuracy")
    print(f"{'='*60}")
    print(f"  Teacher (ensemble):  {teacher_acc:.2f}%")
    print(f"  Student (distilled): {student_acc:.2f}%")

    # ======================================================================
    # 1b. Calibration Metrics (ECE-15, NLL, Brier)
    # ======================================================================
    labels_np = true_labels_t.numpy()
    stu_probs_np = student_probs.numpy()
    tea_probs_np = teacher_probs_t.numpy()

    stu_ece = compute_ece(stu_probs_np, labels_np)
    stu_nll = compute_nll(stu_probs_np, labels_np)
    stu_brier = compute_brier(stu_probs_np, labels_np)
    tea_ece = compute_ece(tea_probs_np, labels_np)
    tea_nll = compute_nll(tea_probs_np, labels_np)
    tea_brier = compute_brier(tea_probs_np, labels_np)

    print(f"\n{'='*60}")
    print(f"  1b. Calibration Metrics (clean CIFAR-10 test)")
    print(f"{'='*60}")
    print(f"  {'Model':<30} {'ECE-15':>10} {'NLL':>10} {'Brier':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Teacher (ensemble)':<30} {tea_ece:>10.4f} {tea_nll:>10.4f} {tea_brier:>10.4f}")
    print(f"  {'Student (distilled)':<30} {stu_ece:>10.4f} {stu_nll:>10.4f} {stu_brier:>10.4f}")

    # ======================================================================
    # 2. Correctness agreement
    # ======================================================================
    student_correct = student_preds.eq(true_labels_t)
    teacher_correct = teacher_preds.eq(true_labels_t)

    both_correct = (student_correct & teacher_correct).sum().item()
    both_wrong = (~student_correct & ~teacher_correct).sum().item()
    student_only = (student_correct & ~teacher_correct).sum().item()
    teacher_only = (~student_correct & teacher_correct).sum().item()
    n = len(true_labels_t)
    agreement = (both_correct + both_wrong) / n * 100

    print(f"\n{'='*60}")
    print(f"  2. Correctness Agreement")
    print(f"{'='*60}")
    print(f"  Both correct:          {both_correct:5d}  ({both_correct/n*100:.1f}%)")
    print(f"  Both wrong:            {both_wrong:5d}  ({both_wrong/n*100:.1f}%)")
    print(f"  Student correct only:  {student_only:5d}  ({student_only/n*100:.1f}%)")
    print(f"  Teacher correct only:  {teacher_only:5d}  ({teacher_only/n*100:.1f}%)")
    print(f"  Total agreement:       {agreement:.2f}%")
    same_pred = student_preds.eq(teacher_preds).float().mean().item() * 100
    print(f"  Same predicted class:  {same_pred:.2f}%")

    # ======================================================================
    # 3. EU correlation — per dataset
    # ======================================================================
    print(f"\n{'='*60}")
    print(f"  3. EU Correlation (student EU vs teacher EU)")
    print(f"{'='*60}")
    print(f"  {'Dataset':<30} {'Pearson':>10} {'Spearman':>10} {'Stu mean':>10} {'Tea mean':>10}")
    print(f"  {'-'*70}")

    r_p = pearson_corr(student_eu, teacher_eu_t)
    r_s = spearman_corr(student_eu, teacher_eu_t)
    print(f"  {'Clean CIFAR-10 test':<30} {r_p:>10.4f} {r_s:>10.4f} "
          f"{student_eu.mean().item():>10.4f} {teacher_eu_t.mean().item():>10.4f}")

    wrong_mask = ~student_correct
    if wrong_mask.sum() > 10:
        r_p_w = pearson_corr(student_eu[wrong_mask], teacher_eu_t[wrong_mask])
        r_s_w = spearman_corr(student_eu[wrong_mask], teacher_eu_t[wrong_mask])
        print(f"  {'  └ misclassified only':<30} {r_p_w:>10.4f} {r_s_w:>10.4f}")

    # Corrupted CIFAR-10 test
    corrupted_student_eus = {}
    corrupted_teacher_eus = {}
    corrupted_student_probs = {}
    for ctype in CORRUPTION_TYPES:
        key = f"corrupt_{ctype}_test_eu"
        if key not in data:
            continue
        c_teacher_eu = torch.from_numpy(data[key]).float()
        c_loader = get_corrupted_test_loader(args.data_dir, ctype, args.batch_size)
        c_probs, c_student_eu = predict_student(model, c_loader, device)
        corrupted_student_eus[ctype] = c_student_eu
        corrupted_teacher_eus[ctype] = c_teacher_eu
        corrupted_student_probs[ctype] = c_probs
        r_p = pearson_corr(c_student_eu, c_teacher_eu)
        r_s = spearman_corr(c_student_eu, c_teacher_eu)
        print(f"  {f'Corrupted: {ctype}':<30} {r_p:>10.4f} {r_s:>10.4f} "
              f"{c_student_eu.mean().item():>10.4f} {c_teacher_eu.mean().item():>10.4f}")

    # All OOD datasets (seen + unseen) — EU correlation
    # name -> (student_eu, teacher_eu, student_probs, teacher_tu, teacher_au, display_name)
    all_ood_data = {}
    for ood_items in [SEEN_OOD, UNSEEN_OOD]:
        for ood_id, cache_key, display_name in ood_items:
            if cache_key not in data:
                continue
            ood_teacher_eu = torch.from_numpy(data[cache_key]).float()
            tu_key = cache_key.replace("_eu", "_tu")
            au_key = cache_key.replace("_eu", "_au")
            ood_tea_tu = data[tu_key] if tu_key in data else None
            ood_tea_au = data[au_key] if au_key in data else None
            try:
                ood_loader = get_ood_loader(ood_id, args.data_dir, args.batch_size, args.num_workers)
                ood_probs, ood_student_eu = predict_student(model, ood_loader, device)
            except Exception as e:
                print(f"  {f'OOD: {display_name}':<30} SKIPPED ({e})")
                continue
            min_n = min(len(ood_student_eu), len(ood_teacher_eu))
            s_eu = ood_student_eu[:min_n]
            t_eu = ood_teacher_eu[:min_n]
            s_probs = ood_probs[:min_n]
            t_tu_np = ood_tea_tu[:min_n] if ood_tea_tu is not None else None
            t_au_np = ood_tea_au[:min_n] if ood_tea_au is not None else None
            all_ood_data[ood_id] = (s_eu.numpy(), t_eu.numpy(), s_probs.numpy(),
                                    t_tu_np, t_au_np, display_name)
            r_p = pearson_corr(s_eu, t_eu)
            r_s = spearman_corr(s_eu, t_eu)
            print(f"  {f'OOD: {display_name}':<30} {r_p:>10.4f} {r_s:>10.4f} "
                  f"{s_eu.mean().item():>10.4f} {t_eu.mean().item():>10.4f}")

    # ======================================================================
    # 4. OOD Detection AUROC
    # ======================================================================
    clean_stu_eu = student_eu.numpy()
    clean_tea_eu = teacher_eu_t.numpy()
    clean_stu_probs = student_probs.numpy()

    if corrupted_student_eus:
        shifted_stu_eu = torch.cat(list(corrupted_student_eus.values())).numpy()
        shifted_tea_eu = torch.cat(list(corrupted_teacher_eus.values())).numpy()
        shifted_stu_probs = torch.cat(list(corrupted_student_probs.values())).numpy()
    else:
        shifted_stu_eu = shifted_tea_eu = shifted_stu_probs = None

    # --- 4a. Seen OOD (used in Phase 2 training) ---
    seen_names = {oid for oid, _, _ in SEEN_OOD}
    seen_available = {k: v for k, v in all_ood_data.items() if k in seen_names}
    if seen_available:
        print(f"\n{'='*60}")
        print(f"  4a. OOD Detection — SEEN OOD (used in Phase 2 training)")
        print(f"{'='*60}")
        for ood_id, (s_eu, t_eu, s_probs, _ttu, _tau, dname) in seen_available.items():
            print_ood_table("Clean CIFAR-10", dname,
                            clean_stu_eu, s_eu, clean_tea_eu, t_eu,
                            clean_stu_probs, s_probs)

    # --- 4b. Unseen OOD (held out — true generalization test) ---
    unseen_names = {oid for oid, _, _ in UNSEEN_OOD}
    unseen_available = {k: v for k, v in all_ood_data.items() if k in unseen_names}
    if unseen_available:
        print(f"\n{'='*60}")
        print(f"  4b. OOD Detection — UNSEEN OOD (not in training)")
        print(f"{'='*60}")
        for ood_id, (s_eu, t_eu, s_probs, _ttu, _tau, dname) in unseen_available.items():
            print_ood_table("Clean CIFAR-10", dname,
                            clean_stu_eu, s_eu, clean_tea_eu, t_eu,
                            clean_stu_probs, s_probs)

    # --- 4c. Shifted CIFAR-10 vs OOD ---
    if shifted_stu_eu is not None and all_ood_data:
        print(f"\n{'='*60}")
        print(f"  4c. OOD Detection — Shifted CIFAR-10 (ID) vs OOD")
        print(f"{'='*60}")
        for ood_id, (s_eu, t_eu, s_probs, _ttu, _tau, dname) in all_ood_data.items():
            print_ood_table("Shifted CIFAR-10", dname,
                            shifted_stu_eu, s_eu, shifted_tea_eu, t_eu,
                            shifted_stu_probs, s_probs)

    # --- 4d. Clean vs Shifted CIFAR-10 ---
    if shifted_stu_eu is not None:
        print(f"\n{'='*60}")
        print(f"  4d. Distribution Shift Detection")
        print(f"{'='*60}")
        print_ood_table("Clean CIFAR-10", "Shifted CIFAR-10",
                        clean_stu_eu, shifted_stu_eu,
                        clean_tea_eu, shifted_tea_eu,
                        clean_stu_probs, shifted_stu_probs)

    # ======================================================================
    # 5. Uncertainty Decomposition: TU / EU / AU
    # ======================================================================
    def entropy_np(p):
        return -(p * np.log(p + EPS)).sum(axis=-1)

    def decompose_teacher(probs_np, eu_np):
        tu = entropy_np(probs_np)
        au = np.maximum(tu - eu_np, 0.0)
        return tu, au, eu_np

    def decompose_student(probs_np, eu_np):
        tu = entropy_np(probs_np)
        au = np.maximum(tu - eu_np, 0.0)
        return tu, au, eu_np

    # Teacher decomposition on clean CIFAR-10 test (from cache)
    tea_tu = data["test_tu"] if "test_tu" in data else entropy_np(data["test_probs"])
    tea_au = data["test_au"] if "test_au" in data else np.maximum(tea_tu - data["test_eu"], 0.0)
    tea_eu_dec = data["test_eu"]
    # Student decomposition on clean CIFAR-10 test
    stu_tu, stu_au, stu_eu_dec = decompose_student(clean_stu_probs, clean_stu_eu)

    print(f"\n{'='*60}")
    print(f"  5. Uncertainty Decomposition (Clean CIFAR-10 test)")
    print(f"{'='*60}")
    print(f"  {'Metric':<10} {'Teacher mean':>14} {'Student mean':>14} {'Pearson':>10} {'Spearman':>10}")
    print(f"  {'-'*58}")
    for uname, t_vals, s_vals in [("TU", tea_tu, stu_tu), ("AU", tea_au, stu_au), ("EU", tea_eu_dec, stu_eu_dec)]:
        t_t = torch.from_numpy(t_vals).float()
        s_t = torch.from_numpy(s_vals).float()
        rp = pearson_corr(t_t, s_t)
        rs = spearman_corr(t_t, s_t)
        print(f"  {uname:<10} {t_vals.mean():>14.4f} {s_vals.mean():>14.4f} {rp:>10.4f} {rs:>10.4f}")

    # ======================================================================
    # 6. OOD Detection with Decomposed Uncertainties (AUROC)
    # ======================================================================
    if all_ood_data:
        print(f"\n{'='*60}")
        print(f"  6. OOD Detection: CIFAR-10 vs OOD — Decomposed Uncertainties")
        print(f"{'='*60}")
        print(f"  {'Dataset':<16} {'Type':<7} "
              f"{'Tea TU':>8} {'Tea EU':>8} {'Tea AU':>8} | "
              f"{'Stu TU':>8} {'Stu EU':>8} {'Stu AU':>8}")
        print(f"  {'-'*83}")

        id_tea_tu, id_tea_au, id_tea_eu = tea_tu, tea_au, tea_eu_dec
        id_stu_tu, id_stu_au, id_stu_eu = stu_tu, stu_au, stu_eu_dec

        for ood_id, (s_eu, t_eu, s_probs, t_tu_np, t_au_np, dname) in all_ood_data.items():
            ood_type = "seen" if ood_id in seen_names else "unseen"

            # Teacher decomposition for this OOD (from cache)
            a_tea_eu = auroc(id_tea_eu, t_eu)
            if t_tu_np is not None:
                a_tea_tu = auroc(id_tea_tu, t_tu_np)
                a_tea_au = auroc(id_tea_au, t_au_np)
                tea_tu_str = f"{a_tea_tu:>8.4f}"
                tea_au_str = f"{a_tea_au:>8.4f}"
            else:
                tea_tu_str = "     ---"
                tea_au_str = "     ---"

            # Student decomposition for this OOD
            ood_stu_tu = entropy_np(s_probs)
            ood_stu_eu = s_eu
            ood_stu_au = np.maximum(ood_stu_tu - ood_stu_eu, 0.0)

            a_stu_tu = auroc(id_stu_tu, ood_stu_tu)
            a_stu_eu = auroc(id_stu_eu, ood_stu_eu)
            a_stu_au = auroc(id_stu_au, ood_stu_au)

            print(f"  {dname:<16} {ood_type:<7} "
                  f"{tea_tu_str:>8} {a_tea_eu:>8.4f} {tea_au_str:>8} | "
                  f"{a_stu_tu:>8.4f} {a_stu_eu:>8.4f} {a_stu_au:>8.4f}")

        print(f"\n  Student TU = H[softmax(logits)], EU = EU head, AU = TU - EU")
        print(f"  Expectation: EU >> AU on OOD (epistemic dominates)")
        print(f"               AU ≈ stable on ID vs OOD (aleatoric is data-intrinsic)")

    # 7. Single-model baseline
    single_model = load_single_member(args.save_dir, device)
    if single_model is not None and all_ood_data:
        print(f"\n{'='*60}")
        print(f"  7. Baseline: Single Ensemble Member — OOD Detection")
        print(f"{'='*60}")

        @torch.no_grad()
        def predict_single(mdl, ldr):
            probs_list = []
            for batch in ldr:
                imgs = batch[0].to(device)
                logits = mdl(imgs)
                probs_list.append(F.softmax(logits, dim=-1).cpu().numpy())
            return np.concatenate(probs_list, axis=0)

        id_probs_single = predict_single(single_model, test_loader)
        id_ent_single = entropy_np(id_probs_single)
        id_mp_single = 1.0 - id_probs_single.max(axis=-1)
        single_acc = (id_probs_single.argmax(axis=-1) == data["test_labels"]).mean() * 100
        print(f"  Single member test accuracy: {single_acc:.2f}%")

        print(f"\n  {'Dataset':<16} {'Type':<7} {'Entropy':>10} {'1-MaxProb':>10}")
        print(f"  {'-'*48}")

        for ood_id, (s_eu, t_eu, s_probs, _ttu, _tau, dname) in all_ood_data.items():
            ood_type = "seen" if ood_id in seen_names else "unseen"
            try:
                ood_loader = get_ood_loader(ood_id, args.data_dir,
                                            args.batch_size, args.num_workers)
                ood_probs_single = predict_single(single_model, ood_loader)
            except Exception as e:
                print(f"  {dname:<16} {ood_type:<7} SKIPPED ({e})")
                continue
            min_n = min(len(ood_probs_single), len(t_eu))
            ood_probs_single = ood_probs_single[:min_n]
            ood_ent_single = entropy_np(ood_probs_single)
            ood_mp_single = 1.0 - ood_probs_single.max(axis=-1)
            a_ent = auroc(id_ent_single, ood_ent_single)
            a_mp = auroc(id_mp_single, ood_mp_single)
            print(f"  {dname:<16} {ood_type:<7} {a_ent:>10.4f} {a_mp:>10.4f}")

        if shifted_stu_probs is not None:
            id_loader_shift = get_test_loader(args.data_dir, args.batch_size, args.num_workers)
            clean_imgs_all = torch.cat([batch[0] for batch in id_loader_shift], 0)
            shifted_probs_list = []
            for ctype in CORRUPTION_TYPES:
                corrupted = apply_corruption(clean_imgs_all, ctype, seed=CORRUPTION_SEED)
                ds_c = TensorDataset(corrupted, torch.zeros(len(corrupted), dtype=torch.long))
                c_loader = DataLoader(ds_c, batch_size=args.batch_size, shuffle=False, num_workers=0)
                shifted_probs_list.append(predict_single(single_model, c_loader))
            shifted_probs_single = np.concatenate(shifted_probs_list, axis=0)
            shifted_ent_single = entropy_np(shifted_probs_single)
            shifted_mp_single = 1.0 - shifted_probs_single.max(axis=-1)
            a_ent = auroc(id_ent_single, shifted_ent_single)
            a_mp = auroc(id_mp_single, shifted_mp_single)
            print(f"  {'Shifted CIFAR-10':<16} {'shift':<7} {a_ent:>10.4f} {a_mp:>10.4f}")

    # ======================================================================
    # 8. Selective Prediction — AURC & Sparsification
    # ======================================================================
    print(f"\n{'='*60}")
    print(f"  8. Selective Prediction (AURC) — Clean CIFAR-10 test")
    print(f"{'='*60}")

    errors = (student_preds.numpy() != labels_np).astype(float)
    stu_ent_aurc = -(stu_probs_np * np.log(stu_probs_np + EPS)).sum(axis=-1)
    stu_mp_aurc = 1.0 - stu_probs_np.max(axis=-1)

    aurc_rows = [
        ("Teacher EU",      teacher_eu_t.numpy()),
        ("Student EU (ours)", student_eu.numpy()),
        ("Student entropy",  stu_ent_aurc),
        ("1 - MaxProb",      stu_mp_aurc),
        ("Oracle",           errors),            # errors as score: correct (0) first
    ]
    random_aurc = errors.mean()  # constant risk = overall error rate

    print(f"  {'Method':<30} {'AURC↓':>10} {'OracleGap↓':>12} {'@90%cov↑':>10} {'@80%cov↑':>10}")
    print(f"  {'-'*74}")
    for row_name, scores in aurc_rows:
        a, oa, gap, a90, a80 = compute_aurc(errors, scores)
        print(f"  {row_name:<30} {a:>10.6f} {gap:>12.6f} {a90:>10.4f} {a80:>10.4f}")
    print(f"  {'Random (baseline)':<30} {random_aurc:>10.6f} {'—':>12} {'—':>10} {'—':>10}")

    # ======================================================================
    # 9. Inference Throughput  (batch=256, 100 batches, after 10-batch warmup)
    # ======================================================================
    print(f"\n{'='*60}")
    print(f"  9. Inference Throughput  (device={device}, bs=256, 100 batches)")
    print(f"{'='*60}")

    img_shape = (3, 32, 32)
    model.eval()
    stu_tp = measure_throughput(lambda x: model(x), img_shape, device)

    # Load all available ensemble members
    all_members = []
    for midx in range(20):
        mpath = os.path.join(args.save_dir, f"member_{midx}.pt")
        if not os.path.exists(mpath):
            break
        mckpt = torch.load(mpath, map_location=device, weights_only=True)
        mcfg = mckpt.get("member_config", {})
        mem = cifar_resnet18(
            num_classes=10,
            dropout_rate=mcfg.get("dropout_rate", 0.0),
            head_init_scale=1.0,
        ).to(device)
        mem.load_state_dict(mckpt["model_state_dict"])
        mem.eval()
        all_members.append(mem)

    print(f"  {'Model':<38} {'Samples/sec':>14} {'Speedup vs ens':>16}")
    print(f"  {'-'*70}")

    if all_members:
        K = len(all_members)
        def _ens_fn(x):
            ps = [F.softmax(m(x), dim=-1) for m in all_members]
            return torch.stack(ps, 0).mean(0)
        ens_tp = measure_throughput(_ens_fn, img_shape, device)
        sgl_tp = measure_throughput(lambda x: all_members[0](x), img_shape, device)
        print(f"  {'Ensemble (K='+str(K)+', sequential)':<38} {ens_tp:>14,.0f} {'1.00x':>16}")
        print(f"  {'Single member':<38} {sgl_tp:>14,.0f} {sgl_tp/ens_tp:>15.2f}x")
        print(f"  {'Student (single pass)':<38} {stu_tp:>14,.0f} {stu_tp/ens_tp:>15.2f}x")
    else:
        print(f"  {'Student (single pass)':<38} {stu_tp:>14,.0f} {'(no members found)':>16}")


def load_single_member(save_dir, device, member_idx=0):
    """Load a single ensemble member as an OOD detection baseline."""
    path = os.path.join(save_dir, f"member_{member_idx}.pt")
    if not os.path.exists(path):
        print(f"  Single member checkpoint not found: {path}")
        return None
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt.get("member_config", {})
    model = cifar_resnet18(
        num_classes=10,
        dropout_rate=cfg.get("dropout_rate", 0.0),
        head_init_scale=1.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    acc = ckpt.get("test_acc", "?")
    print(f"  Loaded single member (member_{member_idx}, acc={acc}%)")
    return model


if __name__ == "__main__":
    main()
