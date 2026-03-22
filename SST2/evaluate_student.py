"""
Comprehensive evaluation of the distilled DistilBERT student on SST-2.

Reports:
    1. Accuracy (teacher ensemble vs student)
    2. Calibration (ECE-15, NLL, Brier)
    3. EU correlation (clean, perturbed, OOD)
    4. OOD detection AUROC (5 OOD datasets)
    5. Selective prediction (AURC)
    6. Inference throughput

Usage:
    python evaluate_student.py --save_dir ./checkpoints --gpu 0
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import create_student, create_teacher
from data import (load_sst2, SST2Dataset, GenericTextDataset, load_ood_datasets,
                  collate_fn, apply_char_perturbations, MAX_SEQ_LEN)


EPS = 1e-8


# ---------------------------------------------------------------------------
# Metrics
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


def compute_aurc(errors_np, scores_np):
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


def entropy_np(p):
    return -(p * np.log(p + EPS)).sum(axis=-1)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_student(model, loader, device):
    model.eval()
    all_probs, all_eu = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits, eu = model(input_ids, attention_mask)
        all_probs.append(F.softmax(logits, dim=-1).cpu())
        all_eu.append(eu.cpu())
    return torch.cat(all_probs), torch.cat(all_eu)


@torch.no_grad()
def predict_teacher_single(model, loader, device):
    model.eval()
    all_probs = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


@torch.no_grad()
def measure_throughput(model, tokenizer, device, batch_size=64, n_batches=100, n_warmup=10):
    """Measure sentences/sec throughput."""
    dummy_ids = torch.randint(0, 1000, (batch_size, MAX_SEQ_LEN), device=device)
    dummy_mask = torch.ones(batch_size, MAX_SEQ_LEN, dtype=torch.long, device=device)
    for _ in range(n_warmup):
        model(dummy_ids, dummy_mask) if not isinstance(model, list) else [m(dummy_ids, dummy_mask) for m in model]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_batches):
        if isinstance(model, list):
            probs_list = [F.softmax(m(dummy_ids, dummy_mask), dim=-1) for m in model]
            _ = torch.stack(probs_list, 0).mean(0)
        else:
            model(dummy_ids, dummy_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (batch_size * n_batches) / (time.time() - t0)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_student_model(save_dir, device):
    path = os.path.join(save_dir, "student.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = create_student(num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded student from {path}")
    print(f"  Dev acc={ckpt.get('dev_acc', '?')}%  "
          f"EU Pearson={ckpt.get('eu_pearson', '?')}  Spearman={ckpt.get('eu_spearman', '?')}")
    return model


def load_single_member(save_dir, device, member_idx=0):
    path = os.path.join(save_dir, f"member_{member_idx}.pt")
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mcfg = ckpt.get("member_config", {})
    model = create_teacher(
        num_classes=2, rank=mcfg.get("rank", 8), alpha=mcfg.get("alpha", 16.0),
        attention_dropout=mcfg.get("attention_dropout", 0.1), init_scale=1.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def load_ensemble_members(save_dir, device):
    members = []
    for idx in range(20):
        m = load_single_member(save_dir, device, idx)
        if m is None:
            break
        members.append(m)
    return members


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate distilled student on SST-2")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    model = load_student_model(args.save_dir, device)
    data = np.load(os.path.join(args.save_dir, "teacher_targets.npz"), allow_pickle=True)

    _, dev_ds, tokenizer = load_sst2("distilbert-base-uncased")
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    teacher_probs_t = torch.from_numpy(data["dev_probs"]).float()
    teacher_eu_t = torch.from_numpy(data["dev_eu"]).float()
    labels_np = data["dev_labels"]
    true_labels_t = torch.from_numpy(labels_np).long()

    student_probs, student_eu = predict_student(model, dev_loader, device)
    student_preds = student_probs.argmax(dim=1)
    teacher_preds = teacher_probs_t.argmax(dim=1)

    # ======================================================================
    # 1. Accuracy
    # ======================================================================
    student_acc = student_preds.eq(true_labels_t).float().mean().item() * 100
    teacher_acc = teacher_preds.eq(true_labels_t).float().mean().item() * 100

    print(f"\n{'='*60}")
    print(f"  1. Accuracy (SST-2 dev)")
    print(f"{'='*60}")
    print(f"  Teacher (ensemble):  {teacher_acc:.2f}%")
    print(f"  Student (distilled): {student_acc:.2f}%")

    # ======================================================================
    # 2. Calibration
    # ======================================================================
    stu_probs_np = student_probs.numpy()
    tea_probs_np = teacher_probs_t.numpy()

    stu_ece = compute_ece(stu_probs_np, labels_np)
    stu_nll = compute_nll(stu_probs_np, labels_np)
    stu_brier = compute_brier(stu_probs_np, labels_np)
    tea_ece = compute_ece(tea_probs_np, labels_np)
    tea_nll = compute_nll(tea_probs_np, labels_np)
    tea_brier = compute_brier(tea_probs_np, labels_np)

    print(f"\n{'='*60}")
    print(f"  2. Calibration (SST-2 dev)")
    print(f"{'='*60}")
    print(f"  {'Model':<30} {'ECE-15':>10} {'NLL':>10} {'Brier':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Teacher (ensemble)':<30} {tea_ece:>10.4f} {tea_nll:>10.4f} {tea_brier:>10.4f}")
    print(f"  {'Student (distilled)':<30} {stu_ece:>10.4f} {stu_nll:>10.4f} {stu_brier:>10.4f}")

    # ======================================================================
    # 3. EU Correlation
    # ======================================================================
    print(f"\n{'='*60}")
    print(f"  3. EU Correlation (student vs teacher)")
    print(f"{'='*60}")
    print(f"  {'Dataset':<30} {'Pearson':>10} {'Spearman':>10}")
    print(f"  {'-'*50}")

    r_p = pearson_corr(student_eu, teacher_eu_t)
    r_s = spearman_corr(student_eu, teacher_eu_t)
    print(f"  {'Clean SST-2 dev':<30} {r_p:>10.4f} {r_s:>10.4f}")

    # Perturbed dev
    if "char_perturbed_dev_eu" in data:
        from datasets import load_dataset
        raw_ds = load_dataset("glue", "sst2")
        dev_texts = raw_ds["validation"]["sentence"]
        char_dev_texts = [apply_char_perturbations(t, seed=4026 + i)
                          for i, t in enumerate(dev_texts)]
        char_dev_ds = SST2Dataset(char_dev_texts, raw_ds["validation"]["label"],
                                   tokenizer)
        char_loader = DataLoader(char_dev_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,
                                 collate_fn=collate_fn)
        char_probs, char_eu = predict_student(model, char_loader, device)
        char_tea_eu = torch.from_numpy(data["char_perturbed_dev_eu"]).float()
        min_n = min(len(char_eu), len(char_tea_eu))
        r_p = pearson_corr(char_eu[:min_n], char_tea_eu[:min_n])
        r_s = spearman_corr(char_eu[:min_n], char_tea_eu[:min_n])
        print(f"  {'Char-perturbed SST-2 dev':<30} {r_p:>10.4f} {r_s:>10.4f}")

    # ======================================================================
    # 4. OOD Detection
    # ======================================================================
    print(f"\n{'='*60}")
    print(f"  4. OOD Detection AUROC (SST-2 dev as ID)")
    print(f"{'='*60}")

    clean_stu_eu = student_eu.numpy()
    clean_tea_eu = teacher_eu_t.numpy()
    clean_stu_probs = stu_probs_np

    # Load OOD datasets
    print("\n  Loading OOD datasets...")
    ood_datasets = load_ood_datasets(tokenizer, max_samples=5000)

    # Pre-load ensemble once for all OOD datasets
    members = load_ensemble_members(args.save_dir, device)

    if ood_datasets:
        print(f"\n  {'OOD Dataset':<20} {'Teacher EU':>12} {'Student EU':>12} {'Stu Entropy':>12} {'1-MaxProb':>12}")
        print(f"  {'-'*70}")

        for ood_name, ood_ds in ood_datasets.items():
            ood_loader = DataLoader(ood_ds, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers,
                                    collate_fn=collate_fn)
            ood_probs, ood_eu = predict_student(model, ood_loader, device)
            ood_eu_np = ood_eu.numpy()
            ood_probs_np = ood_probs.numpy()

            # Teacher EU on OOD
            if members:
                all_member_probs = []
                with torch.no_grad():
                    for batch in ood_loader:
                        ids = batch["input_ids"].to(device)
                        mask = batch["attention_mask"].to(device)
                        batch_probs = []
                        for m in members:
                            logits = m(ids, mask)
                            batch_probs.append(F.softmax(logits, dim=-1).cpu())
                        all_member_probs.append(torch.stack(batch_probs, dim=1))
                all_mp = torch.cat(all_member_probs, dim=0)
                mean_p = all_mp.mean(dim=1)
                tu = entropy_np(mean_p.numpy())
                au = np.mean([entropy_np(all_mp[:, m].numpy()) for m in range(all_mp.shape[1])], axis=0)
                ood_tea_eu = tu - au
            else:
                ood_tea_eu = ood_eu_np  # fallback

            a_tea = auroc(clean_tea_eu, ood_tea_eu)
            a_stu = auroc(clean_stu_eu, ood_eu_np)

            id_ent = entropy_np(clean_stu_probs)
            ood_ent = entropy_np(ood_probs_np)
            a_ent = auroc(id_ent, ood_ent)

            id_mp = 1.0 - clean_stu_probs.max(axis=-1)
            ood_mp = 1.0 - ood_probs_np.max(axis=-1)
            a_mp = auroc(id_mp, ood_mp)

            print(f"  {ood_name:<20} {a_tea:>12.4f} {a_stu:>12.4f} {a_ent:>12.4f} {a_mp:>12.4f}")

    # ======================================================================
    # 5. Selective Prediction (AURC)
    # ======================================================================
    print(f"\n{'='*60}")
    print(f"  5. Selective Prediction (AURC) — SST-2 dev")
    print(f"{'='*60}")

    errors = (student_preds.numpy() != labels_np).astype(float)
    stu_ent = entropy_np(stu_probs_np)
    stu_mp = 1.0 - stu_probs_np.max(axis=-1)

    aurc_rows = [
        ("Teacher EU", clean_tea_eu),
        ("Student EU (ours)", clean_stu_eu),
        ("Student entropy", stu_ent),
        ("1 - MaxProb", stu_mp),
        ("Oracle", errors),
    ]

    print(f"  {'Method':<30} {'AURC':>10} {'OracleGap':>12} {'@90%cov':>10} {'@80%cov':>10}")
    print(f"  {'-'*74}")
    for row_name, scores in aurc_rows:
        a, oa, gap, a90, a80 = compute_aurc(errors, scores)
        print(f"  {row_name:<30} {a:>10.6f} {gap:>12.6f} {a90:>10.4f} {a80:>10.4f}")

    # ======================================================================
    # 6. Throughput
    # ======================================================================
    print(f"\n{'='*60}")
    print(f"  6. Inference Throughput (device={device})")
    print(f"{'='*60}")

    model.eval()
    stu_tp = measure_throughput(model, tokenizer, device)

    members = load_ensemble_members(args.save_dir, device)
    print(f"  {'Model':<38} {'Sent/sec':>14} {'Speedup':>16}")
    print(f"  {'-'*70}")
    if members:
        ens_tp = measure_throughput(members, tokenizer, device)
        sgl_tp = measure_throughput(members[0], tokenizer, device)
        print(f"  {'Ensemble (K='+str(len(members))+', sequential)':<38} {ens_tp:>14,.0f} {'1.00x':>16}")
        print(f"  {'Single member':<38} {sgl_tp:>14,.0f} {sgl_tp/ens_tp:>15.2f}x")
        print(f"  {'DistilBERT student (single pass)':<38} {stu_tp:>14,.0f} {stu_tp/ens_tp:>15.2f}x")
    else:
        print(f"  {'DistilBERT student (single pass)':<38} {stu_tp:>14,.0f}")

    # ======================================================================
    # 7. Summary JSON
    # ======================================================================
    summary = {
        "experiment": "E5_SST2",
        "teacher_acc": teacher_acc,
        "student_acc": student_acc,
        "student_ece": float(stu_ece),
        "student_nll": float(stu_nll),
        "student_brier": float(stu_brier),
        "eu_pearson_clean": float(pearson_corr(student_eu, teacher_eu_t)),
        "eu_spearman_clean": float(spearman_corr(student_eu, teacher_eu_t)),
    }
    summary_path = os.path.join(args.save_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
