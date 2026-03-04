"""
Evaluate a trained deep ensemble on CIFAR-10 with full uncertainty decomposition.

Uncertainty Decomposition (Depeweg et al., 2018; Smith & Gal, 2018):
    - Total Uncertainty  (TU) = H[ E_θ[p(y|x,θ)] ]           (entropy of mean prediction)
    - Aleatoric Uncert.  (AU) = E_θ[ H[p(y|x,θ)] ]           (mean entropy of each member)
    - Epistemic Uncert.  (EU) = TU - AU = I[y; θ | x, D]      (mutual information)

Usage:
    python evaluate_uncertainty.py --save_dir ./checkpoints --data_dir ./data
    python evaluate_uncertainty.py --save_dir ./checkpoints --ood_dataset svhn   # OOD detection
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import cifar_resnet18


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
EPS = 1e-8


def load_ensemble(save_dir, device, num_classes=10):
    """Load all ensemble members from checkpoint directory.

    Dropout layers have no learnable parameters, so we construct every model
    with dropout_rate=0.0 and set eval mode — the state dicts are compatible
    regardless of what dropout rate was used during training.
    """
    members = []
    ckpt_files = sorted(f for f in os.listdir(save_dir) if f.startswith("member_") and f.endswith(".pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No member_*.pt checkpoints found in {save_dir}")

    for fname in ckpt_files:
        path = os.path.join(save_dir, fname)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = cifar_resnet18(num_classes=num_classes).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        members.append(model)

        acc = ckpt.get("test_acc", 0)
        cfg = ckpt.get("member_config", {})
        if cfg:
            augs = cfg.get("augmentations", cfg.get("augmentation", "?"))
            if isinstance(augs, list):
                augs = "+".join(augs) if augs else "base"
            print(f"  Loaded {fname}  acc={acc:.2f}%  "
                  f"augs=[{augs}]  smooth={cfg.get('label_smoothing')}  "
                  f"drop={cfg.get('dropout_rate')}  data={cfg.get('data_fraction', 1.0):.0%}")
        else:
            print(f"  Loaded {fname}  acc={acc:.2f}%  seed={ckpt.get('seed', '?')}")

    print(f"Ensemble size: {len(members)}")
    return members


def get_test_loader(data_dir, batch_size=256, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def get_ood_loader(ood_dataset, data_dir, batch_size=256, num_workers=4):
    """Load an OOD dataset for epistemic uncertainty evaluation."""
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    if ood_dataset == "svhn":
        ood_set = datasets.SVHN(root=os.path.join(data_dir, "svhn"), split="test",
                                download=True, transform=transform)
    elif ood_dataset == "cifar100":
        ood_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown OOD dataset: {ood_dataset}. Choose from: svhn, cifar100")

    return DataLoader(ood_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def entropy(probs):
    """Compute entropy H(p) = -sum(p * log(p)) along last dim."""
    return -(probs * torch.log(probs + EPS)).sum(dim=-1)


@torch.no_grad()
def ensemble_predict(members, loader, device):
    """
    Collect softmax predictions from all ensemble members.

    Returns:
        all_probs: (N, M, C) — softmax predictions per member
        all_targets: (N,) — ground truth labels (if available)
    """
    all_member_probs = [[] for _ in members]
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        all_targets.append(targets)

        for m, model in enumerate(members):
            logits = model(inputs)
            probs = F.softmax(logits, dim=-1)
            all_member_probs[m].append(probs.cpu())

    # (M, N, C) -> (N, M, C)
    all_probs = torch.stack([torch.cat(mp, dim=0) for mp in all_member_probs], dim=0)
    all_probs = all_probs.permute(1, 0, 2)
    all_targets = torch.cat(all_targets, dim=0)

    return all_probs, all_targets


def compute_uncertainty(all_probs):
    """
    Compute uncertainty decomposition from ensemble predictions.

    Args:
        all_probs: (N, M, C) tensor of softmax predictions

    Returns dict with per-sample uncertainties:
        - total_uncertainty:    H[ E_θ[p(y|x,θ)] ]
        - aleatoric_uncertainty: E_θ[ H[p(y|x,θ)] ]
        - epistemic_uncertainty: TU - AU = I[y; θ | x, D]
        - mean_probs: (N, C) mean predictive distribution
        - pred_labels: (N,) predicted class labels
    """
    # Mean predictive distribution: p̄(y|x) = (1/M) Σ_m p_m(y|x)
    mean_probs = all_probs.mean(dim=1)  # (N, C)

    # TU = H[ p̄(y|x) ]
    total_uncertainty = entropy(mean_probs)  # (N,)

    # AU = (1/M) Σ_m H[ p_m(y|x) ]
    member_entropies = entropy(all_probs)  # (N, M)
    aleatoric_uncertainty = member_entropies.mean(dim=1)  # (N,)

    # EU = TU - AU = I[y; θ | x, D]
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty  # (N,)

    pred_labels = mean_probs.argmax(dim=1)

    return {
        "total_uncertainty": total_uncertainty,
        "aleatoric_uncertainty": aleatoric_uncertainty,
        "epistemic_uncertainty": epistemic_uncertainty,
        "mean_probs": mean_probs,
        "pred_labels": pred_labels,
    }


def compute_metrics(uncertainties, targets):
    """Compute accuracy and various uncertainty statistics."""
    preds = uncertainties["pred_labels"]
    correct = preds.eq(targets).float()
    acc = correct.mean().item() * 100

    tu = uncertainties["total_uncertainty"]
    au = uncertainties["aleatoric_uncertainty"]
    eu = uncertainties["epistemic_uncertainty"]

    # Correct vs incorrect predictions
    correct_mask = correct.bool()
    wrong_mask = ~correct_mask

    metrics = {
        "accuracy": acc,
        "TU_mean": tu.mean().item(),
        "AU_mean": au.mean().item(),
        "EU_mean": eu.mean().item(),
    }

    if wrong_mask.any():
        metrics["TU_correct"] = tu[correct_mask].mean().item()
        metrics["TU_wrong"] = tu[wrong_mask].mean().item()
        metrics["EU_correct"] = eu[correct_mask].mean().item()
        metrics["EU_wrong"] = eu[wrong_mask].mean().item()
        metrics["AU_correct"] = au[correct_mask].mean().item()
        metrics["AU_wrong"] = au[wrong_mask].mean().item()

    return metrics


def print_metrics(metrics, title=""):
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    if "accuracy" in metrics:
        print(f"  Ensemble Accuracy:  {metrics['accuracy']:.2f}%")

    print(f"\n  {'Uncertainty':<25} {'Mean':>10}")
    print(f"  {'-'*35}")
    print(f"  {'Total (TU)':<25} {metrics['TU_mean']:>10.4f}")
    print(f"  {'Aleatoric (AU)':<25} {metrics['AU_mean']:>10.4f}")
    print(f"  {'Epistemic (EU)':<25} {metrics['EU_mean']:>10.4f}")

    if "TU_correct" in metrics:
        print(f"\n  {'Metric':<25} {'Correct':>10} {'Wrong':>10}")
        print(f"  {'-'*45}")
        print(f"  {'Total (TU)':<25} {metrics['TU_correct']:>10.4f} {metrics['TU_wrong']:>10.4f}")
        print(f"  {'Aleatoric (AU)':<25} {metrics['AU_correct']:>10.4f} {metrics['AU_wrong']:>10.4f}")
        print(f"  {'Epistemic (EU)':<25} {metrics['EU_correct']:>10.4f} {metrics['EU_wrong']:>10.4f}")


def auroc_from_scores(scores_in, scores_ood):
    """Compute AUROC for OOD detection using uncertainty as score."""
    labels = np.concatenate([np.zeros(len(scores_in)), np.ones(len(scores_ood))])
    scores = np.concatenate([scores_in, scores_ood])

    # Sort by ascending score
    order = np.argsort(scores)
    labels = labels[order]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr_sum = 0.0
    tp = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            tp += 1
        else:
            tpr_sum += tp / n_pos
    return tpr_sum / n_neg


def main():
    parser = argparse.ArgumentParser(description="Evaluate deep ensemble uncertainty on CIFAR-10")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ood_dataset", type=str, default=None,
                        choices=["svhn", "cifar100"],
                        help="OOD dataset for epistemic uncertainty evaluation")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save per-sample predictions and uncertainties to .npz")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading ensemble...")
    members = load_ensemble(args.save_dir, device)

    # --- In-distribution evaluation ---
    print("\nRunning in-distribution (CIFAR-10 test) evaluation...")
    test_loader = get_test_loader(args.data_dir, args.batch_size, args.num_workers)
    all_probs, all_targets = ensemble_predict(members, test_loader, device)
    uncertainties = compute_uncertainty(all_probs)
    metrics = compute_metrics(uncertainties, all_targets)
    print_metrics(metrics, title="CIFAR-10 Test Set")

    if args.save_predictions:
        out_path = os.path.join(args.save_dir, "cifar10_test_predictions.npz")
        np.savez(
            out_path,
            mean_probs=uncertainties["mean_probs"].numpy(),
            pred_labels=uncertainties["pred_labels"].numpy(),
            targets=all_targets.numpy(),
            total_uncertainty=uncertainties["total_uncertainty"].numpy(),
            aleatoric_uncertainty=uncertainties["aleatoric_uncertainty"].numpy(),
            epistemic_uncertainty=uncertainties["epistemic_uncertainty"].numpy(),
        )
        print(f"\n  Predictions saved to {out_path}")

    # --- OOD evaluation ---
    if args.ood_dataset:
        print(f"\nRunning OOD evaluation ({args.ood_dataset})...")
        ood_loader = get_ood_loader(args.ood_dataset, args.data_dir, args.batch_size, args.num_workers)
        ood_probs, ood_targets = ensemble_predict(members, ood_loader, device)
        ood_uncertainties = compute_uncertainty(ood_probs)

        tu_in = uncertainties["total_uncertainty"].numpy()
        tu_ood = ood_uncertainties["total_uncertainty"].numpy()
        eu_in = uncertainties["epistemic_uncertainty"].numpy()
        eu_ood = ood_uncertainties["epistemic_uncertainty"].numpy()

        auroc_tu = auroc_from_scores(tu_in, tu_ood)
        auroc_eu = auroc_from_scores(eu_in, eu_ood)

        print(f"\n{'='*60}")
        print(f"  OOD Detection: CIFAR-10 (in) vs {args.ood_dataset.upper()} (out)")
        print(f"{'='*60}")
        print(f"  AUROC (Total Uncertainty):     {auroc_tu:.4f}")
        print(f"  AUROC (Epistemic Uncertainty): {auroc_eu:.4f}")
        print(f"\n  OOD EU mean: {eu_ood.mean():.4f}  vs  ID EU mean: {eu_in.mean():.4f}")
        print(f"  OOD TU mean: {tu_ood.mean():.4f}  vs  ID TU mean: {tu_in.mean():.4f}")


if __name__ == "__main__":
    main()
