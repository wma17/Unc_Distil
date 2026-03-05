"""
Two-phase ensemble distillation into a single student model.

Phase 1 — Classification KD (backbone + classification head):
    L₁ = (1-α) CE(y, softmax(z_S)) + α τ² KL(softmax(z_T/τ) || softmax(z_S/τ))

Phase 2 — EU regression (freeze backbone + fc, train only EU head):
    L₂ = log1p_MSE(EU_S, EU_T) + β · PairwiseRankingLoss(EU_S, EU_T)
    Training data: 50% clean CIFAR-10 + 25% corrupted CIFAR-10 + 25% OOD

Usage:
    python cache_ensemble_targets.py --save_dir ./checkpoints   # prerequisite
    python distill.py --save_dir ./checkpoints --gpu 0
    python distill.py --save_dir ./checkpoints --gpu 0 --phase2_only
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset
from torchvision import datasets, transforms

from models import cifar_resnet18_student
from cache_ensemble_targets import apply_corruption, CORRUPTION_TYPES, CORRUPTION_SEED


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
EPS = 1e-8


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class DistillDataset(Dataset):
    """CIFAR-10 paired with cached teacher soft labels and EU targets (Phase 1)."""

    def __init__(self, cifar_dataset, teacher_probs, teacher_eu):
        self.cifar = cifar_dataset
        self.teacher_probs = torch.from_numpy(teacher_probs).float()
        self.teacher_eu = torch.from_numpy(teacher_eu).float()

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, label = self.cifar[idx]
        return img, label, self.teacher_probs[idx], self.teacher_eu[idx]


class EUOnlyDataset(Dataset):
    """Image-EU pairs for Phase 2 (no labels/teacher probs needed)."""

    def __init__(self, images, eu_targets):
        self.images = images
        self.eu = torch.from_numpy(eu_targets).float() if isinstance(eu_targets, np.ndarray) else eu_targets.float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.eu[idx]


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def pearson_corr(a, b):
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def spearman_corr(a, b):
    def _rank(x):
        order = x.argsort()
        ranks = torch.empty_like(x)
        ranks[order] = torch.arange(len(x), dtype=x.dtype, device=x.device)
        return ranks
    return torch.corrcoef(torch.stack([_rank(a), _rank(b)]))[0, 1].item()


# ---------------------------------------------------------------------------
# Phase 1: Classification distillation
# ---------------------------------------------------------------------------

def phase1_loss(student_logits, labels, teacher_probs, alpha, tau):
    loss_ce = F.cross_entropy(student_logits, labels)
    teacher_logits = torch.log(teacher_probs + EPS)
    teacher_soft = F.softmax(teacher_logits / tau, dim=-1)
    student_log_soft = F.log_softmax(student_logits / tau, dim=-1)
    loss_kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (tau ** 2)
    loss = (1 - alpha) * loss_ce + alpha * loss_kl
    return loss, loss_ce.detach().item(), loss_kl.detach().item()


def train_phase1_epoch(model, loader, optimizer, alpha, tau, device):
    model.train()
    sum_ce, sum_kl, correct, total = 0, 0, 0, 0
    for imgs, labels, t_probs, _t_eu in loader:
        imgs, labels, t_probs = imgs.to(device), labels.to(device), t_probs.to(device)
        optimizer.zero_grad()
        logits, _eu = model(imgs)
        loss, ce, kl = phase1_loss(logits, labels, t_probs, alpha, tau)
        loss.backward()
        optimizer.step()
        bs = imgs.size(0)
        sum_ce += ce * bs
        sum_kl += kl * bs
        correct += logits.argmax(1).eq(labels).sum().item()
        total += bs
    return sum_ce / total, sum_kl / total, 100.0 * correct / total


@torch.no_grad()
def eval_phase1(model, loader, alpha, tau, device):
    model.eval()
    sum_ce, sum_kl, correct, total = 0, 0, 0, 0
    for imgs, labels, t_probs, _t_eu in loader:
        imgs, labels, t_probs = imgs.to(device), labels.to(device), t_probs.to(device)
        logits, _eu = model(imgs)
        _loss, ce, kl = phase1_loss(logits, labels, t_probs, alpha, tau)
        bs = imgs.size(0)
        sum_ce += ce * bs
        sum_kl += kl * bs
        correct += logits.argmax(1).eq(labels).sum().item()
        total += bs
    return sum_ce / total, sum_kl / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Phase 2: EU head training (backbone frozen, mixed data)
# ---------------------------------------------------------------------------

def log1p_mse_loss(pred, target):
    """MSE in log(1+x) space — compresses long tail, preserves ranking.

    Working in log-space converts the skewed EU distribution (95% near-zero,
    5% large) into a more balanced one, so the model doesn't ignore the
    low-EU regime where fine-grained ranking still matters.
    """
    pred_t = torch.log1p(pred)
    target_t = torch.log1p(target)
    return F.mse_loss(pred_t, target_t)


def pairwise_ranking_loss(pred, target, n_pairs=256, margin=0.05):
    """Differentiable pairwise ranking loss — directly optimizes ordering.

    Randomly samples pairs (i, j) where target_i > target_j and penalizes
    cases where pred_i < pred_j + margin.
    """
    bs = pred.size(0)
    if bs < 2:
        return pred.new_tensor(0.0)
    idx_i = torch.randint(0, bs, (n_pairs,), device=pred.device)
    idx_j = torch.randint(0, bs, (n_pairs,), device=pred.device)
    # Only keep pairs where targets differ
    t_i, t_j = target[idx_i], target[idx_j]
    mask = t_i > t_j + EPS
    if mask.sum() < 1:
        return pred.new_tensor(0.0)
    p_i, p_j = pred[idx_i][mask], pred[idx_j][mask]
    t_i, t_j = t_i[mask], t_j[mask]
    # MarginRankingLoss: max(0, -(p_i - p_j) + margin)
    return F.margin_ranking_loss(
        p_i, p_j,
        torch.ones(mask.sum(), device=pred.device),
        margin=margin,
    )


def set_phase2_mode(model):
    """Backbone in eval mode (frozen BN stats), EU head layers in train mode."""
    model.eval()
    model.eu_fc1.train()
    model.eu_fc2.train()


def train_phase2_epoch(model, loader, optimizer, rank_weight, device):
    set_phase2_mode(model)
    sum_mse, sum_rank, total = 0, 0, 0
    for imgs, t_eu in loader:
        imgs, t_eu = imgs.to(device), t_eu.to(device)
        optimizer.zero_grad()
        _logits, eu_pred = model(imgs)
        l_mse = log1p_mse_loss(eu_pred, t_eu)
        l_rank = pairwise_ranking_loss(eu_pred, t_eu)
        loss = l_mse + rank_weight * l_rank
        loss.backward()
        optimizer.step()
        bs = imgs.size(0)
        sum_mse += l_mse.detach().item() * bs
        sum_rank += l_rank.detach().item() * bs
        total += bs
    return sum_mse / total, sum_rank / total


@torch.no_grad()
def eval_phase2(model, loader, device):
    model.eval()
    sum_loss, total = 0, 0
    eu_preds, eu_targets = [], []
    for imgs, t_eu in loader:
        imgs, t_eu = imgs.to(device), t_eu.to(device)
        _logits, eu_pred = model(imgs)
        loss = log1p_mse_loss(eu_pred, t_eu)
        sum_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        eu_preds.append(eu_pred.cpu())
        eu_targets.append(t_eu.cpu())
    eu_preds = torch.cat(eu_preds)
    eu_targets = torch.cat(eu_targets)
    r_pearson = pearson_corr(eu_preds, eu_targets)
    r_spearman = spearman_corr(eu_preds, eu_targets)
    return sum_loss / total, r_pearson, r_spearman


# ---------------------------------------------------------------------------
# Phase 2 mixed dataset builder
# ---------------------------------------------------------------------------

def build_phase2_datasets(args, data):
    """Build mixed training dataset: 50% ID + 25% shifted ID + 25% (OOD or fake OOD).

    Returns (train_dataset, test_dataset) where each yields (image, eu_target).
    """
    clean_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ood_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # --- Clean CIFAR-10 train ---
    train_cifar = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=clean_transform)
    clean_imgs, _ = _load_tensor_dataset(train_cifar)
    n_clean = len(clean_imgs)

    # Target sizes for 50% ID, 25% shifted, 25% fake/real OOD
    n_id = n_clean // 2
    n_shifted = n_clean // 4
    n_ood = n_clean // 4
    rng = np.random.RandomState(CORRUPTION_SEED)

    # --- ID: subsample clean ---
    idx_id = rng.choice(n_clean, size=n_id, replace=False)
    clean_train_ds = EUOnlyDataset(clean_imgs[idx_id], data["train_eu"][idx_id])
    print(f"  ID (clean): {n_id} samples, EU mean={data['train_eu'][idx_id].mean():.4f}")

    # --- Shifted ID: corrupted CIFAR-10 (combine corruption types) ---
    corrupt_parts = []
    n_per_corruption = n_shifted // len(CORRUPTION_TYPES)
    for ctype in CORRUPTION_TYPES:
        key = f"corrupt_{ctype}_eu"
        if key not in data:
            print(f"  Warning: {key} not in cache, skipping")
            continue
        corrupted_imgs = apply_corruption(clean_imgs, ctype, seed=CORRUPTION_SEED)
        eu = data[key]
        idx = rng.choice(len(eu), size=min(n_per_corruption, len(eu)), replace=False)
        corrupt_parts.append(EUOnlyDataset(corrupted_imgs[idx], eu[idx]))
    n_corrupt_actual = sum(len(d) for d in corrupt_parts)
    if corrupt_parts:
        print(f"  Shifted (corrupted): {n_corrupt_actual} samples")

    # --- OOD or Fake OOD ---
    p2_mode = data.get("p2_data_mode", "ood")
    if hasattr(p2_mode, "flat"):
        p2_mode = str(p2_mode.flat[0]) if p2_mode.size else "ood"
    else:
        p2_mode = str(p2_mode)

    ood_datasets = []
    if p2_mode == "fake_ood" and "fake_mixup_eu" in data:
        # Fake OOD: mixup + masked (from cache)
        mixup_imgs = torch.from_numpy(data["fake_mixup_imgs"]).float()
        mixup_eu = data["fake_mixup_eu"]
        masked_imgs = torch.from_numpy(data["fake_masked_imgs"]).float()
        masked_eu = data["fake_masked_eu"]
        mf = data.get("fake_ood_mixup_frac", np.array(0.5))
        mixup_frac = float(mf.flat[0]) if hasattr(mf, "flat") and mf.size else 0.5
        n_mixup_target = int(n_ood * mixup_frac)
        n_masked_target = n_ood - n_mixup_target
        # Subsample to exact n_ood
        idx_m = rng.choice(len(mixup_eu), size=min(n_mixup_target, len(mixup_eu)), replace=False)
        idx_k = rng.choice(len(masked_eu), size=min(n_masked_target, len(masked_eu)), replace=False)
        ood_datasets.append(EUOnlyDataset(mixup_imgs[idx_m], mixup_eu[idx_m]))
        ood_datasets.append(EUOnlyDataset(masked_imgs[idx_k], masked_eu[idx_k]))
        print(f"  Fake OOD: mixup={len(idx_m)}, masked={len(idx_k)} (λ∈{{0.2,0.4,0.6,0.8}}, mask_rates={{0.1,0.3,0.5}})")

    else:
        # Real OOD: SVHN + CIFAR-100
        n_svhn = n_ood // 2
        n_c100 = n_ood - n_svhn
        if "svhn_eu" in data:
            svhn_set = datasets.SVHN(root=os.path.join(args.data_dir, "svhn"), split="test",
                                    download=True, transform=ood_transform)
            svhn_imgs, _ = _load_tensor_dataset(svhn_set)
            svhn_eu = data["svhn_eu"]
            idx = rng.choice(len(svhn_eu), size=min(n_svhn, len(svhn_eu)), replace=False)
            ood_datasets.append(EUOnlyDataset(svhn_imgs[idx], svhn_eu[idx]))
            print(f"  SVHN OOD: {len(idx)} samples")
        if "cifar100_eu" in data:
            c100_set = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=clean_transform)
            c100_imgs, _ = _load_tensor_dataset(c100_set)
            c100_eu = data["cifar100_eu"]
            idx = rng.choice(len(c100_eu), size=min(n_c100, len(c100_eu)), replace=False)
            ood_datasets.append(EUOnlyDataset(c100_imgs[idx], c100_eu[idx]))
            print(f"  CIFAR-100 OOD: {len(idx)} samples")

    all_train_parts = [clean_train_ds] + corrupt_parts + ood_datasets
    train_dataset = ConcatDataset(all_train_parts)
    total_n = sum(len(d) for d in all_train_parts)
    print(f"  Phase 2 total: {total_n} (50% ID={n_id}, 25% shifted={n_corrupt_actual}, 25% OOD={n_ood})")

    # --- Test: clean CIFAR-10 only (for consistent eval metric) ---
    test_cifar = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=clean_transform)
    test_imgs, _ = _load_tensor_dataset(test_cifar)
    test_dataset = EUOnlyDataset(test_imgs, data["test_eu"])

    return train_dataset, test_dataset


def _load_tensor_dataset(dataset, batch_size=512):
    """Load an entire torchvision dataset into a single tensor pair."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    imgs, labs = [], []
    for x, y in loader:
        imgs.append(x)
        labs.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    return torch.cat(imgs, 0), torch.cat(labs, 0)


# ---------------------------------------------------------------------------
# Phase 1 data loading (unchanged)
# ---------------------------------------------------------------------------

def build_phase1_loaders(args, train_probs, train_eu, test_probs, test_eu):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_cifar = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
    test_cifar = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(DistillDataset(train_cifar, train_probs, train_eu),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(DistillDataset(test_cifar, test_probs, test_eu),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Two-phase ensemble distillation")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--p1_epochs", type=int, default=200)
    parser.add_argument("--p1_lr", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=4.0)

    parser.add_argument("--p2_epochs", type=int, default=100)
    parser.add_argument("--p2_lr", type=float, default=0.001)
    parser.add_argument("--rank_weight", type=float, default=1.0,
                        help="Weight β for pairwise ranking loss in Phase 2")

    parser.add_argument("--phase2_only", action="store_true",
                        help="Skip Phase 1, load existing Phase 1 checkpoint and run Phase 2 only")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load cached targets ---
    targets_path = os.path.join(args.save_dir, "teacher_targets.npz")
    if not os.path.exists(targets_path):
        raise FileNotFoundError(
            f"{targets_path} not found. Run:\n  python cache_ensemble_targets.py --save_dir {args.save_dir}")
    data = np.load(targets_path)
    train_probs, train_eu = data["train_probs"], data["train_eu"]
    test_probs, test_eu = data["test_probs"], data["test_eu"]

    print(f"Clean EU: train mean={train_eu.mean():.4f} max={train_eu.max():.4f}  "
          f"test mean={test_eu.mean():.4f} max={test_eu.max():.4f}")

    model = cifar_resnet18_student(num_classes=10).to(device)
    p1_path = os.path.join(args.save_dir, "student_phase1.pt")
    final_path = os.path.join(args.save_dir, "student.pt")

    # ==================================================================
    # Phase 1: Classification KD
    # ==================================================================
    if not args.phase2_only:
        p1_train_loader, p1_test_loader = build_phase1_loaders(
            args, train_probs, train_eu, test_probs, test_eu)

        print(f"\n{'='*70}")
        print(f"  Phase 1: Classification distillation  (α={args.alpha}, τ={args.tau})")
        print(f"  {args.p1_epochs} epochs, lr={args.p1_lr}, warmup={args.warmup_epochs} epochs")
        print(f"{'='*70}\n")

        optimizer = optim.SGD(model.parameters(), lr=args.p1_lr, momentum=0.9, weight_decay=5e-4)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.p1_epochs)
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs])
        best_acc = 0.0

        for epoch in range(1, args.p1_epochs + 1):
            t0 = time.time()
            tr_ce, tr_kl, tr_acc = train_phase1_epoch(
                model, p1_train_loader, optimizer, args.alpha, args.tau, device)
            te_ce, te_kl, te_acc = eval_phase1(
                model, p1_test_loader, args.alpha, args.tau, device)
            scheduler.step()
            elapsed = time.time() - t0

            if epoch % 10 == 0 or epoch == 1:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  P1 Epoch {epoch:3d}/{args.p1_epochs} | "
                      f"Train {tr_acc:5.2f}% (ce={tr_ce:.4f} kl={tr_kl:.4f}) | "
                      f"Test {te_acc:5.2f}% (ce={te_ce:.4f} kl={te_kl:.4f}) | "
                      f"LR {lr_now:.6f} | {elapsed:.1f}s")

            if te_acc > best_acc:
                best_acc = te_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch, "test_acc": te_acc, "phase": 1,
                }, p1_path)

        print(f"\n  Phase 1 best accuracy: {best_acc:.2f}%  ->  {p1_path}")
    else:
        print(f"\nSkipping Phase 1, loading {p1_path}")

    # Load best Phase 1 model
    ckpt = torch.load(p1_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  Loaded Phase 1 checkpoint (acc={ckpt['test_acc']:.2f}%)")

    # ==================================================================
    # Phase 2: EU head training (backbone + fc frozen, mixed data)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  Phase 2: EU head regression  (backbone + fc frozen)")
    print(f"  Mixed data: clean + corrupted + OOD")
    print(f"  {args.p2_epochs} epochs, lr={args.p2_lr}")
    print(f"  Loss: log1p_MSE + {args.rank_weight} * PairwiseRankingLoss")
    print(f"{'='*70}\n")

    # Freeze everything except EU head
    for name, param in model.named_parameters():
        if name.startswith("eu_"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Re-initialize EU head
    model.reinit_eu_head()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total_params:,}  (eu_head only)")

    # Build mixed dataset
    print("\n  Building mixed Phase 2 dataset...")
    p2_train_ds, p2_test_ds = build_phase2_datasets(args, data)

    all_eu = torch.cat([p2_train_ds.datasets[i].eu for i in range(len(p2_train_ds.datasets))])
    print(f"\n  Mixed EU mean={all_eu.mean():.4f}  std={all_eu.std():.4f}  "
          f"max={all_eu.max():.4f}\n")

    p2_train_loader = DataLoader(p2_train_ds, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=True)
    p2_test_loader = DataLoader(p2_test_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

    optimizer = optim.Adam(model.eu_head_parameters, lr=args.p2_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.p2_epochs)
    best_spearman = -1.0
    best_pearson = 0.0

    for epoch in range(1, args.p2_epochs + 1):
        t0 = time.time()
        tr_mse, tr_rank = train_phase2_epoch(
            model, p2_train_loader, optimizer, args.rank_weight, device)
        te_loss, r_pear, r_spear = eval_phase2(model, p2_test_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  P2 Epoch {epoch:3d}/{args.p2_epochs} | "
                  f"Train mse={tr_mse:.6f} rank={tr_rank:.6f} | "
                  f"Test mse={te_loss:.6f} | "
                  f"Pearson={r_pear:.4f}  Spearman={r_spear:.4f} | "
                  f"LR {lr_now:.6f} | {elapsed:.1f}s")

        if r_spear > best_spearman:
            best_spearman = r_spear
            best_pearson = r_pear
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "test_acc": ckpt["test_acc"],
                "eu_pearson": r_pear, "eu_spearman": r_spear, "phase": 2,
            }, final_path)

    print(f"\n  Phase 2 best Spearman: {best_spearman:.4f}  (Pearson: {best_pearson:.4f})")
    print(f"  Final student saved to: {final_path}")
    print(f"\n  Next: python evaluate_student.py --save_dir {args.save_dir}")


if __name__ == "__main__":
    main()
