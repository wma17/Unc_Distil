"""
Two-phase ensemble distillation for MNIST into a single student model.

Phase 1 — Classification KD (backbone + classification head):
    L₁ = (1-α) CE(y, softmax(z_S)) + α τ² KL(softmax(z_T/τ) || softmax(z_S/τ))

Phase 2 — EU regression (freeze backbone + fc, train only EU head):
    L₂ = MSE(EU_S, EU_T) + β · PairwiseRankingLoss(EU_S, EU_T)
    Training data: 50% clean MNIST + 25% corrupted MNIST + 25% OOD

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

from models import mnist_convnet_student
from cache_ensemble_targets import apply_corruption, CORRUPTION_TYPES, CORRUPTION_SEED


MNIST_MEAN = (0.1307, 0.1307, 0.1307)
MNIST_STD = (0.3081, 0.3081, 0.3081)
EPS = 1e-8


class DistillDataset(Dataset):
    def __init__(self, mnist_dataset, teacher_probs, teacher_eu):
        self.mnist = mnist_dataset
        self.teacher_probs = torch.from_numpy(teacher_probs).float()
        self.teacher_eu = torch.from_numpy(teacher_eu).float()

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        return img, label, self.teacher_probs[idx], self.teacher_eu[idx]


class EUOnlyDataset(Dataset):
    def __init__(self, images, eu_targets):
        self.images = images
        self.eu = torch.from_numpy(eu_targets).float() if isinstance(eu_targets, np.ndarray) else eu_targets.float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.eu[idx]


def pearson_corr(a, b):
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def spearman_corr(a, b):
    def _rank(x):
        order = x.argsort()
        ranks = torch.empty_like(x)
        ranks[order] = torch.arange(len(x), dtype=x.dtype, device=x.device)
        return ranks
    return torch.corrcoef(torch.stack([_rank(a), _rank(b)]))[0, 1].item()


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


def eu_mse_loss(pred, target):
    return F.mse_loss(pred, target)


def pairwise_ranking_loss(pred, target, n_pairs=256, margin=0.05):
    bs = pred.size(0)
    if bs < 2:
        return pred.new_tensor(0.0)
    idx_i = torch.randint(0, bs, (n_pairs,), device=pred.device)
    idx_j = torch.randint(0, bs, (n_pairs,), device=pred.device)
    t_i, t_j = target[idx_i], target[idx_j]
    mask = t_i > t_j + EPS
    if mask.sum() < 1:
        return pred.new_tensor(0.0)
    p_i, p_j = pred[idx_i][mask], pred[idx_j][mask]
    return F.margin_ranking_loss(
        p_i, p_j,
        torch.ones(mask.sum(), device=pred.device),
        margin=margin,
    )


def set_phase2_mode(model):
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
        l_mse = eu_mse_loss(eu_pred, t_eu)
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
        loss = eu_mse_loss(eu_pred, t_eu)
        sum_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        eu_preds.append(eu_pred.cpu())
        eu_targets.append(t_eu.cpu())
    eu_preds = torch.cat(eu_preds)
    eu_targets = torch.cat(eu_targets)
    r_pearson = pearson_corr(eu_preds, eu_targets)
    r_spearman = spearman_corr(eu_preds, eu_targets)
    return sum_loss / total, r_pearson, r_spearman


def build_phase2_datasets(args, data):
    clean_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    ood_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])

    train_mnist = datasets.MNIST(root=args.data_dir, train=True, download=False, transform=clean_transform)
    clean_imgs, _ = _load_tensor_dataset(train_mnist)
    rng = np.random.RandomState(CORRUPTION_SEED)

    n_clean = len(clean_imgs)
    n_id = n_clean
    n_shifted = n_clean // 2
    n_ood = n_clean // 2

    idx_id = rng.choice(n_clean, size=n_id, replace=False)
    clean_train_ds = EUOnlyDataset(clean_imgs[idx_id], data["train_eu"][idx_id])
    print(f"  ID (clean): {len(idx_id)} samples, EU mean={data['train_eu'][idx_id].mean():.4f}")

    # Corrupted MNIST train
    corrupt_datasets = []
    n_per_corruption = max(1, n_shifted // max(len(CORRUPTION_TYPES), 1))
    for ctype in CORRUPTION_TYPES:
        key = f"corrupt_{ctype}_eu"
        if key not in data:
            continue
        corrupted_imgs = apply_corruption(clean_imgs, ctype, seed=CORRUPTION_SEED)
        eu = data[key]
        idx = rng.choice(len(eu), size=min(n_per_corruption, len(eu)), replace=False)
        corrupt_datasets.append(EUOnlyDataset(corrupted_imgs[idx], eu[idx]))
        print(f"  Corrupted {ctype}: {len(idx)} samples, EU mean={eu[idx].mean():.4f}")

    ood_datasets = []
    p2_mode = data.get("p2_data_mode", np.array("fake_ood"))
    if hasattr(p2_mode, "flat"):
        p2_mode = str(p2_mode.flat[0]) if p2_mode.size else "fake_ood"
    else:
        p2_mode = str(p2_mode)

    if p2_mode == "fake_ood" and "fake_mixup_eu" in data and "fake_masked_eu" in data:
        mixup_imgs = torch.from_numpy(data["fake_mixup_imgs"]).float()
        mixup_eu = data["fake_mixup_eu"]
        masked_imgs = torch.from_numpy(data["fake_masked_imgs"]).float()
        masked_eu = data["fake_masked_eu"]
        mf = data.get("fake_ood_mixup_frac", np.array(0.5))
        mixup_frac = float(mf.flat[0]) if hasattr(mf, "flat") and mf.size else 0.5

        n_mixup_target = min(int(n_ood * mixup_frac), len(mixup_eu))
        n_masked_target = min(n_ood - n_mixup_target, len(masked_eu))

        idx_m = rng.choice(len(mixup_eu), size=n_mixup_target, replace=False) if n_mixup_target > 0 else np.empty(0, dtype=np.int64)
        idx_k = rng.choice(len(masked_eu), size=n_masked_target, replace=False) if n_masked_target > 0 else np.empty(0, dtype=np.int64)
        ood_datasets.append(EUOnlyDataset(mixup_imgs[idx_m], mixup_eu[idx_m]))
        ood_datasets.append(EUOnlyDataset(masked_imgs[idx_k], masked_eu[idx_k]))
        print(f"  Fake OOD: mixup={len(idx_m)}, masked={len(idx_k)}")
    else:
        if "fashionmnist_eu" in data:
            fmnist_set = datasets.FashionMNIST(root=args.data_dir, train=False,
                                               download=False, transform=ood_transform)
            fmnist_imgs, _ = _load_tensor_dataset(fmnist_set)
            fmnist_eu = data["fashionmnist_eu"]
            n_f = min(n_ood // 2, len(fmnist_eu))
            idx = rng.choice(len(fmnist_eu), size=n_f, replace=False)
            ood_datasets.append(EUOnlyDataset(fmnist_imgs[idx], fmnist_eu[idx]))
            print(f"  FashionMNIST OOD: {n_f} samples, EU mean={fmnist_eu[idx].mean():.4f}")

        if "omniglot_eu" in data:
            try:
                omniglot_set = datasets.Omniglot(root=args.data_dir, background=False,
                                                 download=False, transform=ood_transform)
                omni_imgs, _ = _load_tensor_dataset(omniglot_set)
                omni_eu = data["omniglot_eu"]
                n_k = min(n_ood - sum(len(d) for d in ood_datasets), len(omni_eu))
                idx = rng.choice(len(omni_eu), size=n_k, replace=False)
                ood_datasets.append(EUOnlyDataset(omni_imgs[idx], omni_eu[idx]))
                print(f"  Omniglot OOD: {n_k} samples, EU mean={omni_eu[idx].mean():.4f}")
            except Exception as e:
                print(f"  Omniglot skipped: {e}")

    all_train_parts = [clean_train_ds] + corrupt_datasets + ood_datasets
    train_dataset = ConcatDataset(all_train_parts)
    total_n = sum(len(d) for d in all_train_parts)
    print(f"  Phase 2 total training samples: {total_n} "
          f"(clean={len(clean_train_ds)}, corrupted={sum(len(d) for d in corrupt_datasets)}, "
          f"ood={sum(len(d) for d in ood_datasets)})")

    test_mnist = datasets.MNIST(root=args.data_dir, train=False, download=False, transform=clean_transform)
    test_imgs, _ = _load_tensor_dataset(test_mnist)
    test_dataset = EUOnlyDataset(test_imgs, data["test_eu"])

    return train_dataset, test_dataset


def _load_tensor_dataset(dataset, batch_size=512):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    imgs, labs = [], []
    for x, y in loader:
        imgs.append(x)
        labs.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    return torch.cat(imgs, 0), torch.cat(labs, 0)


def build_phase1_loaders(args, train_probs, train_eu, test_probs, test_eu):
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    train_mnist = datasets.MNIST(root=args.data_dir, train=True, download=False, transform=train_transform)
    test_mnist = datasets.MNIST(root=args.data_dir, train=False, download=False, transform=test_transform)
    train_loader = DataLoader(DistillDataset(train_mnist, train_probs, train_eu),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(DistillDataset(test_mnist, test_probs, test_eu),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Two-phase MNIST ensemble distillation")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--p1_epochs", type=int, default=50)
    parser.add_argument("--p1_lr", type=float, default=0.001)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=4.0)

    parser.add_argument("--p2_epochs", type=int, default=100)
    parser.add_argument("--p2_lr", type=float, default=0.005)
    parser.add_argument("--rank_weight", type=float, default=1.0)

    parser.add_argument("--phase2_only", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    targets_path = os.path.join(args.save_dir, "teacher_targets.npz")
    if not os.path.exists(targets_path):
        raise FileNotFoundError(
            f"{targets_path} not found. Run:\n  python cache_ensemble_targets.py --save_dir {args.save_dir}")
    data = np.load(targets_path)
    train_probs, train_eu = data["train_probs"], data["train_eu"]
    test_probs, test_eu = data["test_probs"], data["test_eu"]

    print(f"Clean EU: train mean={train_eu.mean():.4f} max={train_eu.max():.4f}  "
          f"test mean={test_eu.mean():.4f} max={test_eu.max():.4f}")

    model = mnist_convnet_student(num_classes=10).to(device)
    p1_path = os.path.join(args.save_dir, "student_phase1.pt")
    final_path = os.path.join(args.save_dir, "student.pt")

    # ==================================================================
    # Phase 1
    # ==================================================================
    if not args.phase2_only:
        p1_train_loader, p1_test_loader = build_phase1_loaders(
            args, train_probs, train_eu, test_probs, test_eu)

        print(f"\n{'='*70}")
        print(f"  Phase 1: Classification distillation  (α={args.alpha}, τ={args.tau})")
        print(f"  {args.p1_epochs} epochs, lr={args.p1_lr}, warmup={args.warmup_epochs} epochs")
        print(f"{'='*70}\n")

        optimizer = optim.Adam(model.parameters(), lr=args.p1_lr, weight_decay=1e-4)
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

            if epoch % 5 == 0 or epoch == 1:
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

    ckpt = torch.load(p1_path, map_location=device, weights_only=True)
    state = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("eu_")}
    model.load_state_dict(state, strict=False)
    print(f"  Loaded Phase 1 checkpoint (acc={ckpt['test_acc']:.2f}%), EU head excluded")

    # ==================================================================
    # Phase 2
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  Phase 2: EU head regression  (backbone + fc frozen)")
    print(f"  {args.p2_epochs} epochs, lr={args.p2_lr}")
    print(f"  Loss: MSE + {args.rank_weight} * PairwiseRankingLoss")
    print(f"{'='*70}\n")

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("eu_")

    model.reinit_eu_head()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total_params:,}")

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

        if epoch % 5 == 0 or epoch == 1:
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
