"""
Run the trained ensemble over multiple data sources and cache teacher soft
labels and epistemic uncertainty (EU) per sample.

Data sources:
    1. Clean CIFAR-10 train + test
    2. Corrupted CIFAR-10 train (Gaussian noise, blur, low contrast)
    3. OOD: SVHN test, CIFAR-100 test

Corruptions are deterministic (fixed seed) so Phase 2 can regenerate the
same images without storing them.

Usage:
    python cache_ensemble_targets.py --save_dir ./checkpoints --data_dir ./data --gpu 0
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from models import cifar_resnet18


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
EPS = 1e-8

CORRUPTION_SEED = 2026
CORRUPTION_TYPES = ["gaussian_noise", "gaussian_blur", "low_contrast"]

# Fake OOD: mixup and masking (used when real OOD unavailable)
FAKE_OOD_SEED = 2027
MIXUP_LAMBDAS = [0.2, 0.4, 0.6, 0.8]  # interpolation rates
MASK_STYLES = ["random_block", "random_pixel", "center_crop_mask"]
MASK_RATES = [0.1, 0.3, 0.5]  # fraction of image masked


def load_ensemble(save_dir, device, num_classes=10):
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
        print(f"  Loaded {fname}  acc={ckpt.get('test_acc', 0):.2f}%")
    print(f"Ensemble size: {len(members)}")
    return members


def entropy(probs):
    return -(probs * torch.log(probs + EPS)).sum(dim=-1)


@torch.no_grad()
def compute_teacher_targets(members, loader, device):
    """Compute mean probs, TU, AU, EU for each sample."""
    all_member_probs = [[] for _ in members]
    all_targets = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        all_targets.append(targets)
        for m, model in enumerate(members):
            probs = F.softmax(model(inputs), dim=-1)
            all_member_probs[m].append(probs.cpu())
    all_probs = torch.stack([torch.cat(mp, dim=0) for mp in all_member_probs], dim=0)
    all_probs = all_probs.permute(1, 0, 2)  # (N, M, C)
    all_targets = torch.cat(all_targets, dim=0)

    mean_probs = all_probs.mean(dim=1)
    tu = entropy(mean_probs)
    au = entropy(all_probs).mean(dim=1)
    eu = tu - au
    return mean_probs.numpy(), tu.numpy(), au.numpy(), eu.numpy(), all_targets.numpy()


# ---------------------------------------------------------------------------
# Deterministic corruptions (must match Phase 2 regeneration)
# ---------------------------------------------------------------------------

def apply_corruption(images_tensor, corruption_type, seed=CORRUPTION_SEED):
    """Apply a deterministic corruption to a batch of normalized image tensors.

    Args:
        images_tensor: (N, 3, 32, 32) float tensor (already normalized)
        corruption_type: one of CORRUPTION_TYPES
        seed: random seed for reproducibility

    Returns:
        corrupted tensor of same shape, clamped to valid range
    """
    rng = torch.Generator().manual_seed(seed)

    if corruption_type == "gaussian_noise":
        noise = torch.randn(images_tensor.shape, generator=rng) * 0.15
        return images_tensor + noise

    elif corruption_type == "gaussian_blur":
        kernel_size = 5
        sigma = 2.0
        from torchvision.transforms.functional import gaussian_blur
        return torch.stack([gaussian_blur(img, kernel_size, sigma) for img in images_tensor])

    elif corruption_type == "low_contrast":
        factor = 0.3
        mean = images_tensor.mean(dim=(1, 2, 3), keepdim=True)
        return mean + factor * (images_tensor - mean)

    else:
        raise ValueError(f"Unknown corruption: {corruption_type}")


# ---------------------------------------------------------------------------
# Fake OOD: mixup and masking (deterministic, seed-based)
# ---------------------------------------------------------------------------

def apply_masking(images_tensor, style, rate, seed=FAKE_OOD_SEED):
    """Apply deterministic masking. Returns masked images (masked regions = 0)."""
    n, c, h, w = images_tensor.shape
    rng = torch.Generator(device=images_tensor.device).manual_seed(seed)
    out = images_tensor.clone()

    if style == "random_block":
        # Mask a random square region; rate = fraction of image area
        side = max(1, int((rate * h * w) ** 0.5))
        for i in range(n):
            si = int(torch.randint(0, h - side + 1 if h >= side else 1, (1,), generator=rng).item())
            sj = int(torch.randint(0, w - side + 1 if w >= side else 1, (1,), generator=rng).item())
            out[i, :, si:si + side, sj:sj + side] = 0

    elif style == "random_pixel":
        # Random pixel dropout; rate = fraction of pixels masked
        mask = torch.rand(n, 1, h, w, device=images_tensor.device, generator=rng) > rate
        out = out * mask.float()

    elif style == "center_crop_mask":
        # Mask center square; rate = fraction of area
        side = max(1, int((rate * h * w) ** 0.5))
        top, left = (h - side) // 2, (w - side) // 2
        out[:, :, top:top + side, left:left + side] = 0

    else:
        raise ValueError(f"Unknown mask style: {style}")
    return out


def generate_fake_ood(images_tensor, n_mixup, n_masked, seed=FAKE_OOD_SEED):
    """Generate mixup and masked samples. Returns (mixup_imgs, masked_imgs) for teacher EU."""
    n = len(images_tensor)
    rng = np.random.RandomState(seed)

    # Mixup: random pairs, cycling through lambdas (0.2, 0.4, 0.6, 0.8)
    mixup_tensors = []
    idx_perm = rng.permutation(n)
    for k in range(n_mixup):
        i = int(idx_perm[(2 * k) % n])
        j = int(idx_perm[(2 * k + 1) % n])
        lam = MIXUP_LAMBDAS[k % len(MIXUP_LAMBDAS)]
        x = (lam * images_tensor[i:i + 1] + (1 - lam) * images_tensor[j:j + 1]).clamp(0, 1)
        mixup_tensors.append(x)
    mixup_imgs = torch.cat(mixup_tensors, dim=0)

    # Masked: cycle through (style, rate) combinations
    combos = [(s, r) for s in MASK_STYLES for r in MASK_RATES]
    masked_tensors = []
    for k in range(n_masked):
        idx = rng.randint(0, n)
        style, rate = combos[k % len(combos)]
        masked = apply_masking(images_tensor[idx:idx + 1], style, rate, seed=seed + 10000 + k)
        masked_tensors.append(masked)
    masked_imgs = torch.cat(masked_tensors, dim=0)

    return mixup_imgs, masked_imgs


def get_clean_tensors(dataset, batch_size=256):
    """Load an entire dataset as a single (images, labels) tensor pair."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    imgs, labs = [], []
    for x, y in loader:
        imgs.append(x)
        labs.append(y)
    return torch.cat(imgs, 0), torch.cat(labs, 0)


def main():
    parser = argparse.ArgumentParser(description="Cache ensemble teacher targets (clean + corrupted + OOD)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--p2_data_mode", type=str, default="ood", choices=["ood", "fake_ood"],
                        help="Phase 2 data: 'ood'=real OOD (SVHN,CIFAR-100), 'fake_ood'=mixup+masked only")
    parser.add_argument("--fake_ood_mixup_frac", type=float, default=0.5,
                        help="Of the 25%% fake OOD, fraction from mixup (rest from masked). Default 0.5 = 12.5%% each")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    print("Loading ensemble...")
    members = load_ensemble(args.save_dir, device)

    clean_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    results = {}

    # === 1. Clean CIFAR-10 train + test ===
    print("\n--- Clean CIFAR-10 train ---")
    train_set = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=clean_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_probs, train_tu, train_au, train_eu, train_labels = compute_teacher_targets(members, train_loader, device)
    results["train_probs"] = train_probs
    results["train_tu"] = train_tu
    results["train_au"] = train_au
    results["train_eu"] = train_eu
    results["train_labels"] = train_labels
    print(f"  TU={train_tu.mean():.4f}  AU={train_au.mean():.4f}  EU={train_eu.mean():.4f}")

    print("\n--- Clean CIFAR-10 test ---")
    test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=clean_transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_probs, test_tu, test_au, test_eu, test_labels = compute_teacher_targets(members, test_loader, device)
    results["test_probs"] = test_probs
    results["test_tu"] = test_tu
    results["test_au"] = test_au
    results["test_eu"] = test_eu
    results["test_labels"] = test_labels
    print(f"  TU={test_tu.mean():.4f}  AU={test_au.mean():.4f}  EU={test_eu.mean():.4f}")

    # === 2. Corrupted CIFAR-10 train ===
    print("\n--- Corrupted CIFAR-10 train ---")
    train_imgs, _ = get_clean_tensors(train_set)

    for ctype in CORRUPTION_TYPES:
        print(f"  Corruption: {ctype} ...")
        corrupted = apply_corruption(train_imgs, ctype, seed=CORRUPTION_SEED)
        c_dataset = TensorDataset(corrupted, torch.zeros(len(corrupted), dtype=torch.long))
        c_loader = DataLoader(c_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        _probs, c_tu, c_au, c_eu, _ = compute_teacher_targets(members, c_loader, device)
        results[f"corrupt_{ctype}_eu"] = c_eu
        results[f"corrupt_{ctype}_tu"] = c_tu
        results[f"corrupt_{ctype}_au"] = c_au
        print(f"    TU={c_tu.mean():.4f}  AU={c_au.mean():.4f}  EU={c_eu.mean():.4f}")

    # Corrupted CIFAR-10 test (for evaluation)
    print("\n--- Corrupted CIFAR-10 test ---")
    test_imgs, _ = get_clean_tensors(test_set)
    for ctype in CORRUPTION_TYPES:
        print(f"  Corruption: {ctype} ...")
        corrupted = apply_corruption(test_imgs, ctype, seed=CORRUPTION_SEED)
        c_dataset = TensorDataset(corrupted, torch.zeros(len(corrupted), dtype=torch.long))
        c_loader = DataLoader(c_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        _probs, c_tu, c_au, c_eu, _ = compute_teacher_targets(members, c_loader, device)
        results[f"corrupt_{ctype}_test_eu"] = c_eu
        results[f"corrupt_{ctype}_test_tu"] = c_tu
        results[f"corrupt_{ctype}_test_au"] = c_au
        print(f"    TU={c_tu.mean():.4f}  AU={c_au.mean():.4f}  EU={c_eu.mean():.4f}")

    # === 3. OOD or Fake OOD (mixup + masked) ===
    if args.p2_data_mode == "ood":
        ood_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

        # SVHN
        print("\n--- SVHN test (OOD) ---")
        svhn_set = datasets.SVHN(root=os.path.join(args.data_dir, "svhn"), split="test",
                                 download=True, transform=ood_transform)
        svhn_loader = DataLoader(svhn_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        _probs, svhn_tu, svhn_au, svhn_eu, _ = compute_teacher_targets(members, svhn_loader, device)
        results["svhn_eu"] = svhn_eu
        results["svhn_tu"] = svhn_tu
        results["svhn_au"] = svhn_au
        print(f"  TU={svhn_tu.mean():.4f}  AU={svhn_au.mean():.4f}  EU={svhn_eu.mean():.4f}  n={len(svhn_eu)}")

        # CIFAR-100
        print("\n--- CIFAR-100 test (OOD) ---")
        c100_set = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=clean_transform)
        c100_loader = DataLoader(c100_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        _probs, c100_tu, c100_au, c100_eu, _ = compute_teacher_targets(members, c100_loader, device)
        results["cifar100_eu"] = c100_eu
        results["cifar100_tu"] = c100_tu
        results["cifar100_au"] = c100_au
        print(f"  TU={c100_tu.mean():.4f}  AU={c100_au.mean():.4f}  EU={c100_eu.mean():.4f}  n={len(c100_eu)}")

    else:
        # Fake OOD: mixup + masked (no external OOD data needed)
        # 25% of train size = fake OOD; split by mixup_frac
        n_fake = len(train_imgs) // 4
        n_mixup = int(n_fake * args.fake_ood_mixup_frac)
        n_masked = n_fake - n_mixup
        print(f"\n--- Fake OOD (mixup + masked, no external data) ---")
        print(f"  Target: 50% ID, 25% shifted, 25% fake OOD (mixup={n_mixup}, masked={n_masked})")

        mixup_imgs, masked_imgs = generate_fake_ood(train_imgs, n_mixup, n_masked, seed=FAKE_OOD_SEED)

        mixup_ds = TensorDataset(mixup_imgs, torch.zeros(len(mixup_imgs), dtype=torch.long))
        mixup_loader = DataLoader(mixup_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        _probs, mixup_tu, mixup_au, mixup_eu, _ = compute_teacher_targets(members, mixup_loader, device)
        results["fake_mixup_eu"] = mixup_eu
        results["fake_mixup_imgs"] = mixup_imgs.numpy()
        print(f"  Mixup (λ∈{{0.2,0.4,0.6,0.8}}): n={len(mixup_eu)}  EU mean={mixup_eu.mean():.4f}")

        masked_ds = TensorDataset(masked_imgs, torch.zeros(len(masked_imgs), dtype=torch.long))
        masked_loader = DataLoader(masked_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        _probs, masked_tu, masked_au, masked_eu, _ = compute_teacher_targets(members, masked_loader, device)
        results["fake_masked_eu"] = masked_eu
        results["fake_masked_imgs"] = masked_imgs.numpy()
        print(f"  Masked (styles={MASK_STYLES}, rates={MASK_RATES}): n={len(masked_eu)}  EU mean={masked_eu.mean():.4f}")

    # === 4. Unseen OOD datasets (not used in Phase 2 training) ===
    # Grayscale datasets need 3-channel conversion
    gray_ood_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    unseen_ood = {
        "mnist": lambda: datasets.MNIST(
            root=args.data_dir, train=False, download=True, transform=gray_ood_transform),
        "fashionmnist": lambda: datasets.FashionMNIST(
            root=args.data_dir, train=False, download=True, transform=gray_ood_transform),
        "stl10": lambda: datasets.STL10(
            root=args.data_dir, split="test", download=True, transform=ood_transform),
    }

    # DTD may fail to download on some systems; handle gracefully
    try:
        _ = datasets.DTD(root=args.data_dir, split="test", download=True, transform=ood_transform)
        unseen_ood["dtd"] = lambda: datasets.DTD(
            root=args.data_dir, split="test", download=True, transform=ood_transform)
    except Exception as e:
        print(f"\n--- DTD: skipping ({e}) ---")

    for name, ds_fn in unseen_ood.items():
        print(f"\n--- {name.upper()} (unseen OOD) ---")
        try:
            ds = ds_fn()
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            _probs, ood_tu, ood_au, ood_eu, _ = compute_teacher_targets(members, loader, device)
            results[f"{name}_eu"] = ood_eu
            results[f"{name}_tu"] = ood_tu
            results[f"{name}_au"] = ood_au
            print(f"  TU={ood_tu.mean():.4f}  AU={ood_au.mean():.4f}  EU={ood_eu.mean():.4f}  n={len(ood_eu)}")
        except Exception as e:
            print(f"  Failed: {e}")

    # === Save ===
    results["eu_max"] = np.array(float(train_eu.max()))
    results["corruption_seed"] = np.array(CORRUPTION_SEED)
    results["p2_data_mode"] = np.array([args.p2_data_mode], dtype=object)
    results["fake_ood_mixup_frac"] = np.array(args.fake_ood_mixup_frac)

    out_path = os.path.join(args.save_dir, "teacher_targets.npz")
    np.savez(out_path, **results)
    print(f"\nSaved all targets to {out_path}")
    print(f"Keys: {sorted(results.keys())}")


if __name__ == "__main__":
    main()
