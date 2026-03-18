"""
Run the trained MNIST ensemble over multiple data sources and cache teacher
soft labels, TU, AU, and EU per sample.

The model uses 3-channel input: MNIST is replicated 1→3ch, so color-based
corruptions (colored backgrounds, tinted digits) are meaningful. RGB OOD
datasets (CIFAR-10, SVHN) are used natively without grayscale conversion.

Data sources:
    1. Clean MNIST train + test (3ch)
    2. Corrupted MNIST train/test (noise, blur, contrast, inversion,
       colored background, colored digits, salt-and-pepper, pixelate)
    3. Seen OOD: FashionMNIST test, Omniglot evaluation
    4. Unseen OOD: EMNIST-Letters, CIFAR-10 (RGB), SVHN (RGB)

Usage:
    python cache_ensemble_targets.py --save_dir ./checkpoints --data_dir ../data --gpu 0
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from models import mnist_convnet


MNIST_MEAN = (0.1307, 0.1307, 0.1307)
MNIST_STD = (0.3081, 0.3081, 0.3081)
EPS = 1e-8

CORRUPTION_SEED = 2026
CORRUPTION_TYPES = [
    "gaussian_noise", "gaussian_blur", "low_contrast",
    "inversion", "colored_background", "colored_digits",
    "salt_pepper", "pixelate",
]

FAKE_OOD_SEED = 2027
MIXUP_LAMBDAS = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]
MASK_STYLES = ["random_block", "random_pixel", "center_crop_mask", "multi_block", "border_mask"]
MASK_RATES = [0.3, 0.5, 0.7, 0.8]


def load_ensemble(save_dir, device, num_classes=10):
    members = []
    ckpt_files = sorted(f for f in os.listdir(save_dir) if f.startswith("member_") and f.endswith(".pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No member_*.pt checkpoints found in {save_dir}")
    for fname in ckpt_files:
        path = os.path.join(save_dir, fname)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = mnist_convnet(num_classes=num_classes).to(device)
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


def _denorm(x):
    """Undo MNIST normalization: x_raw = x * std + mean (per channel)."""
    mean = torch.tensor(MNIST_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(MNIST_STD).view(1, 3, 1, 1)
    return x * std + mean


def _renorm(x):
    """Re-apply MNIST normalization after modification in raw space."""
    mean = torch.tensor(MNIST_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(MNIST_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def apply_corruption(images_tensor, corruption_type, seed=CORRUPTION_SEED):
    """Apply a deterministic corruption to (N, 3, 28, 28) normalized tensors.

    Since the model uses 3-channel input, we can apply color corruptions
    that create inter-channel differences the model has never seen in
    clean MNIST (where all 3 channels are identical).
    """
    rng = torch.Generator().manual_seed(seed)

    if corruption_type == "gaussian_noise":
        noise = torch.randn(images_tensor.shape, generator=rng) * 0.3
        return images_tensor + noise

    elif corruption_type == "gaussian_blur":
        from torchvision.transforms.functional import gaussian_blur
        return torch.stack([gaussian_blur(img, 5, 2.0) for img in images_tensor])

    elif corruption_type == "low_contrast":
        factor = 0.3
        mean = images_tensor.mean(dim=(1, 2, 3), keepdim=True)
        return mean + factor * (images_tensor - mean)

    elif corruption_type == "inversion":
        raw = _denorm(images_tensor)
        return _renorm(1.0 - raw.clamp(0, 1))

    elif corruption_type == "colored_background":
        raw = _denorm(images_tensor).clamp(0, 1)
        digit_mask = (raw.mean(dim=1, keepdim=True) > 0.15).float()
        n = raw.size(0)
        bg_colors = torch.rand(n, 3, 1, 1, generator=rng)
        colored = raw * digit_mask + bg_colors * (1 - digit_mask)
        return _renorm(colored)

    elif corruption_type == "colored_digits":
        raw = _denorm(images_tensor).clamp(0, 1)
        digit_mask = (raw.mean(dim=1, keepdim=True) > 0.15).float()
        n = raw.size(0)
        fg_colors = torch.rand(n, 3, 1, 1, generator=rng)
        colored = fg_colors * digit_mask + raw * (1 - digit_mask)
        return _renorm(colored)

    elif corruption_type == "salt_pepper":
        out = images_tensor.clone()
        mask = torch.rand(images_tensor.shape[0], 1, *images_tensor.shape[2:], generator=rng)
        mask = mask.expand_as(images_tensor)
        salt_val = _renorm(torch.ones(1, 3, 1, 1)).expand_as(images_tensor)
        pepper_val = _renorm(torch.zeros(1, 3, 1, 1)).expand_as(images_tensor)
        out = torch.where(mask < 0.03, salt_val, out)
        out = torch.where(mask > 0.97, pepper_val, out)
        return out

    elif corruption_type == "pixelate":
        raw = _denorm(images_tensor).clamp(0, 1)
        small = F.interpolate(raw, size=7, mode="bilinear", align_corners=False)
        pixelated = F.interpolate(small, size=28, mode="nearest")
        return _renorm(pixelated)

    else:
        raise ValueError(f"Unknown corruption: {corruption_type}")


def apply_masking(images_tensor, style, rate, seed=FAKE_OOD_SEED):
    """Apply deterministic masking in raw pixel space and re-normalize."""
    squeeze = images_tensor.dim() == 3
    if squeeze:
        images_tensor = images_tensor.unsqueeze(0)

    raw = _denorm(images_tensor).clamp(0, 1)
    out = raw.clone()
    n, _c, h, w = out.shape
    rng = torch.Generator().manual_seed(seed)

    if style == "random_block":
        side = max(1, int((rate * h * w) ** 0.5))
        for i in range(n):
            top = int(torch.randint(0, max(1, h - side + 1), (1,), generator=rng).item())
            left = int(torch.randint(0, max(1, w - side + 1), (1,), generator=rng).item())
            out[i, :, top:top + side, left:left + side] = 0

    elif style == "random_pixel":
        keep = torch.rand(n, 1, h, w, generator=rng) > rate
        out = out * keep.float()

    elif style == "center_crop_mask":
        side = max(1, int((rate * h * w) ** 0.5))
        top = (h - side) // 2
        left = (w - side) // 2
        out[:, :, top:top + side, left:left + side] = 0

    elif style == "multi_block":
        n_blocks = max(2, int(rate * 8))
        block_area = rate * h * w / n_blocks
        side = max(2, int(block_area ** 0.5))
        for i in range(n):
            for _ in range(n_blocks):
                top = int(torch.randint(0, max(1, h - side + 1), (1,), generator=rng).item())
                left = int(torch.randint(0, max(1, w - side + 1), (1,), generator=rng).item())
                out[i, :, top:top + side, left:left + side] = 0

    elif style == "border_mask":
        border = max(1, int(rate * min(h, w) / 2))
        out[:, :, :border, :] = 0
        out[:, :, -border:, :] = 0
        out[:, :, :, :border] = 0
        out[:, :, :, -border:] = 0

    else:
        raise ValueError(f"Unknown mask style: {style}")

    masked = _renorm(out)
    return masked[0] if squeeze else masked


def generate_fake_ood(images_tensor, n_mixup, n_masked, seed=FAKE_OOD_SEED):
    """Generate deterministic mixup and masked fake OOD samples."""
    n = len(images_tensor)
    rng = np.random.RandomState(seed)
    raw = _denorm(images_tensor).clamp(0, 1)

    if n_mixup > 0:
        mixup_tensors = []
        idx_perm = rng.permutation(n)
        for k in range(n_mixup):
            i = int(idx_perm[(2 * k) % n])
            j = int(idx_perm[(2 * k + 1) % n])
            lam = MIXUP_LAMBDAS[k % len(MIXUP_LAMBDAS)]
            mixed = (lam * raw[i:i + 1] + (1 - lam) * raw[j:j + 1]).clamp(0, 1)
            mixup_tensors.append(_renorm(mixed))
        mixup_imgs = torch.cat(mixup_tensors, dim=0)
    else:
        mixup_imgs = images_tensor[:0].clone()

    if n_masked > 0:
        combos = [(s, r) for s in MASK_STYLES for r in MASK_RATES]
        masked_tensors = []
        for k in range(n_masked):
            idx = rng.randint(0, n)
            style, rate = combos[k % len(combos)]
            masked = apply_masking(images_tensor[idx:idx + 1], style, rate, seed=seed + 10000 + k)
            masked_tensors.append(masked)
        masked_imgs = torch.cat(masked_tensors, dim=0)
    else:
        masked_imgs = images_tensor[:0].clone()

    return mixup_imgs, masked_imgs


def get_clean_tensors(dataset, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    imgs, labs = [], []
    for x, y in loader:
        imgs.append(x)
        labs.append(y)
    return torch.cat(imgs, 0), torch.cat(labs, 0)


def main():
    parser = argparse.ArgumentParser(description="Cache MNIST ensemble teacher targets")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--p2_data_mode", type=str, default="fake_ood", choices=["ood", "fake_ood"],
                        help="Phase 2 data: 'ood'=FashionMNIST/Omniglot, 'fake_ood'=mixup+masked samples")
    parser.add_argument("--fake_ood_mixup_frac", type=float, default=0.5,
                        help="Fraction of fake OOD assigned to mixup; remainder uses masking")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    print("Loading ensemble...")
    members = load_ensemble(args.save_dir, device)

    clean_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])

    results = {}

    # === 1. Clean MNIST train + test ===
    print("\n--- Clean MNIST train ---")
    train_set = datasets.MNIST(root=args.data_dir, train=True, download=False, transform=clean_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_probs, train_tu, train_au, train_eu, train_labels = compute_teacher_targets(members, train_loader, device)
    results["train_probs"] = train_probs
    results["train_tu"] = train_tu
    results["train_au"] = train_au
    results["train_eu"] = train_eu
    results["train_labels"] = train_labels
    print(f"  TU={train_tu.mean():.4f}  AU={train_au.mean():.4f}  EU={train_eu.mean():.4f}")

    print("\n--- Clean MNIST test ---")
    test_set = datasets.MNIST(root=args.data_dir, train=False, download=False, transform=clean_transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_probs, test_tu, test_au, test_eu, test_labels = compute_teacher_targets(members, test_loader, device)
    results["test_probs"] = test_probs
    results["test_tu"] = test_tu
    results["test_au"] = test_au
    results["test_eu"] = test_eu
    results["test_labels"] = test_labels
    print(f"  TU={test_tu.mean():.4f}  AU={test_au.mean():.4f}  EU={test_eu.mean():.4f}")

    # === 2. Corrupted MNIST train ===
    print("\n--- Corrupted MNIST train ---")
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

    # Corrupted MNIST test
    print("\n--- Corrupted MNIST test ---")
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

    results["p2_data_mode"] = np.array(args.p2_data_mode)
    if args.p2_data_mode == "fake_ood":
        n_fake = len(train_imgs) // 2
        n_mixup = int(n_fake * args.fake_ood_mixup_frac)
        n_masked = n_fake - n_mixup
        print("\n--- Fake OOD (mixup + masked, no external data) ---")
        print(f"  Target: clean={len(train_imgs)}, shifted={len(train_imgs)//2}, fake_ood={n_fake} "
              f"(mixup={n_mixup}, masked={n_masked})")

        mixup_imgs, masked_imgs = generate_fake_ood(train_imgs, n_mixup, n_masked, seed=FAKE_OOD_SEED)

        mixup_ds = TensorDataset(mixup_imgs, torch.zeros(len(mixup_imgs), dtype=torch.long))
        mixup_loader = DataLoader(mixup_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        _probs, mixup_tu, mixup_au, mixup_eu, _ = compute_teacher_targets(members, mixup_loader, device)
        results["fake_mixup_imgs"] = mixup_imgs.numpy()
        results["fake_mixup_eu"] = mixup_eu
        results["fake_mixup_tu"] = mixup_tu
        results["fake_mixup_au"] = mixup_au
        print(f"  Mixup: n={len(mixup_eu)}  TU={mixup_tu.mean():.4f}  AU={mixup_au.mean():.4f}  EU={mixup_eu.mean():.4f}")

        masked_ds = TensorDataset(masked_imgs, torch.zeros(len(masked_imgs), dtype=torch.long))
        masked_loader = DataLoader(masked_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        _probs, masked_tu, masked_au, masked_eu, _ = compute_teacher_targets(members, masked_loader, device)
        results["fake_masked_imgs"] = masked_imgs.numpy()
        results["fake_masked_eu"] = masked_eu
        results["fake_masked_tu"] = masked_tu
        results["fake_masked_au"] = masked_au
        print(f"  Masked: n={len(masked_eu)}  TU={masked_tu.mean():.4f}  AU={masked_au.mean():.4f}  EU={masked_eu.mean():.4f}")

        results["fake_ood_mixup_frac"] = np.array(args.fake_ood_mixup_frac)

    # === 3. Seen OOD (used in Phase 2 training) ===
    # Grayscale OOD → replicate to 3ch (same as MNIST)
    gray_ood_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])

    print("\n--- FashionMNIST test (Seen OOD) ---")
    fmnist_set = datasets.FashionMNIST(root=args.data_dir, train=False, download=False, transform=gray_ood_transform)
    fmnist_loader = DataLoader(fmnist_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    _probs, fmnist_tu, fmnist_au, fmnist_eu, _ = compute_teacher_targets(members, fmnist_loader, device)
    results["fashionmnist_eu"] = fmnist_eu
    results["fashionmnist_tu"] = fmnist_tu
    results["fashionmnist_au"] = fmnist_au
    print(f"  TU={fmnist_tu.mean():.4f}  AU={fmnist_au.mean():.4f}  EU={fmnist_eu.mean():.4f}  n={len(fmnist_eu)}")

    print("\n--- Omniglot evaluation (Seen OOD) ---")
    try:
        omniglot_set = datasets.Omniglot(root=args.data_dir, background=False, download=False, transform=gray_ood_transform)
        omniglot_loader = DataLoader(omniglot_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        _probs, omni_tu, omni_au, omni_eu, _ = compute_teacher_targets(members, omniglot_loader, device)
        results["omniglot_eu"] = omni_eu
        results["omniglot_tu"] = omni_tu
        results["omniglot_au"] = omni_au
        print(f"  TU={omni_tu.mean():.4f}  AU={omni_au.mean():.4f}  EU={omni_eu.mean():.4f}  n={len(omni_eu)}")
    except Exception as e:
        print(f"  Skipping Omniglot: {e}")

    # === 4. Unseen OOD ===
    # RGB datasets stay RGB — the model sees real color for the first time
    rgb_ood_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])

    unseen_ood = {
        "cifar10": lambda: datasets.CIFAR10(
            root=args.data_dir, train=False, download=False, transform=rgb_ood_transform),
        "svhn": lambda: datasets.SVHN(
            root=os.path.join(args.data_dir, "svhn"), split="test",
            download=False, transform=rgb_ood_transform),
    }

    try:
        datasets.EMNIST(root=args.data_dir, split="letters", train=False, download=False, transform=gray_ood_transform)
        unseen_ood["emnist_letters"] = lambda: datasets.EMNIST(
            root=args.data_dir, split="letters", train=False, download=False, transform=gray_ood_transform)
    except Exception as e:
        print(f"\n--- EMNIST-Letters: skipping ({e}) ---")

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

    out_path = os.path.join(args.save_dir, "teacher_targets.npz")
    np.savez(out_path, **results)
    print(f"\nSaved all targets to {out_path}")
    print(f"Keys: {sorted(results.keys())}")


if __name__ == "__main__":
    main()
