"""
Data handling for the Tiny-ImageNet pipeline.

Provides:
    - download_tiny_imagenet()     — fetch + extract the 237 MB dataset
    - TinyImageNetDataset          — custom Dataset for train / val
    - get_train_transform(aug_cfg) — diversity-aware training transforms
    - get_val_transform()          — deterministic eval transform
    - Corruption functions         — gaussian_noise, blur, contrast, jpeg, brightness, shot_noise
    - OOD data loaders             — SVHN, CIFAR-10, CIFAR-100, LSUN, iSUN, Places365
"""

from __future__ import annotations

import io
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NUM_CLASSES = 200


# ═══════════════════════════════════════════════════════════════════════════
#  Tiny-ImageNet download & dataset
# ═══════════════════════════════════════════════════════════════════════════

def download_tiny_imagenet(data_dir: str = "../data") -> str:
    """Download and extract Tiny-ImageNet-200 to ``data_dir/tiny-imagenet-200``."""
    root = os.path.join(data_dir, "tiny-imagenet-200")
    if os.path.isdir(root) and os.path.isdir(os.path.join(root, "train")):
        print(f"Tiny-ImageNet already present at {root}")
        return root

    import urllib.request

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    os.makedirs(data_dir, exist_ok=True)

    print(f"Downloading Tiny-ImageNet from {url}  ...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Extracting to {data_dir}  ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    os.remove(zip_path)
    print("Done.")
    return root


class TinyImageNetDataset(Dataset):
    """PyTorch Dataset for Tiny-ImageNet-200 (train or val split)."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = root
        self.split = split
        self.transform = transform

        wnid_path = os.path.join(root, "wnids.txt")
        with open(wnid_path) as f:
            self.classes = sorted(line.strip() for line in f if line.strip())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[str, int]] = []
        if split == "train":
            self._load_train()
        elif split == "val":
            self._load_val()
        else:
            raise ValueError(f"Unknown split: {split}")

    def _load_train(self):
        train_dir = os.path.join(self.root, "train")
        for cls_name in self.classes:
            cls_dir = os.path.join(train_dir, cls_name, "images")
            if not os.path.isdir(cls_dir):
                continue
            label = self.class_to_idx[cls_name]
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def _load_val(self):
        val_dir = os.path.join(self.root, "val")
        ann_path = os.path.join(val_dir, "val_annotations.txt")
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, wnid = parts[0], parts[1]
                if wnid in self.class_to_idx:
                    img_path = os.path.join(val_dir, "images", fname)
                    self.samples.append((img_path, self.class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ═══════════════════════════════════════════════════════════════════════════
#  Transforms
# ═══════════════════════════════════════════════════════════════════════════

def get_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _normalize_aug_tokens(aug_cfg) -> List[str]:
    """Normalize a train-augmentation config into a list of tokens."""
    if isinstance(aug_cfg, str):
        if aug_cfg in ("", "basic"):
            return []
        return [tok for tok in aug_cfg.split("+") if tok and tok != "basic"]
    if aug_cfg is None:
        return []
    return [str(tok) for tok in aug_cfg if str(tok) and str(tok) != "basic"]


def get_train_transform(aug_cfg="basic") -> transforms.Compose:
    """Return a training transform based on one or more augmentation tokens.

    Supported tokens:
        "randaugment" — RandAugment(2,9)
        "autoaugment" — AutoAugment(ImageNet policy)
        "colorjitter" — strong color jitter
        "perspective" — random perspective warp
        "erasing"     — random erasing after normalization
        "mixup"       — batch-level, ignored here
        "cutmix"      — batch-level, ignored here
    """
    aug_tokens = _normalize_aug_tokens(aug_cfg)
    aug_set = set(aug_tokens)

    base = [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
    ]

    if "randaugment" in aug_set:
        base.append(transforms.RandAugment(num_ops=2, magnitude=9))
    elif "autoaugment" in aug_set:
        base.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
    if "colorjitter" in aug_set:
        base.append(transforms.ColorJitter(0.4, 0.4, 0.4, 0.1))
    if "perspective" in aug_set:
        base.append(transforms.RandomPerspective(distortion_scale=0.25, p=0.4))

    base += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    if "erasing" in aug_set or not aug_tokens:
        base.append(transforms.RandomErasing(p=0.25))

    return transforms.Compose(base)


# ═══════════════════════════════════════════════════════════════════════════
#  Corruptions (operate on normalised tensors)
# ═══════════════════════════════════════════════════════════════════════════

def _to_tensor_norm(x):
    """Ensure x is a normalised float tensor."""
    if isinstance(x, Image.Image):
        x = transforms.ToTensor()(x)
        x = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)
    return x


def gaussian_noise(x: torch.Tensor, std: float = 0.15) -> torch.Tensor:
    return x + torch.randn_like(x) * std


def gaussian_blur(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    return transforms.GaussianBlur(kernel_size, sigma=(1.0, 2.0))(x)


def low_contrast(x: torch.Tensor, factor: float = 0.5) -> torch.Tensor:
    return x * factor


def jpeg_compression(x: torch.Tensor, quality: int = 10) -> torch.Tensor:
    """Simulate JPEG compression artifacts."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (x * std + mean).clamp(0, 1)
    img_pil = transforms.ToPILImage()(img)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    img_jpeg = Image.open(buf).convert("RGB")
    out = transforms.ToTensor()(img_jpeg)
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(out)


def brightness(x: torch.Tensor, factor: float = 1.6) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (x * std + mean).clamp(0, 1) * factor
    img = img.clamp(0, 1)
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(img)


def shot_noise(x: torch.Tensor, scale: float = 60.0) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (x * std + mean).clamp(1e-6, 1)
    noisy = torch.poisson(img * scale) / scale
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(noisy.clamp(0, 1))


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "low_contrast": low_contrast,
    "jpeg_compression": jpeg_compression,
    "brightness": brightness,
    "shot_noise": shot_noise,
}

FAKE_OOD_SEED = 2027
MIXUP_LAMBDAS = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]
MASK_STYLES = ["random_block", "random_pixel", "center_crop_mask", "multi_block", "border_mask"]
MASK_RATES = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
PATCH_SHUFFLE_GRIDS = [2, 4, 7]
CUTPASTE_BOX_FRACS = [0.15, 0.25, 0.35, 0.5]
HEAVY_NOISE_SIGMAS = [0.3, 0.5, 0.8, 1.0]
MULTI_CORRUPT_COMBOS = [
    ("gaussian_noise", "gaussian_blur"),
    ("gaussian_noise", "low_contrast"),
    ("gaussian_blur", "low_contrast"),
    ("shot_noise", "gaussian_blur"),
    ("gaussian_noise", "gaussian_blur", "low_contrast"),
    ("shot_noise", "brightness", "gaussian_blur"),
]
VIT_PATCH_SIZE = 16  # DeiT-S patch size


def _denorm_imagenet(x: torch.Tensor) -> torch.Tensor:
    shape = (1, 3, 1, 1) if x.dim() == 4 else (3, 1, 1)
    mean = x.new_tensor(IMAGENET_MEAN).view(shape)
    std = x.new_tensor(IMAGENET_STD).view(shape)
    return x * std + mean


def _renorm_imagenet(x: torch.Tensor) -> torch.Tensor:
    shape = (1, 3, 1, 1) if x.dim() == 4 else (3, 1, 1)
    mean = x.new_tensor(IMAGENET_MEAN).view(shape)
    std = x.new_tensor(IMAGENET_STD).view(shape)
    return (x - mean) / std


def apply_corruption(dataset, corruption_fn, max_samples: int = 5000):
    """Apply a corruption to a dataset, returning (images_tensor, labels)."""
    imgs, labels = [], []
    n = min(len(dataset), max_samples)
    for i in range(n):
        img, lbl = dataset[i]
        imgs.append(corruption_fn(img))
        labels.append(lbl)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)


def apply_masking(images: torch.Tensor, style: str, rate: float, seed: int = FAKE_OOD_SEED) -> torch.Tensor:
    """Mask a normalized image tensor or batch in raw pixel space."""
    squeeze = images.dim() == 3
    if squeeze:
        images = images.unsqueeze(0)

    raw = _denorm_imagenet(images).clamp(0, 1)
    out = raw.clone()
    n, _c, h, w = out.shape
    rng = torch.Generator(device=out.device).manual_seed(seed)

    if style == "random_block":
        side = max(1, int((rate * h * w) ** 0.5))
        for i in range(n):
            top = int(torch.randint(0, max(1, h - side + 1), (1,), generator=rng, device=out.device).item())
            left = int(torch.randint(0, max(1, w - side + 1), (1,), generator=rng, device=out.device).item())
            out[i, :, top:top + side, left:left + side] = 0

    elif style == "random_pixel":
        keep = torch.rand(n, 1, h, w, generator=rng, device=out.device) > rate
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
                top = int(torch.randint(0, max(1, h - side + 1), (1,), generator=rng, device=out.device).item())
                left = int(torch.randint(0, max(1, w - side + 1), (1,), generator=rng, device=out.device).item())
                out[i, :, top:top + side, left:left + side] = 0

    elif style == "border_mask":
        border = max(1, int(rate * min(h, w) / 2))
        out[:, :, :border, :] = 0
        out[:, :, -border:, :] = 0
        out[:, :, :, :border] = 0
        out[:, :, :, -border:] = 0

    else:
        raise ValueError(f"Unknown mask style: {style}")

    masked = _renorm_imagenet(out)
    return masked[0] if squeeze else masked


def apply_patch_shuffle(images: torch.Tensor, grid_size: int, seed: int = FAKE_OOD_SEED) -> torch.Tensor:
    """Shuffle image patches on a fixed grid in raw pixel space."""
    squeeze = images.dim() == 3
    if squeeze:
        images = images.unsqueeze(0)

    raw = _denorm_imagenet(images).clamp(0, 1)
    n, c, h, w = raw.shape
    if h % grid_size != 0 or w % grid_size != 0:
        raise ValueError(f"grid_size={grid_size} must divide image size ({h}, {w})")

    cell_h = h // grid_size
    cell_w = w // grid_size
    patches = raw.reshape(n, c, grid_size, cell_h, grid_size, cell_w)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
        n, grid_size * grid_size, c, cell_h, cell_w
    )

    rng = torch.Generator(device=raw.device).manual_seed(seed)
    shuffled = patches.clone()
    for i in range(n):
        perm = torch.randperm(grid_size * grid_size, generator=rng, device=raw.device)
        shuffled[i] = patches[i, perm]

    out = shuffled.reshape(n, grid_size, grid_size, c, cell_h, cell_w)
    out = out.permute(0, 3, 1, 4, 2, 5).reshape(n, c, h, w)
    out = _renorm_imagenet(out)
    return out[0] if squeeze else out


def apply_cutpaste(
    images_a: torch.Tensor,
    images_b: torch.Tensor,
    box_frac: float,
    seed: int = FAKE_OOD_SEED,
) -> torch.Tensor:
    """Paste a random patch from image B into image A in raw pixel space."""
    squeeze = images_a.dim() == 3
    if squeeze:
        images_a = images_a.unsqueeze(0)
        images_b = images_b.unsqueeze(0)

    raw_a = _denorm_imagenet(images_a).clamp(0, 1)
    raw_b = _denorm_imagenet(images_b).clamp(0, 1)
    out = raw_a.clone()
    n, _c, h, w = out.shape
    rng = torch.Generator(device=out.device).manual_seed(seed)

    side = max(1, int((box_frac * h * w) ** 0.5))
    for i in range(n):
        top_a = int(torch.randint(0, max(1, h - side + 1), (1,), generator=rng, device=out.device).item())
        left_a = int(torch.randint(0, max(1, w - side + 1), (1,), generator=rng, device=out.device).item())
        top_b = int(torch.randint(0, max(1, h - side + 1), (1,), generator=rng, device=out.device).item())
        left_b = int(torch.randint(0, max(1, w - side + 1), (1,), generator=rng, device=out.device).item())
        out[i, :, top_a:top_a + side, left_a:left_a + side] = raw_b[i, :, top_b:top_b + side, left_b:left_b + side]

    pasted = _renorm_imagenet(out)
    return pasted[0] if squeeze else pasted


def apply_heavy_noise(images: torch.Tensor, sigma: float, seed: int = FAKE_OOD_SEED) -> torch.Tensor:
    """Apply very heavy Gaussian noise in raw pixel space."""
    squeeze = images.dim() == 3
    if squeeze:
        images = images.unsqueeze(0)
    raw = _denorm_imagenet(images).clamp(0, 1)
    rng = torch.Generator(device=raw.device).manual_seed(seed)
    noisy = raw + torch.randn(raw.shape, generator=rng, device=raw.device) * sigma
    out = _renorm_imagenet(noisy.clamp(0, 1))
    return out[0] if squeeze else out


def apply_multi_corrupt(images: torch.Tensor, corrupt_names: tuple, seed: int = FAKE_OOD_SEED) -> torch.Tensor:
    """Stack 2-3 corruptions on the same image."""
    out = images.clone()
    for cname in corrupt_names:
        if cname in CORRUPTIONS:
            out = CORRUPTIONS[cname](out)
    return out


def apply_pixel_permute(images: torch.Tensor, patch_size: int = VIT_PATCH_SIZE,
                        seed: int = FAKE_OOD_SEED) -> torch.Tensor:
    """Randomly permute pixels within each ViT-sized patch, destroying local structure."""
    squeeze = images.dim() == 3
    if squeeze:
        images = images.unsqueeze(0)
    raw = _denorm_imagenet(images).clamp(0, 1)
    n, c, h, w = raw.shape
    rng = torch.Generator(device=raw.device).manual_seed(seed)

    ph = h // patch_size
    pw = w // patch_size
    # reshape into patches: (n, c, ph, patch_size, pw, patch_size)
    crop_h, crop_w = ph * patch_size, pw * patch_size
    cropped = raw[:, :, :crop_h, :crop_w]
    patches = cropped.reshape(n, c, ph, patch_size, pw, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5)  # (n, ph, pw, c, patch_size, patch_size)
    patches = patches.reshape(n, ph * pw, c, patch_size * patch_size)

    for i in range(n):
        for p in range(ph * pw):
            perm = torch.randperm(patch_size * patch_size, generator=rng, device=raw.device)
            patches[i, p] = patches[i, p, :, perm]

    patches = patches.reshape(n, ph, pw, c, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5)  # (n, c, ph, patch_size, pw, patch_size)
    out = patches.reshape(n, c, crop_h, crop_w)
    # Pad back if needed
    if crop_h < h or crop_w < w:
        full = raw.clone()
        full[:, :, :crop_h, :crop_w] = out
        out = full
    out = _renorm_imagenet(out)
    return out[0] if squeeze else out


class FakeHeavyNoiseDataset(Dataset):
    """On-the-fly heavy Gaussian noise samples."""

    def __init__(self, base_ds: Dataset, idxs, sigmas, seed_base: int):
        self.base_ds = base_ds
        self.idxs = np.asarray(idxs, dtype=np.int64)
        self.sigmas = np.asarray(sigmas, dtype=np.float32)
        self.seed_base = int(seed_base)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        img, _ = self.base_ds[int(self.idxs[idx])]
        sigma = float(self.sigmas[idx])
        noisy = apply_heavy_noise(img, sigma, seed=self.seed_base + int(idx))
        return noisy, 0


class FakeMultiCorruptDataset(Dataset):
    """On-the-fly multi-corruption stacked samples."""

    def __init__(self, base_ds: Dataset, idxs, combo_idxs):
        self.base_ds = base_ds
        self.idxs = np.asarray(idxs, dtype=np.int64)
        self.combo_idxs = np.asarray(combo_idxs, dtype=np.int64)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        img, _ = self.base_ds[int(self.idxs[idx])]
        combo = MULTI_CORRUPT_COMBOS[int(self.combo_idxs[idx]) % len(MULTI_CORRUPT_COMBOS)]
        corrupted = apply_multi_corrupt(img, combo)
        return corrupted, 0


class FakePixelPermuteDataset(Dataset):
    """On-the-fly pixel permutation within ViT patches."""

    def __init__(self, base_ds: Dataset, idxs, seed_base: int):
        self.base_ds = base_ds
        self.idxs = np.asarray(idxs, dtype=np.int64)
        self.seed_base = int(seed_base)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        img, _ = self.base_ds[int(self.idxs[idx])]
        permuted = apply_pixel_permute(img, seed=self.seed_base + int(idx))
        return permuted, 0


def generate_fake_ood_specs(
    num_source: int,
    n_mixup: int,
    n_masked: int,
    n_patchshuffle: int = 0,
    n_cutpaste: int = 0,
    n_heavy_noise: int = 0,
    n_multi_corrupt: int = 0,
    n_pixel_permute: int = 0,
    seed: int = FAKE_OOD_SEED,
) -> Dict[str, np.ndarray]:
    """Generate deterministic metadata for fake OOD regeneration."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_source)

    mix_a = np.empty(n_mixup, dtype=np.int64)
    mix_b = np.empty(n_mixup, dtype=np.int64)
    mix_lam = np.empty(n_mixup, dtype=np.float32)
    for k in range(n_mixup):
        mix_a[k] = int(perm[(2 * k) % num_source])
        mix_b[k] = int(perm[(2 * k + 1) % num_source])
        mix_lam[k] = float(MIXUP_LAMBDAS[k % len(MIXUP_LAMBDAS)])

    combos = [(s_idx, rate) for s_idx in range(len(MASK_STYLES)) for rate in MASK_RATES]
    masked_idx = rng.randint(0, num_source, size=n_masked).astype(np.int64)
    masked_style_idx = np.empty(n_masked, dtype=np.int64)
    masked_rates = np.empty(n_masked, dtype=np.float32)
    for k in range(n_masked):
        style_idx, rate = combos[k % len(combos)]
        masked_style_idx[k] = style_idx
        masked_rates[k] = rate

    specs = {
        "fake_mixup_idx_a": mix_a,
        "fake_mixup_idx_b": mix_b,
        "fake_mixup_lambdas": mix_lam,
        "fake_masked_idx": masked_idx,
        "fake_masked_style_idx": masked_style_idx,
        "fake_masked_rates": masked_rates,
        "fake_masked_seed_base": np.array(seed + 10000, dtype=np.int64),
    }

    if n_patchshuffle > 0:
        patch_idx = rng.randint(0, num_source, size=n_patchshuffle).astype(np.int64)
        patch_grid_sizes = np.empty(n_patchshuffle, dtype=np.int64)
        for k in range(n_patchshuffle):
            patch_grid_sizes[k] = PATCH_SHUFFLE_GRIDS[k % len(PATCH_SHUFFLE_GRIDS)]
        specs.update({
            "fake_patchshuffle_idx": patch_idx,
            "fake_patchshuffle_grid_sizes": patch_grid_sizes,
            "fake_patchshuffle_seed_base": np.array(seed + 20000, dtype=np.int64),
        })

    if n_cutpaste > 0:
        cut_idx_a = np.empty(n_cutpaste, dtype=np.int64)
        cut_idx_b = np.empty(n_cutpaste, dtype=np.int64)
        cut_box_fracs = np.empty(n_cutpaste, dtype=np.float32)
        for k in range(n_cutpaste):
            cut_idx_a[k] = int(perm[(3 * k) % num_source])
            cut_idx_b[k] = int(perm[(3 * k + 1) % num_source])
            cut_box_fracs[k] = float(CUTPASTE_BOX_FRACS[k % len(CUTPASTE_BOX_FRACS)])
        specs.update({
            "fake_cutpaste_idx_a": cut_idx_a,
            "fake_cutpaste_idx_b": cut_idx_b,
            "fake_cutpaste_box_fracs": cut_box_fracs,
            "fake_cutpaste_seed_base": np.array(seed + 30000, dtype=np.int64),
        })

    if n_heavy_noise > 0:
        hn_idx = rng.randint(0, num_source, size=n_heavy_noise).astype(np.int64)
        hn_sigmas = np.empty(n_heavy_noise, dtype=np.float32)
        for k in range(n_heavy_noise):
            hn_sigmas[k] = float(HEAVY_NOISE_SIGMAS[k % len(HEAVY_NOISE_SIGMAS)])
        specs.update({
            "fake_heavy_noise_idx": hn_idx,
            "fake_heavy_noise_sigmas": hn_sigmas,
            "fake_heavy_noise_seed_base": np.array(seed + 40000, dtype=np.int64),
        })

    if n_multi_corrupt > 0:
        mc_idx = rng.randint(0, num_source, size=n_multi_corrupt).astype(np.int64)
        mc_combo_idx = np.empty(n_multi_corrupt, dtype=np.int64)
        for k in range(n_multi_corrupt):
            mc_combo_idx[k] = k % len(MULTI_CORRUPT_COMBOS)
        specs.update({
            "fake_multi_corrupt_idx": mc_idx,
            "fake_multi_corrupt_combo_idx": mc_combo_idx,
        })

    if n_pixel_permute > 0:
        pp_idx = rng.randint(0, num_source, size=n_pixel_permute).astype(np.int64)
        specs.update({
            "fake_pixel_permute_idx": pp_idx,
            "fake_pixel_permute_seed_base": np.array(seed + 50000, dtype=np.int64),
        })

    return specs


class FakeMixupDataset(Dataset):
    """On-the-fly mixup samples defined by precomputed source indices."""

    def __init__(self, base_ds: Dataset, idx_a, idx_b, lambdas):
        self.base_ds = base_ds
        self.idx_a = np.asarray(idx_a, dtype=np.int64)
        self.idx_b = np.asarray(idx_b, dtype=np.int64)
        self.lambdas = np.asarray(lambdas, dtype=np.float32)

    def __len__(self):
        return len(self.idx_a)

    def __getitem__(self, idx):
        img_a, _ = self.base_ds[int(self.idx_a[idx])]
        img_b, _ = self.base_ds[int(self.idx_b[idx])]
        lam = float(self.lambdas[idx])
        raw_a = _denorm_imagenet(img_a).clamp(0, 1)
        raw_b = _denorm_imagenet(img_b).clamp(0, 1)
        mixed = (lam * raw_a + (1 - lam) * raw_b).clamp(0, 1)
        return _renorm_imagenet(mixed), 0


class FakeMaskedDataset(Dataset):
    """On-the-fly masked samples defined by precomputed source indices."""

    def __init__(self, base_ds: Dataset, idxs, style_idxs, rates, seed_base: int):
        self.base_ds = base_ds
        self.idxs = np.asarray(idxs, dtype=np.int64)
        self.style_idxs = np.asarray(style_idxs, dtype=np.int64)
        self.rates = np.asarray(rates, dtype=np.float32)
        self.seed_base = int(seed_base)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        img, _ = self.base_ds[int(self.idxs[idx])]
        style = MASK_STYLES[int(self.style_idxs[idx])]
        rate = float(self.rates[idx])
        masked = apply_masking(img, style, rate, seed=self.seed_base + int(idx))
        return masked, 0


class FakePatchShuffleDataset(Dataset):
    """On-the-fly patch-shuffled samples defined by precomputed source indices."""

    def __init__(self, base_ds: Dataset, idxs, grid_sizes, seed_base: int):
        self.base_ds = base_ds
        self.idxs = np.asarray(idxs, dtype=np.int64)
        self.grid_sizes = np.asarray(grid_sizes, dtype=np.int64)
        self.seed_base = int(seed_base)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        img, _ = self.base_ds[int(self.idxs[idx])]
        grid = int(self.grid_sizes[idx])
        shuffled = apply_patch_shuffle(img, grid, seed=self.seed_base + int(idx))
        return shuffled, 0


class FakeCutPasteDataset(Dataset):
    """On-the-fly cut-paste samples defined by precomputed source indices."""

    def __init__(self, base_ds: Dataset, idx_a, idx_b, box_fracs, seed_base: int):
        self.base_ds = base_ds
        self.idx_a = np.asarray(idx_a, dtype=np.int64)
        self.idx_b = np.asarray(idx_b, dtype=np.int64)
        self.box_fracs = np.asarray(box_fracs, dtype=np.float32)
        self.seed_base = int(seed_base)

    def __len__(self):
        return len(self.idx_a)

    def __getitem__(self, idx):
        img_a, _ = self.base_ds[int(self.idx_a[idx])]
        img_b, _ = self.base_ds[int(self.idx_b[idx])]
        box_frac = float(self.box_fracs[idx])
        cutpaste = apply_cutpaste(img_a, img_b, box_frac, seed=self.seed_base + int(idx))
        return cutpaste, 0


def build_fake_ood_datasets(base_ds: Dataset, specs: Dict[str, np.ndarray]):
    """Rebuild fake OOD family datasets from cached metadata."""
    families: Dict[str, Dataset] = {}

    if {"fake_mixup_idx_a", "fake_mixup_idx_b", "fake_mixup_lambdas"}.issubset(specs):
        families["mixup"] = FakeMixupDataset(
            base_ds,
            specs["fake_mixup_idx_a"],
            specs["fake_mixup_idx_b"],
            specs["fake_mixup_lambdas"],
        )

    if {"fake_masked_idx", "fake_masked_style_idx", "fake_masked_rates", "fake_masked_seed_base"}.issubset(specs):
        seed_base = int(np.asarray(specs["fake_masked_seed_base"]).reshape(-1)[0])
        families["masked"] = FakeMaskedDataset(
            base_ds,
            specs["fake_masked_idx"],
            specs["fake_masked_style_idx"],
            specs["fake_masked_rates"],
            seed_base=seed_base,
        )

    if {"fake_patchshuffle_idx", "fake_patchshuffle_grid_sizes", "fake_patchshuffle_seed_base"}.issubset(specs):
        seed_base = int(np.asarray(specs["fake_patchshuffle_seed_base"]).reshape(-1)[0])
        families["patchshuffle"] = FakePatchShuffleDataset(
            base_ds,
            specs["fake_patchshuffle_idx"],
            specs["fake_patchshuffle_grid_sizes"],
            seed_base=seed_base,
        )

    if {"fake_cutpaste_idx_a", "fake_cutpaste_idx_b", "fake_cutpaste_box_fracs", "fake_cutpaste_seed_base"}.issubset(specs):
        seed_base = int(np.asarray(specs["fake_cutpaste_seed_base"]).reshape(-1)[0])
        families["cutpaste"] = FakeCutPasteDataset(
            base_ds,
            specs["fake_cutpaste_idx_a"],
            specs["fake_cutpaste_idx_b"],
            specs["fake_cutpaste_box_fracs"],
            seed_base=seed_base,
        )

    if {"fake_heavy_noise_idx", "fake_heavy_noise_sigmas", "fake_heavy_noise_seed_base"}.issubset(specs):
        seed_base = int(np.asarray(specs["fake_heavy_noise_seed_base"]).reshape(-1)[0])
        families["heavy_noise"] = FakeHeavyNoiseDataset(
            base_ds,
            specs["fake_heavy_noise_idx"],
            specs["fake_heavy_noise_sigmas"],
            seed_base=seed_base,
        )

    if {"fake_multi_corrupt_idx", "fake_multi_corrupt_combo_idx"}.issubset(specs):
        families["multi_corrupt"] = FakeMultiCorruptDataset(
            base_ds,
            specs["fake_multi_corrupt_idx"],
            specs["fake_multi_corrupt_combo_idx"],
        )

    if {"fake_pixel_permute_idx", "fake_pixel_permute_seed_base"}.issubset(specs):
        seed_base = int(np.asarray(specs["fake_pixel_permute_seed_base"]).reshape(-1)[0])
        families["pixel_permute"] = FakePixelPermuteDataset(
            base_ds,
            specs["fake_pixel_permute_idx"],
            seed_base=seed_base,
        )

    return families


# ═══════════════════════════════════════════════════════════════════════════
#  OOD data loaders
# ═══════════════════════════════════════════════════════════════════════════

def _ood_transform():
    """Standard OOD transform — resize to 224, centre-crop, normalise."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _ood_transform_rgb():
    """OOD transform for already-RGB datasets."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_ood_loaders(
    data_dir: str = "../data",
    batch_size: int = 128,
    max_samples: int = 5000,
) -> Dict[str, DataLoader]:
    """Build OOD data loaders.  Missing datasets are silently skipped."""
    loaders: Dict[str, DataLoader] = {}
    tf_rgb = _ood_transform_rgb()
    tf_gray = _ood_transform()

    # SVHN
    try:
        ds = datasets.SVHN(os.path.join(data_dir, "svhn"), split="test",
                           transform=tf_rgb, download=False)
        sub = Subset(ds, list(range(min(max_samples, len(ds)))))
        loaders["SVHN"] = DataLoader(sub, batch_size=batch_size, num_workers=2)
    except Exception as e:
        print(f"  [skip] SVHN: {e}")

    # CIFAR-10
    try:
        ds = datasets.CIFAR10(data_dir, train=False, transform=tf_rgb, download=False)
        sub = Subset(ds, list(range(min(max_samples, len(ds)))))
        loaders["CIFAR-10"] = DataLoader(sub, batch_size=batch_size, num_workers=2)
    except Exception as e:
        print(f"  [skip] CIFAR-10: {e}")

    # CIFAR-100
    try:
        ds = datasets.CIFAR100(data_dir, train=False, transform=tf_rgb, download=False)
        sub = Subset(ds, list(range(min(max_samples, len(ds)))))
        loaders["CIFAR-100"] = DataLoader(sub, batch_size=batch_size, num_workers=2)
    except Exception as e:
        print(f"  [skip] CIFAR-100: {e}")

    # LSUN (classroom subset -- needs lmdb)
    try:
        ds = datasets.LSUN(os.path.join(data_dir, "lsun"), classes=["classroom_val"],
                           transform=tf_rgb)
        sub = Subset(ds, list(range(min(max_samples, len(ds)))))
        loaders["LSUN"] = DataLoader(sub, batch_size=batch_size, num_workers=2)
    except Exception as e:
        print(f"  [skip] LSUN: {e}")

    # iSUN
    try:
        isun_dir = os.path.join(data_dir, "iSUN")
        if os.path.isdir(isun_dir):
            ds = datasets.ImageFolder(isun_dir, transform=tf_rgb)
            sub = Subset(ds, list(range(min(max_samples, len(ds)))))
            loaders["iSUN"] = DataLoader(sub, batch_size=batch_size, num_workers=2)
        else:
            print(f"  [skip] iSUN: directory not found at {isun_dir}")
    except Exception as e:
        print(f"  [skip] iSUN: {e}")

    # Places365
    try:
        ds = datasets.Places365(
            os.path.join(data_dir, "places365"), split="val", small=True,
            transform=tf_rgb, download=False,
        )
        sub = Subset(ds, list(range(min(max_samples, len(ds)))))
        loaders["Places365"] = DataLoader(sub, batch_size=batch_size, num_workers=2)
    except Exception as e:
        print(f"  [skip] Places365: {e}")

    # STL10 — torchvision stores in <root>/stl10_binary/
    try:
        ds = datasets.STL10(data_dir, split="test", transform=tf_rgb, download=False)
        sub = Subset(ds, list(range(min(max_samples, len(ds)))))
        loaders["STL10"] = DataLoader(sub, batch_size=batch_size, num_workers=2)
    except Exception as e:
        print(f"  [skip] STL10: {e}")

    # DTD — torchvision stores in <root>/dtd/
    try:
        ds = datasets.DTD(data_dir, split="test", transform=tf_rgb, download=False)
        sub = Subset(ds, list(range(min(max_samples, len(ds)))))
        loaders["DTD"] = DataLoader(sub, batch_size=batch_size, num_workers=2)
    except Exception as e:
        print(f"  [skip] DTD: {e}")

    # FashionMNIST — torchvision stores in <root>/FashionMNIST/
    try:
        ds = datasets.FashionMNIST(data_dir, train=False, transform=tf_gray, download=False)
        sub = Subset(ds, list(range(min(max_samples, len(ds)))))
        loaders["FashionMNIST"] = DataLoader(sub, batch_size=batch_size, num_workers=2)
    except Exception as e:
        print(f"  [skip] FashionMNIST: {e}")

    # MNIST — torchvision stores in <root>/MNIST/
    try:
        ds = datasets.MNIST(data_dir, train=False, transform=tf_gray, download=False)
        sub = Subset(ds, list(range(min(max_samples, len(ds)))))
        loaders["MNIST"] = DataLoader(sub, batch_size=batch_size, num_workers=2)
    except Exception as e:
        print(f"  [skip] MNIST: {e}")

    return loaders


# ═══════════════════════════════════════════════════════════════════════════
#  CLI: download
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data_dir", type=str, default="../data")
    args = parser.parse_args()

    if args.download:
        download_tiny_imagenet(args.data_dir)
    else:
        print("Pass --download to fetch Tiny-ImageNet-200.")
