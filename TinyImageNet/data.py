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
            for fname in os.listdir(cls_dir):
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


def get_train_transform(aug_cfg: str = "basic") -> transforms.Compose:
    """Return a training transform based on augmentation config.

    aug_cfg options:
        "basic"       — random crop + flip
        "randaugment" — basic + RandAugment(2,9)
        "autoaugment" — basic + AutoAugment(ImageNet policy)
        "colorjitter" — basic + strong color jitter
        "cutmix"      — (handled at batch level, use basic here)
    """
    base = [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
    ]

    if aug_cfg == "randaugment":
        base.append(transforms.RandAugment(num_ops=2, magnitude=9))
    elif aug_cfg == "autoaugment":
        base.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
    elif aug_cfg == "colorjitter":
        base.append(transforms.ColorJitter(0.4, 0.4, 0.4, 0.1))

    base += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    if aug_cfg == "randaugment" or aug_cfg == "basic":
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


def apply_corruption(dataset, corruption_fn, max_samples: int = 5000):
    """Apply a corruption to a dataset, returning (images_tensor, labels)."""
    imgs, labels = [], []
    n = min(len(dataset), max_samples)
    for i in range(n):
        img, lbl = dataset[i]
        imgs.append(corruption_fn(img))
        labels.append(lbl)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)


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
