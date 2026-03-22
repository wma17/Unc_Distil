"""
Data handling for Pascal VOC 2012 semantic segmentation pipeline.

Provides:
    - VOC 2012 + SBD augmented training set (10,582 images)
    - VOC 2012 val set (1,449 images)
    - Training transforms with diversity options
    - Corruption functions for Phase 2 Tier 2
    - Fake OOD generation (mixup + block masking) for Phase 2 Tier 3
    - OOD loaders: COCO-exclusive, DTD
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 21
IGNORE_INDEX = 255
CROP_SIZE = 512


# ═══════════════════════════════════════════════════════════════════════════
#  VOC 2012 + SBD Dataset
# ═══════════════════════════════════════════════════════════════════════════

class VOCSegDataset(Dataset):
    """Pascal VOC 2012 segmentation dataset with optional SBD augmentation.

    Uses torchvision.datasets.VOCSegmentation for the base VOC split,
    then adds SBD images for the augmented training set.
    """

    def __init__(self, root: str, split: str = "train", transform=None,
                 target_transform=None, use_sbd: bool = True):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.images: List[str] = []
        self.masks: List[str] = []

        # VOC 2012
        voc_root = os.path.join(root, "VOCdevkit", "VOC2012")
        if not os.path.exists(voc_root):
            # Download
            datasets.VOCSegmentation(root, year="2012", image_set=split, download=True)

        seg_dir = os.path.join(voc_root, "SegmentationClass")
        img_dir = os.path.join(voc_root, "JPEGImages")
        set_file = os.path.join(voc_root, "ImageSets", "Segmentation", f"{split}.txt")

        if os.path.exists(set_file):
            with open(set_file) as f:
                ids = [line.strip() for line in f if line.strip()]
            for img_id in ids:
                img_path = os.path.join(img_dir, f"{img_id}.jpg")
                mask_path = os.path.join(seg_dir, f"{img_id}.png")
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.images.append(img_path)
                    self.masks.append(mask_path)

        # SBD augmentation (train only)
        if split == "train" and use_sbd:
            sbd_root = os.path.join(root, "benchmark_RELEASE", "dataset")
            if not os.path.exists(sbd_root):
                sbd_root = os.path.join(root, "SBD", "benchmark_RELEASE", "dataset")
            if os.path.exists(sbd_root):
                self._add_sbd(sbd_root, set(ids) if 'ids' in dir() else set())

        print(f"  VOC {split}: {len(self.images)} images"
              f"{' (+ SBD)' if split == 'train' and use_sbd else ''}")

    def _add_sbd(self, sbd_root, voc_ids):
        """Add SBD images not already in VOC train."""
        sbd_img_dir = os.path.join(sbd_root, "img")
        sbd_cls_dir = os.path.join(sbd_root, "cls")
        sbd_train_file = os.path.join(sbd_root, "train.txt")

        if not os.path.exists(sbd_train_file):
            return

        with open(sbd_train_file) as f:
            sbd_ids = [line.strip() for line in f if line.strip()]

        added = 0
        for img_id in sbd_ids:
            if img_id in voc_ids:
                continue
            img_path = os.path.join(sbd_img_dir, f"{img_id}.jpg")
            # SBD masks are .mat files - need conversion
            mat_path = os.path.join(sbd_cls_dir, f"{img_id}.mat")
            png_path = os.path.join(sbd_cls_dir, f"{img_id}.png")

            if os.path.exists(img_path) and os.path.exists(png_path):
                self.images.append(img_path)
                self.masks.append(png_path)
                added += 1
            elif os.path.exists(img_path) and os.path.exists(mat_path):
                # Convert .mat to .png on first access
                try:
                    from scipy.io import loadmat
                    mat = loadmat(mat_path)
                    mask = mat["GTcls"][0, 0]["Segmentation"]
                    mask_img = Image.fromarray(mask.astype(np.uint8))
                    mask_img.save(png_path)
                    self.images.append(img_path)
                    self.masks.append(png_path)
                    added += 1
                except Exception:
                    pass

        if added > 0:
            print(f"    Added {added} SBD images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        if self.transform is not None:
            img, mask = self.transform(img, mask)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask


# ═══════════════════════════════════════════════════════════════════════════
#  Transforms (joint image + mask)
# ═══════════════════════════════════════════════════════════════════════════

class SegTransformTrain:
    """Joint training transform for image + mask."""

    def __init__(self, crop_size=CROP_SIZE, aug_mode="default"):
        self.crop_size = crop_size
        self.aug_mode = aug_mode

    def __call__(self, img, mask):
        # Random horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Resize shorter side to crop_size
        w, h = img.size
        scale = self.crop_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        # Random crop
        if new_w > self.crop_size or new_h > self.crop_size:
            i = random.randint(0, max(0, new_h - self.crop_size))
            j = random.randint(0, max(0, new_w - self.crop_size))
            img = img.crop((j, i, j + self.crop_size, i + self.crop_size))
            mask = mask.crop((j, i, j + self.crop_size, i + self.crop_size))

        # Augmentation diversity
        if self.aug_mode == "colorjitter":
            img = transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)(img)
        elif self.aug_mode == "scale":
            s = random.uniform(0.5, 2.0)
            nw, nh = int(self.crop_size * s), int(self.crop_size * s)
            img = img.resize((nw, nh), Image.BILINEAR)
            mask = mask.resize((nw, nh), Image.NEAREST)
            # Re-crop to crop_size
            if nw >= self.crop_size and nh >= self.crop_size:
                i = random.randint(0, nh - self.crop_size)
                j = random.randint(0, nw - self.crop_size)
                img = img.crop((j, i, j + self.crop_size, i + self.crop_size))
                mask = mask.crop((j, i, j + self.crop_size, i + self.crop_size))
            else:
                img = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
                mask = mask.resize((self.crop_size, self.crop_size), Image.NEAREST)
        elif self.aug_mode == "rotation":
            angle = random.uniform(-10, 10)
            img = img.rotate(angle, Image.BILINEAR, fillcolor=(128, 128, 128))
            mask = mask.rotate(angle, Image.NEAREST, fillcolor=IGNORE_INDEX)

        # Pad if needed
        w, h = img.size
        if w < self.crop_size or h < self.crop_size:
            pad_w = max(0, self.crop_size - w)
            pad_h = max(0, self.crop_size - h)
            img = transforms.functional.pad(img, (0, 0, pad_w, pad_h), fill=0)
            mask = transforms.functional.pad(mask, (0, 0, pad_w, pad_h),
                                              fill=IGNORE_INDEX)

        # To tensor + normalize
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(img)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return img, mask


class SegTransformVal:
    """Validation transform: resize + center crop + normalize."""

    def __init__(self, crop_size=CROP_SIZE):
        self.crop_size = crop_size

    def __call__(self, img, mask):
        # Resize shorter side
        w, h = img.size
        scale = self.crop_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        # Center crop
        left = (new_w - self.crop_size) // 2
        top = (new_h - self.crop_size) // 2
        img = img.crop((left, top, left + self.crop_size, top + self.crop_size))
        mask = mask.crop((left, top, left + self.crop_size, top + self.crop_size))

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(img)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img, mask


# ═══════════════════════════════════════════════════════════════════════════
#  Corruptions (Phase 2 Tier 2)
# ═══════════════════════════════════════════════════════════════════════════

def gaussian_noise_img(img_tensor, sigma=0.1):
    return img_tensor + torch.randn_like(img_tensor) * sigma


def gaussian_blur_img(img_tensor, kernel_size=5, sigma=2.0):
    from torchvision.transforms.functional import gaussian_blur
    return gaussian_blur(img_tensor, kernel_size, sigma)


def low_contrast_img(img_tensor, factor=0.5):
    mean = img_tensor.mean()
    return mean + factor * (img_tensor - mean)


CORRUPTIONS = {
    "gaussian_noise": lambda x: gaussian_noise_img(x, sigma=0.1),
    "gaussian_blur": lambda x: gaussian_blur_img(x, kernel_size=5, sigma=2.0),
    "low_contrast": lambda x: low_contrast_img(x, factor=0.5),
}


# ═══════════════════════════════════════════════════════════════════════════
#  EU target datasets (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════

class EUSegDataset(Dataset):
    """Image + per-pixel EU target for Phase 2 training."""

    def __init__(self, images, eu_maps):
        """
        images: list of image tensors (C, H, W) or tensor (N, C, H, W)
        eu_maps: numpy array (N, H, W) of per-pixel EU targets
        """
        self.images = images
        self.eu_maps = eu_maps

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        eu = torch.from_numpy(self.eu_maps[idx].astype(np.float32))
        return img, eu


class CorruptedEUSegDataset(Dataset):
    """Apply corruption to images and return with EU targets."""

    def __init__(self, images, eu_maps, corruption_fn):
        self.images = images
        self.eu_maps = eu_maps
        self.corruption_fn = corruption_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.corruption_fn(self.images[idx])
        eu = torch.from_numpy(self.eu_maps[idx].astype(np.float32))
        return img, eu


class MixupEUSegDataset(Dataset):
    """Mixup pairs of images for Phase 2 Tier 3 synthetic OOD."""

    def __init__(self, images, eu_maps, n_samples, seed=2027):
        self.images = images
        self.eu_maps = eu_maps
        self.n_samples = n_samples
        self.lambdas = [0.2, 0.5, 0.8]
        self.rng = np.random.RandomState(seed)
        self.idx_a = self.rng.randint(0, len(images), size=n_samples)
        self.idx_b = self.rng.randint(0, len(images), size=n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        lam = self.lambdas[idx % len(self.lambdas)]
        img_a = self.images[int(self.idx_a[idx])]
        img_b = self.images[int(self.idx_b[idx])]
        mixed = lam * img_a + (1 - lam) * img_b
        # Use max EU from both as target (conservative)
        eu_a = self.eu_maps[int(self.idx_a[idx])]
        eu_b = self.eu_maps[int(self.idx_b[idx])]
        eu = np.maximum(eu_a, eu_b).astype(np.float32)
        return mixed, torch.from_numpy(eu)


class BlockMaskedEUSegDataset(Dataset):
    """Random block masking for Phase 2 Tier 3."""

    def __init__(self, images, eu_maps, n_samples, seed=2027):
        self.images = images
        self.eu_maps = eu_maps
        self.n_samples = n_samples
        self.mask_rates = [0.3, 0.5, 0.7]
        self.rng = np.random.RandomState(seed)
        self.idxs = self.rng.randint(0, len(images), size=n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        rate = self.mask_rates[idx % len(self.mask_rates)]
        img = self.images[int(self.idxs[idx])].clone()
        _, h, w = img.shape
        side = max(1, int((rate * h * w) ** 0.5))
        top = random.randint(0, max(0, h - side))
        left = random.randint(0, max(0, w - side))
        img[:, top:top + side, left:left + side] = 0
        eu = self.eu_maps[int(self.idxs[idx])].astype(np.float32)
        return img, torch.from_numpy(eu)


# ═══════════════════════════════════════════════════════════════════════════
#  OOD Datasets
# ═══════════════════════════════════════════════════════════════════════════

def get_dtd_loader(data_dir, batch_size=8, crop_size=CROP_SIZE):
    """DTD (far-OOD) — texture images, no objects."""
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    try:
        ds = datasets.DTD(data_dir, split="test", download=True, transform=transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    except Exception as e:
        print(f"  [skip] DTD: {e}")
        return None


def get_coco_loader(data_dir, batch_size=8, crop_size=CROP_SIZE, max_samples=2000):
    """COCO val2017 images as near-OOD. Uses ImageFolder if available."""
    coco_dir = os.path.join(data_dir, "coco_val2017")
    if not os.path.isdir(coco_dir):
        # Try alternative paths
        for alt in ["coco/val2017", "COCO/val2017"]:
            alt_path = os.path.join(data_dir, alt)
            if os.path.isdir(alt_path):
                coco_dir = alt_path
                break
        else:
            print(f"  [skip] COCO: directory not found at {coco_dir}")
            return None

    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    try:
        from torchvision.datasets import ImageFolder
        # Need a folder-of-folders structure; wrap if needed
        if any(f.endswith(('.jpg', '.png')) for f in os.listdir(coco_dir)[:5]):
            # Flat directory, create a simple wrapper
            class FlatImageDataset(Dataset):
                def __init__(self, root, transform, max_n):
                    self.paths = sorted([
                        os.path.join(root, f)
                        for f in os.listdir(root)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ])[:max_n]
                    self.transform = transform

                def __len__(self):
                    return len(self.paths)

                def __getitem__(self, idx):
                    img = Image.open(self.paths[idx]).convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    return img, 0  # dummy label

            ds = FlatImageDataset(coco_dir, transform, max_samples)
        else:
            ds = ImageFolder(coco_dir, transform=transform)

        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    except Exception as e:
        print(f"  [skip] COCO: {e}")
        return None
