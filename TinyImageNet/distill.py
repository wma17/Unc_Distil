"""
Two-phase knowledge distillation for the LoRA ViT ensemble → single student.

Phase 1 — Classification KD (partial-freeze + heavy augmentation):
    L₁ = (1-α)·CE(y, p_s) + α·τ²·KL(p_t ∥ p_s)
    Last N blocks + head trainable.  MixUp/CutMix, drop-path, label smoothing,
    gradient clipping, EMA.

Phase 2 — EU head regression:
    L₂ = MSE(EU_s, EU_t) + β·PairwiseRankingLoss
    Backbone + classification head frozen; only EU head trained.
    Data mix: 50 % clean ID + 25 % corrupted + 25 % OOD (SVHN + CIFAR-100).

Usage:
    python distill.py --save_dir ./checkpoints --gpu 0
    python distill.py --save_dir ./checkpoints --gpu 0 --phase2_only
"""

from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset

from data import (
    CORRUPTIONS,
    build_fake_ood_datasets,
    TinyImageNetDataset,
    download_tiny_imagenet,
    get_ood_loaders,
    get_train_transform,
    get_val_transform,
    apply_corruption,
)
from models import create_student


# ── Loss / augmentation utilities ─────────────────────────────────────────

def log1p_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE in log(1+x) space — compresses long tail, preserves ranking."""
    return F.mse_loss(torch.log1p(pred.clamp(min=0)), torch.log1p(target))


def asymmetric_log1p_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                               under_weight: float = 2.0) -> torch.Tensor:
    """MSE in log(1+x) space with heavier penalty for under-prediction of EU."""
    log_pred = torch.log1p(pred.clamp(min=0))
    log_tgt = torch.log1p(target)
    sq_err = (log_pred - log_tgt) ** 2
    weights = torch.where(pred < target, under_weight, 1.0)
    return (weights * sq_err).mean()


def pairwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor,
                          margin: float = 0.0) -> torch.Tensor:
    """Encourage correct relative ordering of EU predictions."""
    n = pred.size(0)
    if n < 2:
        return torch.tensor(0.0, device=pred.device)
    i = torch.randint(0, n, (n,), device=pred.device)
    j = torch.randint(0, n, (n,), device=pred.device)
    sign = (target[i] - target[j]).sign()
    diff = pred[i] - pred[j]
    return F.margin_ranking_loss(diff, -diff, sign, margin=margin)


def mixup_data(x, y_hard, y_soft, alpha=0.2):
    """MixUp: interpolate images and labels."""
    if alpha <= 0:
        return x, y_hard, y_soft, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, (y_hard, y_hard[idx]), (y_soft, y_soft[idx]), lam


def cutmix_data(x, y_hard, y_soft, alpha=1.0):
    """CutMix: cut-and-paste rectangular regions between images."""
    if alpha <= 0:
        return x, y_hard, y_soft, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cy = np.random.randint(0, H)
    cx = np.random.randint(0, W)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed_x, (y_hard, y_hard[idx]), (y_soft, y_soft[idx]), lam


class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow


# ── Phase 1: Classification KD ───────────────────────────────────────────

class KDTrainDataset(torch.utils.data.Dataset):
    """Wraps a dataset with pre-computed teacher probs so shuffling stays aligned."""

    def __init__(self, base_ds, teacher_probs: np.ndarray):
        self.base_ds = base_ds
        self.teacher_probs = teacher_probs

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        return img, label, torch.tensor(self.teacher_probs[idx], dtype=torch.float32)


def _build_llrd_param_groups(model, base_lr, wd, n_unfreeze, llrd_factor,
                             backbone_lr_factor=0.01):
    """Build parameter groups with layer-wise learning rate decay.

    Last `n_unfreeze` blocks get `base_lr` with LLRD.
    Earlier blocks (and embed/pos) get `base_lr * backbone_lr_factor` (very small,
    like the ensemble backbone_lr_factor so pretrained weights are barely disturbed).
    """
    n_blocks = len(model.blocks)
    first_unfrozen = max(0, n_blocks - n_unfreeze)

    param_groups = []
    seen = set()

    for p in model.head.parameters():
        if p.requires_grad:
            param_groups.append({"params": [p], "lr": base_lr, "lr_scale": 1.0})
            seen.add(id(p))
    for p in model.norm.parameters():
        if p.requires_grad:
            param_groups.append({"params": [p], "lr": base_lr, "lr_scale": 1.0})
            seen.add(id(p))

    # Active blocks: last n_unfreeze, with LLRD
    for i in range(n_blocks - 1, first_unfrozen - 1, -1):
        dist = (n_blocks - 1) - i
        scale = llrd_factor ** dist
        block_lr = base_lr * scale
        block_params = [p for p in model.blocks[i].parameters()
                        if p.requires_grad and id(p) not in seen]
        if block_params:
            param_groups.append({
                "params": block_params, "lr": block_lr, "lr_scale": scale,
            })
            for p in block_params:
                seen.add(id(p))
            print(f"    block[{i}]: lr_scale={scale:.4f}  lr={block_lr:.6f}  (active)")

    # Deep backbone blocks: very low lr, like ensemble backbone_lr_factor
    backbone_lr = base_lr * backbone_lr_factor
    for i in range(first_unfrozen - 1, -1, -1):
        block_params = [p for p in model.blocks[i].parameters()
                        if p.requires_grad and id(p) not in seen]
        if block_params:
            param_groups.append({
                "params": block_params, "lr": backbone_lr, "lr_scale": backbone_lr_factor,
            })
            for p in block_params:
                seen.add(id(p))
            print(f"    block[{i}]: lr_scale={backbone_lr_factor:.4f}  lr={backbone_lr:.6f}  (backbone)")

    # Embed / pos embed: very low lr
    embed_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in seen]
    if embed_params:
        param_groups.append({
            "params": embed_params, "lr": backbone_lr, "lr_scale": backbone_lr_factor,
        })
        print(f"    embed/pos: lr_scale={backbone_lr_factor:.4f}  lr={backbone_lr:.6f}  "
              f"({len(embed_params)} tensors, backbone)")

    return param_groups


def run_phase1(model, targets, train_ds, val_ds, args, device):
    print("\n" + "=" * 60)
    print("Phase 1: Classification Knowledge Distillation")
    print(f"  Partial freeze: last {args.p1_unfreeze_blocks} blocks + norm + head")
    print(f"  LLRD factor: {args.llrd}  |  base_lr: {args.p1_lr}")
    print(f"  Augmentations: {args.p1_augmentations}")
    print(f"  Strategies: MixUp(alpha={args.mixup_alpha}) / CutMix(alpha={args.cutmix_alpha}), "
          f"DropPath=0.1, LabelSmooth={args.p1_label_smoothing}, "
          f"GradClip={args.grad_clip}, EMA={args.ema_decay}")
    print("=" * 60)

    n_blocks = len(model.blocks)
    n_unfreeze = min(args.p1_unfreeze_blocks, n_blocks)

    # Unfreeze all backbone params; EU head stays frozen.
    # Deep blocks get a very small lr (backbone_lr_factor) via the param groups.
    for p in model.parameters():
        p.requires_grad = True
    for p in model.eu_head_parameters:
        p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,}")
    print(f"  Backbone lr factor: {args.p1_backbone_lr_factor}  "
          f"(deep blocks lr = {args.p1_lr * args.p1_backbone_lr_factor:.2e})")

    kd_ds = KDTrainDataset(train_ds, targets["train_mean_probs"])
    train_loader = DataLoader(
        kd_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    param_groups = _build_llrd_param_groups(
        model, args.p1_lr, args.p1_wd, args.p1_unfreeze_blocks, args.llrd,
        backbone_lr_factor=args.p1_backbone_lr_factor)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.p1_wd)
    scaler = GradScaler()
    ema = EMA(model, decay=args.ema_decay)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    alpha = args.alpha
    tau = args.tau

    best_acc = 0.0
    for epoch in range(args.p1_epochs):
        cos_factor = 0.5 * (1 + np.cos(np.pi * epoch / args.p1_epochs))
        if epoch < args.warmup:
            cos_factor = (epoch + 1) / args.warmup
        for pg in optimizer.param_groups:
            pg["lr"] = args.p1_lr * pg["lr_scale"] * cos_factor
        head_lr = args.p1_lr * cos_factor

        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels, t_probs in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            t_probs = t_probs.to(device, non_blocking=True)

            use_cutmix = np.random.rand() < 0.5
            if use_cutmix:
                imgs, (labels_a, labels_b), (tp_a, tp_b), lam = cutmix_data(
                    imgs, labels, t_probs, alpha=args.cutmix_alpha)
            else:
                imgs, (labels_a, labels_b), (tp_a, tp_b), lam = mixup_data(
                    imgs, labels, t_probs, alpha=args.mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits, _ = model(imgs)
                ce_a = F.cross_entropy(logits, labels_a, label_smoothing=args.p1_label_smoothing)
                ce_b = F.cross_entropy(logits, labels_b, label_smoothing=args.p1_label_smoothing)
                ce_loss = lam * ce_a + (1 - lam) * ce_b

                log_s = F.log_softmax(logits / tau, dim=-1)
                t_log_a = F.log_softmax(torch.log(tp_a + 1e-8) / tau, dim=-1)
                t_log_b = F.log_softmax(torch.log(tp_b + 1e-8) / tau, dim=-1)
                kl_a = F.kl_div(log_s, t_log_a, reduction="batchmean", log_target=True)
                kl_b = F.kl_div(log_s, t_log_b, reduction="batchmean", log_target=True)
                kl_loss = (lam * kl_a + (1 - lam) * kl_b) * (tau ** 2)

                loss = (1 - alpha) * ce_loss + alpha * kl_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels_a).sum().item()
            total += imgs.size(0)

        train_acc = correct / total

        orig_state = copy.deepcopy(model.state_dict())
        ema.apply(model)
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast():
                    logits, _ = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total

        print(f"  P1 [{epoch+1:02d}/{args.p1_epochs}] lr={head_lr:.6f}  "
              f"loss={total_loss/total:.4f}  train_acc={train_acc:.4f}  "
              f"val_acc(ema)={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, os.path.join(args.save_dir, "student_phase1.pt"))
            print(f"    -> saved (best val_acc={best_acc:.4f})")

        model.load_state_dict(orig_state)

    print(f"  Phase 1 done. Best val_acc = {best_acc:.4f}")


# ── Phase 2: EU head regression ──────────────────────────────────────────

@torch.no_grad()
def _extract_eu_inputs(model, loader, device):
    """Extract [CLS_feat || softmax(logits)] from frozen backbone in batches."""
    model.eval()
    all_feats = []
    for imgs, *_ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast():
            feat = model._features(imgs)
            logits = model.head(feat)
            probs = F.softmax(logits, dim=-1)
        eu_in = torch.cat([feat, probs], dim=-1)
        all_feats.append(eu_in.cpu())
    return torch.cat(all_feats)


def _build_p2_loaders(targets, train_ds, val_ds, args):
    """Build data loaders for Phase 2 using a 50/25/25 clean/shifted/fake-OOD mix."""
    root = download_tiny_imagenet(args.data_dir)
    loaders = []
    rng = np.random.RandomState(2026)

    clean_eu = targets["train_EU"]
    n_clean = min(len(clean_eu), 30000)
    n_shifted = n_clean // 2
    n_ood = n_clean // 2
    indices = rng.choice(len(clean_eu), size=n_clean, replace=False)
    clean_sub = Subset(train_ds, indices.tolist())
    loaders.append(("clean", DataLoader(clean_sub, batch_size=64,
                                        num_workers=args.workers, pin_memory=True),
                     clean_eu[indices]))
    print(f"  Phase 2 data — clean: {n_clean}")

    per_corrupt = max(1, n_shifted // max(len(CORRUPTIONS), 1))
    corrupt_base = TinyImageNetDataset(root, split="val", transform=get_val_transform())
    for cname in CORRUPTIONS:
        key = f"corrupt_{cname}_EU"
        if key not in targets:
            continue
        c_eu = targets[key]
        n = min(per_corrupt, len(c_eu))
        if n < 1:
            continue
        c_imgs, _ = apply_corruption(
            corrupt_base,
            CORRUPTIONS[cname], max_samples=n,
        )
        c_ds = TensorDataset(c_imgs[:n])
        loaders.append((cname, DataLoader(c_ds, batch_size=64, num_workers=0),
                         c_eu[:n]))
        print(f"  Phase 2 data — {cname}: {n}")

    p2_mode = targets.get("p2_data_mode", np.array("fake_ood"))
    if hasattr(p2_mode, "flat"):
        p2_mode = str(p2_mode.flat[0]) if p2_mode.size else "fake_ood"
    else:
        p2_mode = str(p2_mode)

    if p2_mode == "fake_ood":
        def _scalar_target(name, default):
            arr = targets.get(name, np.array(default))
            if hasattr(arr, "flat") and arr.size:
                return float(arr.flat[0])
            return float(arr) if np.ndim(arr) == 0 else float(default)

        mixup_frac = _scalar_target("fake_ood_mixup_frac", 0.5)
        patchshuffle_frac = _scalar_target("fake_ood_patchshuffle_frac", 0.0)
        cutpaste_frac = _scalar_target("fake_ood_cutpaste_frac", 0.0)
        heavy_noise_frac = _scalar_target("fake_ood_heavy_noise_frac", 0.0)
        multi_corrupt_frac = _scalar_target("fake_ood_multi_corrupt_frac", 0.0)
        pixel_permute_frac = _scalar_target("fake_ood_pixel_permute_frac", 0.0)
        explicit_sum = mixup_frac + patchshuffle_frac + cutpaste_frac + heavy_noise_frac + multi_corrupt_frac + pixel_permute_frac
        masked_frac = _scalar_target(
            "fake_ood_masked_frac",
            max(0.0, 1.0 - explicit_sum),
        )

        fake_specs = {key: targets[key] for key in targets if key.startswith("fake_")}
        fake_datasets = build_fake_ood_datasets(train_ds, fake_specs)
        family_meta = [
            ("mixup", "fake mixup", "fake_mixup_EU", mixup_frac),
            ("patchshuffle", "fake patchshuffle", "fake_patchshuffle_EU", patchshuffle_frac),
            ("cutpaste", "fake cutpaste", "fake_cutpaste_EU", cutpaste_frac),
            ("masked", "fake masked", "fake_masked_EU", masked_frac),
            ("heavy_noise", "fake heavy_noise", "fake_heavy_noise_EU", heavy_noise_frac),
            ("multi_corrupt", "fake multi_corrupt", "fake_multi_corrupt_EU", multi_corrupt_frac),
            ("pixel_permute", "fake pixel_permute", "fake_pixel_permute_EU", pixel_permute_frac),
        ]
        active_families = [
            (family_name, label, eu_key, frac)
            for family_name, label, eu_key, frac in family_meta
            if family_name in fake_datasets and eu_key in targets and frac > 0 and len(targets[eu_key]) > 0
        ]

        total_frac = sum(frac for *_rest, frac in active_families)
        remaining = n_ood
        for idx, (family_name, label, eu_key, frac) in enumerate(active_families):
            family_eu = targets[eu_key]
            if idx == len(active_families) - 1:
                n_target = min(remaining, len(family_eu))
            else:
                weight = (frac / total_frac) if total_frac > 0 else (1.0 / max(1, len(active_families)))
                n_target = min(int(n_ood * weight), len(family_eu), remaining)

            if n_target < 1:
                continue

            idxs = rng.choice(len(family_eu), size=n_target, replace=False)
            loaders.append((
                label,
                DataLoader(
                    Subset(fake_datasets[family_name], idxs.tolist()),
                    batch_size=64,
                    num_workers=args.workers,
                    pin_memory=True,
                ),
                family_eu[idxs],
            ))
            remaining -= n_target
            print(f"  Phase 2 data — {label}: {n_target}")

    if not any(name.startswith("fake ") or name.startswith("OOD ") for name, *_rest in loaders):
        ood_loaders = get_ood_loaders(args.data_dir, batch_size=64, max_samples=max(1, n_ood // 2))
        fallback = [("SVHN", n_ood // 2), ("CIFAR-100", n_ood - (n_ood // 2))]
        for name, target_n in fallback:
            loader = ood_loaders.get(name)
            safe = name.replace("-", "_").replace(" ", "_")
            key = f"ood_{safe}_EU"
            if loader is None or key not in targets or target_n < 1:
                continue
            ood_eu = targets[key]
            ood_imgs = []
            for batch in loader:
                ood_imgs.append(batch[0])
            ood_t = torch.cat(ood_imgs)
            n = min(target_n, len(ood_t), len(ood_eu))
            ood_ds = TensorDataset(ood_t[:n])
            loaders.append((f"OOD {name}",
                            DataLoader(ood_ds, batch_size=64, num_workers=0),
                            ood_eu[:n]))
            print(f"  Phase 2 data — OOD {name}: {n}")
            del ood_t

    return loaders


def run_phase2(model, targets, train_ds, val_ds, args, device):
    print("\n" + "=" * 60)
    print("Phase 2: EU Head Regression")
    print("=" * 60)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.eu_head_parameters:
        p.requires_grad = True
    model.reinit_eu_head()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  EU head trainable params: {trainable:,}")

    p2_sources = _build_p2_loaders(targets, train_ds, val_ds, args)

    print("  Extracting backbone features for Phase 2 data ...")
    all_eu_in, all_eu_tgt = [], []
    for src_name, loader, eu_arr in p2_sources:
        feats = _extract_eu_inputs(model, loader, device)
        all_eu_in.append(feats)
        all_eu_tgt.append(torch.tensor(eu_arr[:len(feats)], dtype=torch.float32))
    all_eu_in = torch.cat(all_eu_in)
    all_eu_tgt = torch.cat(all_eu_tgt)
    print(f"  Cached {len(all_eu_tgt)} feature vectors ({all_eu_in.shape[-1]}-d)  "
          f"EU range=[{all_eu_tgt.min():.6f}, {all_eu_tgt.max():.6f}]")

    feat_ds = TensorDataset(all_eu_in, all_eu_tgt)
    if args.eu_sample_alpha > 0:
        sampling_weights = 1.0 + args.eu_sample_alpha * all_eu_tgt.clamp(min=0)
        sampler = torch.utils.data.WeightedRandomSampler(
            sampling_weights, len(sampling_weights), replacement=True)
        feat_loader = DataLoader(feat_ds, batch_size=256, sampler=sampler,
                                 num_workers=0, drop_last=True)
        print(f"  EU-weighted sampling enabled (alpha={args.eu_sample_alpha})")
    else:
        feat_loader = DataLoader(feat_ds, batch_size=256, shuffle=True,
                                 num_workers=0, drop_last=True)

    val_loader = DataLoader(
        TinyImageNetDataset(download_tiny_imagenet(args.data_dir), split="val",
                            transform=get_val_transform()),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    optimizer = torch.optim.Adam(model.eu_head_parameters, lr=args.p2_lr)

    best_corr = -1.0
    for epoch in range(args.p2_epochs):
        lr = args.p2_lr * 0.5 * (1 + np.cos(np.pi * epoch / args.p2_epochs))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        total_loss, total_mse, total_rank = 0.0, 0.0, 0.0
        count = 0
        for eu_in_batch, eu_tgt_batch in feat_loader:
            eu_in_batch = eu_in_batch.to(device, non_blocking=True)
            eu_tgt_batch = eu_tgt_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            eu_pred = model.eu_fc2(
                F.leaky_relu(model.eu_fc1(eu_in_batch), 0.1)).squeeze(-1)
            if args.asymmetric_weight > 1.0:
                mse = asymmetric_log1p_mse_loss(eu_pred, eu_tgt_batch,
                                                 under_weight=args.asymmetric_weight)
            else:
                mse = log1p_mse_loss(eu_pred, eu_tgt_batch)
            rank = pairwise_ranking_loss(eu_pred, eu_tgt_batch)
            loss = mse + args.rank_weight * rank

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * eu_in_batch.size(0)
            total_mse += mse.item() * eu_in_batch.size(0)
            total_rank += rank.item() * eu_in_batch.size(0)
            count += eu_in_batch.size(0)

        model.eval()
        val_eu_pred = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                with autocast():
                    _, eu_p = model(imgs)
                val_eu_pred.append(eu_p.cpu().numpy())

        val_eu_pred = np.concatenate(val_eu_pred)
        val_eu_tgt = targets["val_EU"][:len(val_eu_pred)]

        pearson = np.corrcoef(val_eu_pred, val_eu_tgt)[0, 1] if val_eu_pred.std() > 1e-8 else 0
        spearman = sp_stats.spearmanr(val_eu_pred, val_eu_tgt).correlation if val_eu_pred.std() > 1e-8 else 0

        print(f"  P2 [{epoch+1:02d}/{args.p2_epochs}] lr={lr:.6f}  "
              f"loss={total_loss/count:.6f}  mse={total_mse/count:.6f}  "
              f"rank={total_rank/count:.6f}  "
              f"pearson={pearson:.4f}  spearman={spearman:.4f}")

        if spearman > best_corr:
            best_corr = spearman
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "spearman": spearman,
                "pearson": pearson,
            }, os.path.join(args.save_dir, "student.pt"))
            print(f"    -> saved (best spearman={best_corr:.4f})")

    print(f"  Phase 2 done. Best spearman = {best_corr:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--p1_epochs", type=int, default=50)
    parser.add_argument("--p1_lr", type=float, default=1e-4)
    parser.add_argument("--p1_wd", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--p1_unfreeze_blocks", type=int, default=2,
                        help="Number of last transformer blocks to unfreeze in Phase 1")
    parser.add_argument("--p1_backbone_lr_factor", type=float, default=0.01,
                        help="LR multiplier for deep backbone blocks (like ensemble backbone_lr_factor)")
    parser.add_argument("--llrd", type=float, default=0.75,
                        help="Layer-wise learning rate decay factor")
    parser.add_argument("--p1_augmentations", type=str, default="randaugment+colorjitter+erasing",
                        help="Image-level augmentations for Phase 1 student KD")
    parser.add_argument("--p1_label_smoothing", type=float, default=0.05)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=2.0)

    parser.add_argument("--p2_epochs", type=int, default=150)
    parser.add_argument("--p2_lr", type=float, default=0.003)
    parser.add_argument("--rank_weight", type=float, default=1.0)
    parser.add_argument("--asymmetric_weight", type=float, default=2.0,
                        help="Under-prediction penalty multiplier for asymmetric loss (1.0=symmetric)")
    parser.add_argument("--eu_sample_alpha", type=float, default=10.0,
                        help="EU-weighted sampling strength (0=uniform, higher=more tail emphasis)")

    parser.add_argument("--phase2_only", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    root = download_tiny_imagenet(args.data_dir)
    val_tf = get_val_transform()

    targets = dict(np.load(os.path.join(args.save_dir, "teacher_targets.npz"),
                           allow_pickle=True))

    model = create_student().to(device)

    train_ds = TinyImageNetDataset(
        root,
        split="train",
        transform=get_train_transform(args.p1_augmentations),
    )
    val_ds = TinyImageNetDataset(root, split="val", transform=val_tf)

    if not args.phase2_only:
        run_phase1(model, targets, train_ds, val_ds, args, device)

    p1_path = os.path.join(args.save_dir, "student_phase1.pt")
    if os.path.isfile(p1_path):
        ckpt = torch.load(p1_path, map_location=device, weights_only=True)
        src = ckpt.get("ema_state_dict", ckpt["model_state_dict"])
        state = {k: v for k, v in src.items() if not k.startswith("eu_")}
        model.load_state_dict(state, strict=False)
        print(f"  Loaded Phase 1 EMA checkpoint (val_acc={ckpt.get('val_acc', '?')})")
    else:
        print("  WARNING: No Phase 1 checkpoint found. Using current weights.")

    train_ds_clean = TinyImageNetDataset(root, split="train", transform=val_tf)
    run_phase2(model, targets, train_ds_clean, val_ds, args, device)


if __name__ == "__main__":
    main()
