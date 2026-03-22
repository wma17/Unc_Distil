"""
Two-phase distillation for VOC 2012: SegFormer ensemble -> SegFormer student.

Phase 1 — Segmentation KD (full student):
    L = (1-α) CE(y, z_S) + α τ² KL(softmax(z_T/τ) || softmax(z_S/τ))
    Applied per-pixel, ignore boundary (label=255).

Phase 2 — Spatial EU head (freeze backbone + decode head, train EU head):
    L = log1p_MSE(EU_S, EU_T) + β · ranking  (per-pixel)
    Three-tier curriculum: clean + corrupted + synthetic OOD

Usage:
    python cache_ensemble_targets.py --save_dir ./checkpoints  # prerequisite
    python distill.py --save_dir ./checkpoints --gpu 0
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from models import create_student, NUM_CLASSES
from data import (VOCSegDataset, SegTransformTrain, SegTransformVal,
                  EUSegDataset, CorruptedEUSegDataset, MixupEUSegDataset,
                  BlockMaskedEUSegDataset, CORRUPTIONS, IGNORE_INDEX)


EPS = 1e-8


# ---------------------------------------------------------------------------
# Correlation
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
# Phase 1
# ---------------------------------------------------------------------------

def phase1_loss(student_logits, masks, teacher_logits, alpha, tau, ignore_index=IGNORE_INDEX):
    """Per-pixel KD loss."""
    # Both logits: (B, C, H, W) — ensure same spatial size
    if student_logits.shape[2:] != masks.shape[1:]:
        student_logits = F.interpolate(student_logits, size=masks.shape[1:],
                                        mode="bilinear", align_corners=False)
    if teacher_logits.shape[2:] != masks.shape[1:]:
        teacher_logits = F.interpolate(teacher_logits, size=masks.shape[1:],
                                        mode="bilinear", align_corners=False)

    loss_ce = F.cross_entropy(student_logits, masks, ignore_index=ignore_index)

    # KL per-pixel (flatten spatial dims)
    B, C, H, W = student_logits.shape
    valid = (masks != ignore_index).view(B, 1, H, W).expand_as(student_logits)

    t_soft = F.softmax(teacher_logits / tau, dim=1)
    s_log_soft = F.log_softmax(student_logits / tau, dim=1)
    kl = F.kl_div(s_log_soft, t_soft, reduction="none")
    kl = (kl * valid.float()).sum() / max(1, valid.float().sum()) * C * (tau ** 2)

    loss = (1 - alpha) * loss_ce + alpha * kl
    return loss, loss_ce.detach().item(), kl.detach().item()


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def log1p_mse_loss_spatial(pred, target):
    """MSE in log(1+x) space for spatial EU maps."""
    return F.mse_loss(torch.log1p(pred), torch.log1p(target))


def pairwise_ranking_loss_spatial(pred, target, n_pairs=512, margin=0.05):
    """Pairwise ranking over randomly sampled pixel pairs within a batch."""
    # Flatten all pixels
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    n = pred_flat.size(0)
    if n < 2:
        return pred_flat.new_tensor(0.0)

    idx_i = torch.randint(0, n, (n_pairs,), device=pred.device)
    idx_j = torch.randint(0, n, (n_pairs,), device=pred.device)
    t_i, t_j = target_flat[idx_i], target_flat[idx_j]
    mask = t_i > t_j + EPS
    if mask.sum() < 1:
        return pred_flat.new_tensor(0.0)
    p_i, p_j = pred_flat[idx_i][mask], pred_flat[idx_j][mask]
    return F.margin_ranking_loss(
        p_i, p_j, torch.ones(mask.sum(), device=pred.device), margin=margin)


def train_phase2_epoch(model, loader, optimizer, rank_weight, device, eu_map_size):
    model.eval()
    # Keep EU head in train mode
    model.eu_conv1.train()
    model.eu_bn.train()
    model.eu_conv2.train()

    sum_mse, sum_rank, total = 0, 0, 0
    for imgs, eu_targets in loader:
        imgs = imgs.to(device)
        eu_targets = eu_targets.to(device)

        optimizer.zero_grad()
        _logits, eu_pred = model(imgs, return_eu=True)

        # Resize EU prediction to match target
        if eu_pred.shape[1:] != eu_targets.shape[1:]:
            eu_pred = F.interpolate(eu_pred.unsqueeze(1),
                                     size=eu_targets.shape[1:],
                                     mode="bilinear", align_corners=False).squeeze(1)

        l_mse = log1p_mse_loss_spatial(eu_pred, eu_targets)
        l_rank = pairwise_ranking_loss_spatial(eu_pred, eu_targets)
        loss = l_mse + rank_weight * l_rank
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        sum_mse += l_mse.detach().item() * bs
        sum_rank += l_rank.detach().item() * bs
        total += bs
    return sum_mse / total, sum_rank / total


@torch.no_grad()
def eval_phase2(model, loader, device, eu_map_size):
    model.eval()
    all_pred, all_target = [], []
    for imgs, eu_targets in loader:
        imgs = imgs.to(device)
        _logits, eu_pred = model(imgs, return_eu=True)
        if eu_pred.shape[1:] != eu_targets.shape[1:]:
            eu_pred = F.interpolate(eu_pred.unsqueeze(1),
                                     size=eu_targets.shape[1:],
                                     mode="bilinear", align_corners=False).squeeze(1)
        all_pred.append(eu_pred.cpu().reshape(-1))
        all_target.append(eu_targets.reshape(-1))

    pred = torch.cat(all_pred)
    target = torch.cat(all_target).float()
    loss = F.mse_loss(torch.log1p(pred), torch.log1p(target)).item()
    r_p = pearson_corr(pred, target)
    r_s = spearman_corr(pred, target)
    return loss, r_p, r_s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Two-phase distillation for VOC")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--p1_iterations", type=int, default=40000)
    parser.add_argument("--p1_lr", type=float, default=6e-5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=4.0)

    parser.add_argument("--p2_iterations", type=int, default=20000)
    parser.add_argument("--p2_lr", type=float, default=0.001)
    parser.add_argument("--rank_weight", type=float, default=1.0)

    parser.add_argument("--phase2_only", action="store_true")
    parser.add_argument("--curriculum", type=str, default="A3",
                        choices=["A1", "A2", "A3"])
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    targets_path = os.path.join(args.save_dir, "teacher_targets.npz")
    if not os.path.exists(targets_path):
        raise FileNotFoundError(f"{targets_path} not found.")
    data = np.load(targets_path, allow_pickle=True)
    eu_map_size = int(data["eu_map_size"]) if "eu_map_size" in data else 128

    model = create_student(num_classes=NUM_CLASSES).to(device)
    p1_path = os.path.join(args.save_dir, "student_phase1.pt")
    final_path = os.path.join(args.save_dir, "student.pt")

    # ==================================================================
    # Phase 1: Segmentation KD
    # ==================================================================
    if not args.phase2_only:
        print(f"\n{'='*70}")
        print(f"  Phase 1: Segmentation distillation (α={args.alpha}, τ={args.tau})")
        print(f"  {args.p1_iterations} iterations, lr={args.p1_lr}")
        print(f"{'='*70}\n")

        # Load a teacher for online KD (member 0)
        from models import create_teacher
        import json
        teacher = create_teacher(num_classes=NUM_CLASSES).to(device)
        teacher_ckpt = torch.load(os.path.join(args.save_dir, "member_0.pt"),
                                   map_location=device, weights_only=False)
        teacher.load_state_dict(teacher_ckpt["model_state_dict"], strict=False)
        teacher.eval()

        train_transform = SegTransformTrain(crop_size=512)
        val_transform = SegTransformVal(crop_size=512)
        train_ds = VOCSegDataset(args.data_dir, split="train", transform=train_transform)
        val_ds = VOCSegDataset(args.data_dir, split="val", transform=val_transform,
                                use_sbd=False)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True,
                                  drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

        optimizer = optim.AdamW(model.parameters(), lr=args.p1_lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0, (1 - step / args.p1_iterations) ** 0.9))

        best_miou = 0.0
        global_step = 0
        epoch = 0

        while global_step < args.p1_iterations:
            epoch += 1
            model.train()
            epoch_loss = 0
            n_batches = 0

            for imgs, masks in train_loader:
                if global_step >= args.p1_iterations:
                    break
                imgs, masks = imgs.to(device), masks.to(device)

                with torch.no_grad():
                    teacher_logits = teacher(imgs)

                optimizer.zero_grad()
                student_logits, _ = model(imgs, return_eu=False)
                loss, ce, kl = phase1_loss(student_logits, masks, teacher_logits,
                                            args.alpha, args.tau)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1

                epoch_loss += loss.item()
                n_batches += 1

            if epoch % 5 == 0 or global_step >= args.p1_iterations:
                from train_ensemble import compute_miou
                model.eval()
                val_mious = []
                with torch.no_grad():
                    for imgs, masks in val_loader:
                        imgs, masks = imgs.to(device), masks.to(device)
                        logits, _ = model(imgs, return_eu=False)
                        logits_up = F.interpolate(logits, size=masks.shape[1:],
                                                   mode="bilinear", align_corners=False)
                        preds = logits_up.argmax(1)
                        for b in range(preds.shape[0]):
                            val_mious.append(compute_miou(
                                preds[b].cpu().numpy(), masks[b].cpu().numpy()))
                val_miou = np.mean(val_mious)
                print(f"  P1 Epoch {epoch} (step {global_step}) | "
                      f"Loss {epoch_loss/max(1,n_batches):.4f} | "
                      f"Val mIoU {val_miou:.4f}")

                if val_miou > best_miou:
                    best_miou = val_miou
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "step": global_step, "val_miou": val_miou, "phase": 1,
                    }, p1_path)

        del teacher
        torch.cuda.empty_cache()
        print(f"\n  Phase 1 best mIoU: {best_miou:.4f}  ->  {p1_path}")
    else:
        print(f"\nSkipping Phase 1, loading {p1_path}")

    # Load best Phase 1 model
    ckpt = torch.load(p1_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  Loaded Phase 1 (mIoU={ckpt.get('val_miou', '?')})")

    # ==================================================================
    # Phase 2: EU head training
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  Phase 2: Spatial EU head (backbone + decode head frozen)")
    print(f"  Curriculum: {args.curriculum}  |  {args.p2_iterations} iters")
    print(f"{'='*70}\n")

    # Freeze everything except EU head
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("eu_")
    model.reinit_eu_head()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} (EU head only)")

    # Build Phase 2 datasets
    val_transform = SegTransformVal(crop_size=512)
    train_ds = VOCSegDataset(args.data_dir, split="train", transform=val_transform)

    # Load all train images into memory for Phase 2
    print("  Loading train images for Phase 2...")
    train_imgs = []
    for i in range(len(train_ds)):
        img, _ = train_ds[i]
        train_imgs.append(img)
    train_imgs_tensor = torch.stack(train_imgs)

    n_train = len(train_imgs)
    datasets_list = []

    # Tier 1: clean
    clean_ds = EUSegDataset(train_imgs_tensor, data["train_eu"][:n_train])
    datasets_list.append(clean_ds)
    print(f"  Tier 1 (clean): {len(clean_ds)}")

    # Tier 2: corrupted
    if args.curriculum in ("A2", "A3"):
        n_corrupt = n_train // 4
        n_per = n_corrupt // len(CORRUPTIONS)
        for cname, cfn in CORRUPTIONS.items():
            key = f"corrupt_{cname}_eu"
            if key not in data:
                continue
            c_ds = CorruptedEUSegDataset(
                train_imgs_tensor[:n_per], data[key][:n_per], cfn)
            datasets_list.append(c_ds)
        print(f"  Tier 2 (corrupted): ~{n_corrupt}")

    # Tier 3: synthetic OOD
    if args.curriculum == "A3":
        n_ood = n_train // 4
        n_mixup = n_ood // 2
        n_masked = n_ood - n_mixup
        mixup_ds = MixupEUSegDataset(train_imgs_tensor, data["train_eu"][:n_train],
                                      n_mixup, seed=2027)
        masked_ds = BlockMaskedEUSegDataset(train_imgs_tensor, data["train_eu"][:n_train],
                                             n_masked, seed=2028)
        datasets_list.extend([mixup_ds, masked_ds])
        print(f"  Tier 3 (OOD): {n_mixup} mixup + {n_masked} masked")

    p2_train_ds = ConcatDataset(datasets_list)
    print(f"  Phase 2 total: {len(p2_train_ds)}")

    # Val EU dataset
    val_ds_p2 = VOCSegDataset(args.data_dir, split="val", transform=val_transform,
                               use_sbd=False)
    val_imgs = [val_ds_p2[i][0] for i in range(len(val_ds_p2))]
    val_imgs_tensor = torch.stack(val_imgs)
    p2_val_ds = EUSegDataset(val_imgs_tensor, data["val_eu"][:len(val_imgs)])

    p2_train_loader = DataLoader(p2_train_ds, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=True)
    p2_val_loader = DataLoader(p2_val_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

    optimizer = optim.Adam(model.eu_head_parameters, lr=args.p2_lr)
    max_epochs = max(1, args.p2_iterations // len(p2_train_loader) + 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_spearman = -1.0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_mse, tr_rank = train_phase2_epoch(
            model, p2_train_loader, optimizer, args.rank_weight, device, eu_map_size)
        te_loss, r_pear, r_spear = eval_phase2(model, p2_val_loader, device, eu_map_size)
        scheduler.step()
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            print(f"  P2 Epoch {epoch:3d}/{max_epochs} | "
                  f"Train mse={tr_mse:.6f} rank={tr_rank:.6f} | "
                  f"Val mse={te_loss:.6f} | "
                  f"Pearson={r_pear:.4f} Spearman={r_spear:.4f} | {elapsed:.1f}s")

        if r_spear > best_spearman:
            best_spearman = r_spear
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_miou": ckpt.get("val_miou", 0),
                "eu_pearson": r_pear, "eu_spearman": r_spear, "phase": 2,
            }, final_path)

    print(f"\n  Phase 2 best Spearman: {best_spearman:.4f}")
    print(f"  Final student saved to: {final_path}")


if __name__ == "__main__":
    main()
