"""
Evaluate BNN baselines for TinyImageNet and compare with the distilled student.

Metrics aligned with evaluate_student.py:
  - Accuracy, ECE-15, NLL, Brier Score
  - EU Pearson / Spearman (vs teacher ensemble EU)
  - OOD AUROC (seen + unseen)
  - AURC + selective prediction
  - Inference throughput

Usage:
    python evaluate_baselines.py --save_dir ./checkpoints --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.amp import autocast
from torch.utils.data import DataLoader

from data import (TinyImageNetDataset, get_val_transform, get_ood_loaders,
                  download_tiny_imagenet)
from models import (create_ensemble_member, create_student,
                    load_saved_member_state, EMBED_DIM, NUM_CLASSES)
from train_baselines import (EDLHead, load_member0, enable_attention_dropout,
                              get_val_loader, EPS)
from evaluate_student import (compute_ece, compute_nll, compute_brier,
                               compute_aurc, measure_throughput,
                               predict_student, predict_member,
                               auroc_safe, entropy, ood_type,
                               SEEN_OOD, UNSEEN_OOD)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def mutual_information(probs_stack):
    """EU = H[E[p]] - E[H[p]] from (T, N, K) array."""
    mean_p = probs_stack.mean(0)
    h_mean = -(mean_p * np.log(mean_p + EPS)).sum(-1)
    h_each = -(probs_stack * np.log(probs_stack + EPS)).sum(-1)
    return np.maximum(h_mean - h_each.mean(0), 0.0), mean_p


def pearson_corr(a, b):
    a_t = torch.from_numpy(np.asarray(a)).float()
    b_t = torch.from_numpy(np.asarray(b)).float()
    return torch.corrcoef(torch.stack([a_t, b_t]))[0, 1].item()


def spearman_corr(a, b):
    def _rank(x):
        o = x.argsort(); r = torch.empty_like(x)
        r[o] = torch.arange(len(x), dtype=x.dtype)
        return r
    a_t = torch.from_numpy(np.asarray(a)).float()
    b_t = torch.from_numpy(np.asarray(b)).float()
    return torch.corrcoef(torch.stack([_rank(a_t), _rank(b_t)]))[0, 1].item()


# ---------------------------------------------------------------------------
# Baseline inference (val set)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_backbone(backbone, imgs, device):
    """Extract CLS features from DeiT backbone (eval, autocast)."""
    imgs = imgs.to(device)
    with autocast("cuda"):
        x_e = backbone.patch_embed(imgs)
        cls = backbone.cls_token.expand(x_e.size(0), -1, -1)
        x_e = torch.cat([cls, x_e], dim=1)
        x_e = backbone.pos_drop(x_e + backbone.pos_embed)
        x_e = backbone.blocks(x_e)
        x_e = backbone.norm(x_e)
        feat = x_e[:, 0].float()
    return feat


def predict_mc_dropout(mc_model, loader, device, T=16):
    mc_model.to(device).train()  # activates Dropout layers
    all_imgs, all_labels = [], []
    for imgs, labels in loader:
        all_imgs.append(imgs); all_labels.append(labels.numpy())
    all_imgs = torch.cat(all_imgs)

    sample_probs = []
    bs = 32
    for _ in range(T):
        chunks = []
        for i in range(0, len(all_imgs), bs):
            batch = all_imgs[i:i+bs].to(device)
            with autocast("cuda"):
                logits = mc_model(batch)
            chunks.append(F.softmax(logits.float(), -1).cpu().numpy())
        sample_probs.append(np.concatenate(chunks))

    eu, mean_probs = mutual_information(np.stack(sample_probs))
    return eu, mean_probs, np.concatenate(all_labels)


def predict_edl(backbone, edl_head, loader, device):
    backbone.to(device).eval(); edl_head.to(device).eval()
    all_eu_epi, all_eu_ale, all_probs, all_labels = [], [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            feat = _run_backbone(backbone, imgs, device)
            alpha = edl_head(feat)
            S     = alpha.sum(-1, keepdim=True)
            p_hat = (alpha / S).cpu()
            u_ale = (torch.digamma(S + 1)
                     - (alpha / S * torch.digamma(alpha + 1)).sum(-1, keepdim=True)
                     ).cpu().squeeze(-1)
            h_p   = -(p_hat * torch.log(p_hat + EPS)).sum(-1)
            u_epi = (h_p - u_ale).clamp(min=0.0)
            all_eu_epi.append(u_epi.numpy())
            all_eu_ale.append(u_ale.numpy())
            all_probs.append(p_hat.numpy())
            all_labels.append(labels.numpy())
    return (np.concatenate(all_eu_epi), np.concatenate(all_eu_ale),
            np.concatenate(all_probs), np.concatenate(all_labels))


def predict_llla(backbone, la, loader, device, T=16):
    all_imgs, all_labels = [], []
    for imgs, labels in loader:
        all_imgs.append(imgs); all_labels.append(labels.numpy())
    all_imgs = torch.cat(all_imgs)

    sample_probs = []
    bs = 32
    for i in range(0, len(all_imgs), bs):
        batch = all_imgs[i:i+bs].to(device)
        logit_samps = la.predictive_samples(batch, pred_type="glm", n_samples=T)
        sample_probs.append(F.softmax(logit_samps.float(), -1).cpu().numpy())
    probs_stack = np.concatenate(sample_probs, axis=1)
    eu, mean_probs = mutual_information(probs_stack)
    return eu, mean_probs, np.concatenate(all_labels)


def predict_sgld(backbone, samples, loader, device):
    backbone.to(device).eval()
    all_imgs, all_labels = [], []
    for imgs, labels in loader:
        all_imgs.append(imgs); all_labels.append(labels.numpy())
    all_imgs = torch.cat(all_imgs)

    # Cache CLS features once
    bs = 32
    feat_chunks = []
    with torch.no_grad():
        for i in range(0, len(all_imgs), bs):
            feat_chunks.append(
                _run_backbone(backbone, all_imgs[i:i+bs], device).cpu())
    feats = torch.cat(feat_chunks)  # (N, 384)

    sample_probs = []
    for s in samples:
        w = s["weight"].to(device); b = s["bias"].to(device)
        chunks = []
        for i in range(0, len(feats), bs):
            logits = feats[i:i+bs].to(device) @ w.T + b
            chunks.append(F.softmax(logits, -1).cpu().numpy())
        sample_probs.append(np.concatenate(chunks))

    eu, mean_probs = mutual_information(np.stack(sample_probs))
    return eu, mean_probs, np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Helper: reload backbone from saved member state
# ---------------------------------------------------------------------------

def _reload_backbone(save_dir, baseline_dir, state_key):
    cfg_path = os.path.join(save_dir, "ensemble_configs.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            configs = json.load(f)
        cfg = configs[0]
        model = create_ensemble_member(
            rank=cfg.get("rank", 8), alpha=cfg.get("alpha", None),
            lora_dropout=cfg.get("lora_dropout", 0.0),
            targets=cfg.get("targets", "qkv+proj"),
            unfreeze_blocks=cfg.get("unfreeze_blocks", 0), pretrained=True)
    else:
        model = create_ensemble_member(pretrained=True)

    ckpt  = torch.load(os.path.join(baseline_dir, state_key), map_location="cpu")
    state = ckpt.get("backbone_member_state", ckpt.get("member_state", ckpt))
    load_saved_member_state(model, {"member_state": state}, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_aurc(name, errors, scores):
    a, _, gap, a90, a80 = compute_aurc(errors, np.asarray(scores))
    print(f"  {name:<35} {a:>10.6f} {gap:>12.6f} {a90:>8.4f} {a80:>8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir",    default="./checkpoints")
    p.add_argument("--data_dir",    default="/home/maw6/maw6/unc_regression/data")
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--T",           type=int, default=16)
    p.add_argument("--methods", nargs="+",
                   default=["mc_dropout", "edl", "llla", "sgld"],
                   choices=["mc_dropout", "edl", "llla", "sgld"])
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    baseline_dir = os.path.join(args.save_dir, "baselines")
    root = download_tiny_imagenet(args.data_dir)
    val_tf = get_val_transform()

    npz = dict(np.load(os.path.join(args.save_dir, "teacher_targets.npz"),
                       allow_pickle=True))
    teacher_eu_np  = npz["val_eu"]        # (N,)
    true_labels    = npz["val_labels"].astype(int)
    teacher_probs  = npz["val_mean_probs"] # (N, 200)

    val_loader = get_val_loader(root, args.batch_size, args.num_workers)
    img_shape  = (3, 224, 224)

    # -----------------------------------------------------------------------
    # Load baselines
    # -----------------------------------------------------------------------
    baselines = {}  # name -> (eu_np, probs_np, labels_np)

    # MC Dropout
    mc_path = os.path.join(baseline_dir, "mc_dropout.pt")
    if "mc_dropout" in args.methods and os.path.isfile(mc_path):
        print("\n[MC Dropout]")
        ckpt = torch.load(mc_path, map_location="cpu")
        mc_m = load_member0(args.save_dir, device)
        mc_m  = enable_attention_dropout(mc_m, ckpt.get("dropout_p", 0.1))
        # Reload saved fine-tuned state
        load_saved_member_state(mc_m, {"member_state": ckpt["member_state"]}, strict=False)
        eu, probs, labels = predict_mc_dropout(mc_m, val_loader, device, T=args.T)
        baselines["MC Dropout"] = (eu, probs, labels)
        print(f"  EU={eu.mean():.4f}  acc={100*(probs.argmax(-1)==labels).mean():.2f}%")

    # EDL
    edl_path = os.path.join(baseline_dir, "edl_head.pt")
    if "edl" in args.methods and os.path.isfile(edl_path):
        print("\n[EDL]")
        ckpt = torch.load(edl_path, map_location="cpu")
        backbone = _reload_backbone(args.save_dir, baseline_dir, "edl_head.pt")
        load_saved_member_state(backbone,
                                {"member_state": ckpt["backbone_member_state"]},
                                strict=False)
        head = EDLHead(feat_dim=EMBED_DIM, num_classes=NUM_CLASSES)
        head.load_state_dict(ckpt["edl_head_state_dict"])
        eu_epi, eu_ale, probs, labels = predict_edl(backbone, head, val_loader, device)
        baselines["EDL"] = (eu_epi, probs, labels)
        print(f"  EU_epi={eu_epi.mean():.4f}  EU_ale={eu_ale.mean():.4f}"
              f"  acc={100*(probs.argmax(-1)==labels).mean():.2f}%")

    # LLLA
    la_obj = None
    llla_path    = os.path.join(baseline_dir, "llla.pt")
    llla_bb_path = os.path.join(baseline_dir, "llla_backbone.pt")
    if "llla" in args.methods and os.path.isfile(llla_path):
        print("\n[LLLA]")
        try:
            from laplace import Laplace

            bb_llla_ckpt = torch.load(llla_bb_path, map_location="cpu")
            backbone_llla = _reload_backbone(args.save_dir, baseline_dir, "llla_backbone.pt")
            load_saved_member_state(backbone_llla,
                                    {"member_state": bb_llla_ckpt["backbone_member_state"]},
                                    strict=False)

            class DeiTWithHead(nn.Module):
                def __init__(self, b):
                    super().__init__(); self.b = b
                def forward(self, x):
                    x_e = self.b.patch_embed(x)
                    cls = self.b.cls_token.expand(x_e.size(0), -1, -1)
                    x_e = torch.cat([cls, x_e], dim=1)
                    x_e = self.b.pos_drop(x_e + self.b.pos_embed)
                    x_e = self.b.blocks(x_e); x_e = self.b.norm(x_e)
                    return self.b.head(x_e[:, 0])

            wrapped = DeiTWithHead(backbone_llla)
            la_obj = Laplace(wrapped, "classification",
                             subset_of_weights="last_layer",
                             hessian_structure="kron")
            la_obj.load_state_dict(torch.load(llla_path, map_location="cpu"))
            la_obj.model.to(device)
            eu, probs, labels = predict_llla(backbone_llla, la_obj, val_loader, device, T=args.T)
            baselines["LLLA (KFAC)"] = (eu, probs, labels)
            print(f"  EU={eu.mean():.4f}  acc={100*(probs.argmax(-1)==labels).mean():.2f}%")
        except ImportError:
            print("  laplace-torch not installed.")
        except Exception as e:
            print(f"  Error: {e}")

    # SGLD
    sgld_path = os.path.join(baseline_dir, "sgld_samples.pt")
    sgld_backbone = None; sgld_samples = None
    if "sgld" in args.methods and os.path.isfile(sgld_path):
        print("\n[SGLD]")
        ckpt = torch.load(sgld_path, map_location="cpu")
        sgld_backbone = _reload_backbone(args.save_dir, baseline_dir, "sgld_samples.pt")
        load_saved_member_state(sgld_backbone,
                                {"member_state": ckpt["backbone_member_state"]},
                                strict=False)
        sgld_samples = ckpt["samples"]
        eu, probs, labels = predict_sgld(sgld_backbone, sgld_samples, val_loader, device)
        baselines["SGLD"] = (eu, probs, labels)
        print(f"  EU={eu.mean():.4f}  acc={100*(probs.argmax(-1)==labels).mean():.2f}%")

    # Student for comparison
    student_eu_np = None; student_probs_np = None
    student_path = os.path.join(args.save_dir, "student.pt")
    if os.path.isfile(student_path):
        print("\n[Student for comparison]")
        stu = create_student().to(device)
        stu_ckpt = torch.load(student_path, map_location=device, weights_only=False)
        stu.load_state_dict(stu_ckpt["model_state_dict"])
        stu.eval()
        s_probs, s_eu, _ = predict_student(stu, val_loader, device)
        student_eu_np    = s_eu
        student_probs_np = s_probs

    if not baselines:
        print("\nNo baselines found. Run train_baselines.py first.")
        return

    tea_eu_np = teacher_eu_np

    # -----------------------------------------------------------------------
    # Accuracy & Calibration table
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  TinyImageNet Baseline Evaluation Summary")
    print(f"{'='*80}")
    print(f"\n  Accuracy & Calibration (TinyImageNet val set)")
    print(f"  {'Method':<30} {'Acc%':>8} {'ECE-15':>8} {'NLL':>8} {'Brier':>8} "
          f"{'EU-Pear':>8} {'EU-Spear':>9}")
    print(f"  {'-'*80}")

    tea_acc   = 100.0 * (teacher_probs.argmax(-1) == true_labels).mean()
    tea_ece   = compute_ece(teacher_probs, true_labels)
    tea_nll   = compute_nll(teacher_probs, true_labels)
    tea_brier = compute_brier(teacher_probs, true_labels)
    print(f"  {'Teacher (ensemble)':<30} {tea_acc:>8.2f} {tea_ece:>8.4f} "
          f"{tea_nll:>8.4f} {tea_brier:>8.4f} {'—':>8} {'—':>9}")

    if student_probs_np is not None:
        s_acc   = 100.0 * (student_probs_np.argmax(-1) == true_labels).mean()
        s_ece   = compute_ece(student_probs_np, true_labels)
        s_nll   = compute_nll(student_probs_np, true_labels)
        s_brier = compute_brier(student_probs_np, true_labels)
        rp      = pearson_corr(student_eu_np, tea_eu_np)
        rs      = spearman_corr(student_eu_np, tea_eu_np)
        print(f"  {'Student (distilled)':<30} {s_acc:>8.2f} {s_ece:>8.4f} "
              f"{s_nll:>8.4f} {s_brier:>8.4f} {rp:>8.4f} {rs:>9.4f}")

    for name, (eu_np, probs_np, _) in baselines.items():
        acc   = 100.0 * (probs_np.argmax(-1) == true_labels).mean()
        ece   = compute_ece(probs_np, true_labels)
        nll   = compute_nll(probs_np, true_labels)
        brier = compute_brier(probs_np, true_labels)
        rp    = pearson_corr(eu_np, tea_eu_np)
        rs    = spearman_corr(eu_np, tea_eu_np)
        print(f"  {name:<30} {acc:>8.2f} {ece:>8.4f} {nll:>8.4f} "
              f"{brier:>8.4f} {rp:>8.4f} {rs:>9.4f}")

    # -----------------------------------------------------------------------
    # OOD Detection AUROC
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  OOD Detection AUROC (ID = TinyImageNet val)")
    print(f"{'='*80}")

    ood_loaders = get_ood_loaders(args.data_dir, args.batch_size)
    id_tea_eu  = teacher_eu_np
    id_stu_eu  = student_eu_np
    if id_stu_eu is not None:
        id_stu_ent = entropy(student_probs_np)

    col_w = 11
    method_names = ["Teacher EU"]
    if student_probs_np is not None:
        method_names += ["Student EU", "Stu Ent"]
    method_names += list(baselines.keys())
    id_eu_map = {"Teacher EU": id_tea_eu}
    if student_probs_np is not None:
        id_eu_map["Student EU"] = id_stu_eu
        id_eu_map["Stu Ent"]    = id_stu_ent
    for name, (eu, _, _) in baselines.items():
        id_eu_map[name] = eu

    hdr = f"  {'OOD Dataset':<22} {'Type':<8}"
    for m in method_names:
        hdr += f" {m[:col_w]:>{col_w}}"
    print(hdr); print(f"  {'-'*80}")

    for ood_name, ood_loader in ood_loaders.items():
        cache_key = f"ood_{ood_name}_eu"
        ood_type_label = ood_type(ood_name)

        try:
            # Teacher OOD EU from cache if available
            if cache_key in npz:
                ood_tea_eu = npz[cache_key]
            else:
                ood_tea_eu = None

            row = f"  {ood_name:<22} {ood_type_label:<8}"

            # Teacher EU
            if ood_tea_eu is not None:
                row += f" {auroc_safe(id_tea_eu, ood_tea_eu[:len(ood_tea_eu)]):{col_w}.4f}"
            else:
                row += f" {'N/A':>{col_w}}"

            # Student EU + Entropy
            if student_probs_np is not None:
                try:
                    ood_probs_s, ood_eu_s, _ = predict_student(stu, ood_loader, device)
                    row += f" {auroc_safe(id_stu_eu, ood_eu_s):{col_w}.4f}"
                    row += f" {auroc_safe(id_stu_ent, entropy(ood_probs_s)):{col_w}.4f}"
                except Exception:
                    row += f" {'N/A':>{col_w}} {'N/A':>{col_w}}"

            # Baselines on OOD
            for bl_name, (id_eu, _, _) in baselines.items():
                try:
                    if bl_name == "MC Dropout":
                        mc_m2 = load_member0(args.save_dir, device)
                        ckpt2 = torch.load(mc_path, map_location="cpu")
                        mc_m2 = enable_attention_dropout(mc_m2, ckpt2.get("dropout_p", 0.1))
                        load_saved_member_state(mc_m2, {"member_state": ckpt2["member_state"]},
                                                strict=False)
                        ood_eu_bl, _, _ = predict_mc_dropout(mc_m2, ood_loader, device, T=args.T)
                    elif bl_name == "EDL":
                        ckpt2 = torch.load(edl_path, map_location="cpu")
                        bb2   = _reload_backbone(args.save_dir, baseline_dir, "edl_head.pt")
                        load_saved_member_state(bb2, {"member_state": ckpt2["backbone_member_state"]},
                                                strict=False)
                        hd2   = EDLHead(feat_dim=EMBED_DIM, num_classes=NUM_CLASSES)
                        hd2.load_state_dict(ckpt2["edl_head_state_dict"])
                        ood_eu_bl, _, _, _ = predict_edl(bb2, hd2, ood_loader, device)
                    elif bl_name == "LLLA (KFAC)" and la_obj is not None:
                        bb3 = _reload_backbone(args.save_dir, baseline_dir, "llla_backbone.pt")
                        load_saved_member_state(
                            bb3, {"member_state": torch.load(llla_bb_path,
                                  map_location="cpu")["backbone_member_state"]}, strict=False)
                        ood_eu_bl, _, _ = predict_llla(bb3, la_obj, ood_loader, device, T=args.T)
                    elif bl_name == "SGLD" and sgld_backbone is not None:
                        ood_eu_bl, _, _ = predict_sgld(sgld_backbone, sgld_samples,
                                                        ood_loader, device)
                    else:
                        row += f" {'N/A':>{col_w}}"
                        continue
                    row += f" {auroc_safe(id_eu, ood_eu_bl):{col_w}.4f}"
                except Exception as e:
                    row += f" {'N/A':>{col_w}}"
            print(row)

        except Exception as e:
            print(f"  {ood_name:<22} SKIPPED ({e})")

    # -----------------------------------------------------------------------
    # AURC Selective Prediction
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  Selective Prediction AURC (TinyImageNet val)")
    print(f"{'='*80}")
    print(f"  {'Method':<35} {'AURC↓':>10} {'OracleGap↓':>12} {'@90%↑':>8} {'@80%↑':>8}")
    print(f"  {'-'*75}")

    tea_errors = (teacher_probs.argmax(-1) != true_labels).astype(float)
    _print_aurc("Teacher EU", tea_errors, tea_eu_np)
    if student_probs_np is not None:
        stu_errors = (student_probs_np.argmax(-1) != true_labels).astype(float)
        _print_aurc("Student EU (distilled)", stu_errors, student_eu_np)
        _print_aurc("Student entropy", stu_errors, entropy(student_probs_np))

    for name, (eu_np, probs_np, _) in baselines.items():
        bl_errors = (probs_np.argmax(-1) != true_labels).astype(float)
        _print_aurc(f"{name} EU", bl_errors, eu_np)
        _print_aurc(f"{name} entropy", bl_errors, entropy(probs_np))

    # -----------------------------------------------------------------------
    # Inference Throughput
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  Inference Throughput (device={device}, bs=32, 30 batches)")
    print(f"{'='*80}")
    print(f"  {'Method':<38} {'Samples/sec':>14}")
    print(f"  {'-'*54}")

    # Load ensemble for reference
    all_members = []
    cfg_path = os.path.join(args.save_dir, "ensemble_configs.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            configs = json.load(f)
        for idx, cfg in enumerate(configs):
            mpath = os.path.join(args.save_dir, f"member_{idx}.pt")
            if not os.path.isfile(mpath): break
            m = create_ensemble_member(
                rank=cfg.get("rank", 8), alpha=cfg.get("alpha", None),
                lora_dropout=0.0,
                targets=cfg.get("targets", "qkv+proj"),
                unfreeze_blocks=cfg.get("unfreeze_blocks", 0), pretrained=True)
            mckpt = torch.load(mpath, map_location=device, weights_only=True)
            load_saved_member_state(m, mckpt, strict=False)
            m.to(device).eval()
            all_members.append(m)

    if all_members:
        K = len(all_members)
        def _ens_fn(x):
            return torch.stack([F.softmax(m(x), -1) for m in all_members]).mean(0)
        tp_ens = measure_throughput(_ens_fn, img_shape, device, batch_size=32, n_batches=30)
        print(f"  {'Ensemble (K='+str(K)+')':<38} {tp_ens:>14,.0f}")

    if os.path.isfile(student_path):
        tp_stu = measure_throughput(lambda x: stu(x), img_shape, device,
                                    batch_size=32, n_batches=30)
        print(f"  {'Student (single pass)':<38} {tp_stu:>14,.0f}")

    if os.path.isfile(mc_path) and "mc_dropout" in args.methods:
        mc_m_tp = load_member0(args.save_dir, device)
        ckpt_tp = torch.load(mc_path, map_location="cpu")
        mc_m_tp = enable_attention_dropout(mc_m_tp, ckpt_tp.get("dropout_p", 0.1))
        load_saved_member_state(mc_m_tp, {"member_state": ckpt_tp["member_state"]}, strict=False)
        mc_m_tp.to(device).train()
        T = args.T
        def _mc_fn(x):
            return torch.stack([F.softmax(mc_m_tp(x), -1) for _ in range(T)]).mean(0)
        tp_mc = measure_throughput(_mc_fn, img_shape, device, batch_size=32, n_batches=30)
        print(f"  {'MC Dropout (T='+str(T)+')':<38} {tp_mc:>14,.0f}")

    if sgld_backbone is not None and sgld_samples is not None:
        T_sg = len(sgld_samples)
        sgld_backbone.to(device).eval()
        ws   = [s["weight"].to(device) for s in sgld_samples]
        bs_l = [s["bias"].to(device)   for s in sgld_samples]
        def _sgld_fn(x):
            with torch.no_grad():
                feat = _run_backbone(sgld_backbone, x, device)
            return torch.stack([F.softmax(feat @ w.T + b, -1) for w, b in zip(ws, bs_l)]).mean(0)
        tp_sg = measure_throughput(_sgld_fn, img_shape, device, batch_size=32, n_batches=30)
        print(f"  {'SGLD (T='+str(T_sg)+')':<38} {tp_sg:>14,.0f}")

    print(f"\nBaseline evaluation complete.")


if __name__ == "__main__":
    main()
