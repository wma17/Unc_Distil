"""
Evaluate BNN baselines for CIFAR-10 and compare with the distilled student.

Computes EU via mutual information (MI = H[E[p]] - E[H[p]]) for all
sampling-based methods (MC Dropout, LLLA, SGLD).

Metrics aligned with evaluate_student.py:
  - Accuracy, ECE-15, NLL, Brier Score
  - EU Pearson / Spearman (vs teacher ensemble EU)
  - OOD AUROC (student EU, entropy, 1-MaxProb)
  - AURC + selective prediction
  - Inference throughput (samples/sec)

Usage:
    python evaluate_baselines.py --save_dir ./checkpoints --gpu 0
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from models import cifar_resnet18, cifar_resnet18_student, BasicBlock
from train_baselines import (MCDropoutResNet, EDLHead, FEAT_DIM, NUM_CLASSES,
                              CIFAR10_MEAN, CIFAR10_STD)
from cache_ensemble_targets import apply_corruption, CORRUPTION_TYPES, CORRUPTION_SEED
from evaluate_student import (
    get_ood_loader, get_corrupted_test_loader,
    pearson_corr, spearman_corr,
    compute_ece, compute_nll, compute_brier, compute_aurc, auroc,
    measure_throughput, SEEN_OOD, UNSEEN_OOD,
)


EPS = 1e-8


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_test_loader(data_dir, batch_size, num_workers=4):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = datasets.CIFAR10(data_dir, train=False, download=False, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# EU helpers (mutual information from K samples)
# ---------------------------------------------------------------------------

def mutual_information(probs_stack):
    """Compute EU = H[E[p]] - E[H[p]] from (T, N, K) probability array.

    Args:
        probs_stack: np.ndarray of shape (T, N, K) — T samples, N examples, K classes
    Returns:
        eu: np.ndarray of shape (N,)
        mean_probs: np.ndarray of shape (N, K)
    """
    mean_probs = probs_stack.mean(axis=0)                               # (N, K)
    h_mean = -(mean_probs * np.log(mean_probs + EPS)).sum(axis=-1)     # (N,)
    h_each = -(probs_stack * np.log(probs_stack + EPS)).sum(axis=-1)   # (T, N)
    mean_h = h_each.mean(axis=0)                                        # (N,)
    eu = np.maximum(h_mean - mean_h, 0.0)
    return eu, mean_probs


# ---------------------------------------------------------------------------
# MC Dropout inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_mc_dropout(mc_model, loader, device, T=16):
    """Run T stochastic forward passes; return (eu, mean_probs, labels)."""
    mc_model.to(device)
    mc_model.train()   # activates Dropout2d at test time

    # Collect all images (one pass to get structure)
    all_imgs, all_labels = [], []
    for imgs, labels in loader:
        all_imgs.append(imgs)
        all_labels.append(labels)
    all_imgs   = torch.cat(all_imgs).to(device)
    all_labels = torch.cat(all_labels)

    sample_probs = []
    for _ in range(T):
        probs_list = []
        bs = 256
        for i in range(0, len(all_imgs), bs):
            batch = all_imgs[i:i+bs]
            logits = mc_model(batch)
            probs_list.append(F.softmax(logits, dim=-1).cpu().numpy())
        sample_probs.append(np.concatenate(probs_list, axis=0))

    probs_stack = np.stack(sample_probs, axis=0)  # (T, N, K)
    eu, mean_probs = mutual_information(probs_stack)
    return eu, mean_probs, all_labels.numpy()


# ---------------------------------------------------------------------------
# EDL inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_edl(backbone, edl_head, loader, device):
    """Run EDL forward pass; return (eu_epi, eu_ale, mean_probs, labels)."""
    backbone.to(device).eval()
    edl_head.to(device).eval()

    all_eu_epi, all_eu_ale, all_probs, all_labels = [], [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        # Extract features via frozen backbone
        out = F.relu(backbone.bn1(backbone.conv1(imgs)))
        out = backbone.layer1(out); out = backbone.layer2(out)
        out = backbone.layer3(out); out = backbone.layer4(out)
        out = backbone.avgpool(out)
        feat = out.view(out.size(0), -1)

        alpha = edl_head(feat)                           # (B, K)
        S     = alpha.sum(dim=-1, keepdim=True)          # (B, 1)
        p_hat = (alpha / S).cpu()                        # (B, K) — mean of Dirichlet

        # Aleatoric: U_ale = ψ(S+1) - Σ (α_k/S)·ψ(α_k+1)
        u_ale = (torch.digamma(S + 1)
                 - (alpha / S * torch.digamma(alpha + 1)).sum(dim=-1, keepdim=True)
                 ).cpu().squeeze(-1)

        # Epistemic: U_epi = H[p_hat] - U_ale  (clamped ≥ 0)
        h_p = -(p_hat * torch.log(p_hat + EPS)).sum(dim=-1)
        u_epi = (h_p - u_ale).clamp(min=0.0)

        all_eu_epi.append(u_epi.numpy())
        all_eu_ale.append(u_ale.numpy())
        all_probs.append(p_hat.numpy())
        all_labels.append(labels.numpy())

    return (np.concatenate(all_eu_epi), np.concatenate(all_eu_ale),
            np.concatenate(all_probs), np.concatenate(all_labels))


# ---------------------------------------------------------------------------
# LLLA inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_llla(backbone, la, loader, device, T=16):
    """Use Laplace posterior samples to compute EU via MI."""
    try:
        from laplace import Laplace
    except ImportError:
        print("  [LLLA] laplace-torch not installed.")
        return None, None, None

    backbone.to(device).eval()

    class BackboneWithFC(nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b
        def forward(self, x):
            out = F.relu(self.b.bn1(self.b.conv1(x)))
            out = self.b.layer1(out); out = self.b.layer2(out)
            out = self.b.layer3(out); out = self.b.layer4(out)
            out = self.b.avgpool(out)
            return self.b.fc(out.view(out.size(0), -1))

    all_imgs, all_labels = [], []
    for imgs, labels in loader:
        all_imgs.append(imgs)
        all_labels.append(labels.numpy())
    all_imgs = torch.cat(all_imgs)

    # Predictive samples from Laplace posterior: (T, N, K) → probs
    sample_probs = []
    bs = 128
    for i in range(0, len(all_imgs), bs):
        batch = all_imgs[i:i+bs].to(device)
        # la.predictive_samples returns logit samples (T, B, K)
        logit_samps = la.predictive_samples(batch, pred_type="glm", n_samples=T)
        prob_samps   = F.softmax(logit_samps, dim=-1).cpu().numpy()  # (T, B, K)
        sample_probs.append(prob_samps)
    # Concatenate along the N dimension
    probs_stack = np.concatenate(sample_probs, axis=1)  # (T, N, K)
    eu, mean_probs = mutual_information(probs_stack)
    return eu, mean_probs, np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# SGLD inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_sgld(backbone, samples, loader, device):
    """Run inference with SGLD last-layer samples; return (eu, mean_probs, labels)."""
    backbone.to(device).eval()

    all_imgs, all_labels = [], []
    for imgs, labels in loader:
        all_imgs.append(imgs)
        all_labels.append(labels.numpy())
    all_imgs = torch.cat(all_imgs).to(device)

    # Collect backbone features once
    feat_chunks = []
    bs = 256
    for i in range(0, len(all_imgs), bs):
        batch = all_imgs[i:i+bs]
        out = F.relu(backbone.bn1(backbone.conv1(batch)))
        out = backbone.layer1(out); out = backbone.layer2(out)
        out = backbone.layer3(out); out = backbone.layer4(out)
        out = backbone.avgpool(out)
        feat_chunks.append(out.view(out.size(0), -1).cpu())
    feats = torch.cat(feat_chunks)  # (N, 512)

    sample_probs = []
    for s in samples:
        w = s["weight"].to(device)   # (K, 512)
        b = s["bias"].to(device)     # (K,)
        probs_chunks = []
        for i in range(0, len(feats), bs):
            f_batch = feats[i:i+bs].to(device)
            logits = f_batch @ w.T + b
            probs_chunks.append(F.softmax(logits, dim=-1).cpu().numpy())
        sample_probs.append(np.concatenate(probs_chunks, axis=0))

    probs_stack = np.stack(sample_probs, axis=0)  # (T, N, K)
    eu, mean_probs = mutual_information(probs_stack)
    return eu, mean_probs, np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def print_row(name, acc, ece, nll, brier, eu_p, eu_s):
    print(f"  {name:<30} {acc:>8.2f} {ece:>8.4f} {nll:>8.4f} {brier:>8.4f} "
          f"{eu_p:>8.4f} {eu_s:>8.4f}")


def get_ood_eu(model_fn, ood_id, data_dir, batch_size, num_workers, device):
    """Run model_fn (returns eu, probs) on an OOD loader."""
    try:
        ood_loader = get_ood_loader(ood_id, data_dir, batch_size, num_workers)
        eu, probs, _ = model_fn(ood_loader, device)
        return eu, probs
    except Exception as e:
        print(f"    [{ood_id}] SKIPPED: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Throughput helpers
# ---------------------------------------------------------------------------

def _mc_throughput(mc_model, T, img_shape, device, batch_size=256, n_batches=50):
    dummy = torch.randn(batch_size, *img_shape, device=device)
    mc_model.train()
    # warmup
    for _ in range(5):
        for _ in range(T):
            mc_model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_batches):
        for _ in range(T):
            mc_model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (batch_size * n_batches) / (time.time() - t0)


def _sgld_throughput(backbone, samples, img_shape, device, batch_size=256, n_batches=50):
    dummy = torch.randn(batch_size, *img_shape, device=device)
    # Pre-extract features
    with torch.no_grad():
        out = F.relu(backbone.bn1(backbone.conv1(dummy)))
        out = backbone.layer1(out); out = backbone.layer2(out)
        out = backbone.layer3(out); out = backbone.layer4(out)
        feat = backbone.avgpool(out).view(dummy.size(0), -1)
    ws = [s["weight"].to(device) for s in samples]
    bs_list = [s["bias"].to(device) for s in samples]
    # warmup
    for _ in range(5):
        for w, b in zip(ws, bs_list):
            feat @ w.T + b
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_batches):
        for w, b in zip(ws, bs_list):
            feat @ w.T + b
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (batch_size * n_batches) / (time.time() - t0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir",    default="./checkpoints")
    p.add_argument("--data_dir",    default="/home/maw6/maw6/unc_regression/data")
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--T",           type=int, default=16,
                   help="Number of MC samples for MC Dropout / LLLA / SGLD")
    p.add_argument("--methods", nargs="+",
                   default=["mc_dropout", "edl", "llla", "sgld"],
                   choices=["mc_dropout", "edl", "llla", "sgld"])
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    baseline_dir = os.path.join(args.save_dir, "baselines")
    data = np.load(os.path.join(args.save_dir, "teacher_targets.npz"))
    teacher_eu_t  = data["test_eu"]       # (N,)
    true_labels   = data["test_labels"]   # (N,)
    teacher_probs = data["test_probs"]    # (N, K)

    test_loader = get_test_loader(args.data_dir, args.batch_size, args.num_workers)
    img_shape   = (3, 32, 32)

    # -----------------------------------------------------------------------
    # Load baselines
    # -----------------------------------------------------------------------

    baselines = {}   # name -> (eu_np, probs_np, labels_np)

    # ---- MC Dropout ----
    mc_path = os.path.join(baseline_dir, "mc_dropout.pt")
    if "mc_dropout" in args.methods and os.path.isfile(mc_path):
        print("\n[Loading MC Dropout]")
        ckpt = torch.load(mc_path, map_location="cpu")
        mc_model = MCDropoutResNet(dropout_p=ckpt.get("dropout_p", 0.1))
        mc_model.load_state_dict(ckpt["model_state_dict"])
        eu, probs, labels = predict_mc_dropout(mc_model, test_loader, device, T=args.T)
        baselines["MC Dropout"] = (eu, probs, labels)
        print(f"  EU mean={eu.mean():.4f}  acc={100*(probs.argmax(-1)==labels).mean():.2f}%")
    else:
        if "mc_dropout" in args.methods:
            print(f"  [MC Dropout] checkpoint not found: {mc_path}")

    # ---- EDL ----
    edl_path = os.path.join(baseline_dir, "edl_head.pt")
    if "edl" in args.methods and os.path.isfile(edl_path):
        print("\n[Loading EDL]")
        ckpt = torch.load(edl_path, map_location="cpu")
        backbone = cifar_resnet18()
        backbone.load_state_dict(ckpt["backbone_state_dict"])
        edl_head = EDLHead(feat_dim=FEAT_DIM, num_classes=NUM_CLASSES)
        edl_head.load_state_dict(ckpt["edl_head_state_dict"])

        eu_epi, eu_ale, probs, labels = predict_edl(backbone, edl_head, test_loader, device)
        baselines["EDL"] = (eu_epi, probs, labels)
        baselines["EDL (aleatoric)"] = (eu_ale, probs, labels)  # for reference
        print(f"  EU_epi mean={eu_epi.mean():.4f}  EU_ale mean={eu_ale.mean():.4f}"
              f"  acc={100*(probs.argmax(-1)==labels).mean():.2f}%")
    else:
        if "edl" in args.methods:
            print(f"  [EDL] checkpoint not found: {edl_path}")

    # ---- LLLA ----
    llla_path = os.path.join(baseline_dir, "llla.pt")
    llla_bb_path = os.path.join(baseline_dir, "llla_backbone.pt")
    la_obj = None
    if "llla" in args.methods and os.path.isfile(llla_path):
        print("\n[Loading LLLA]")
        try:
            from laplace import Laplace

            llla_bb_ckpt = torch.load(llla_bb_path, map_location="cpu")
            backbone_llla = cifar_resnet18()
            backbone_llla.load_state_dict(llla_bb_ckpt["backbone_state_dict"])

            class BackboneWithFC(nn.Module):
                def __init__(self, b):
                    super().__init__()
                    self.b = b
                def forward(self, x):
                    out = F.relu(self.b.bn1(self.b.conv1(x)))
                    out = self.b.layer1(out); out = self.b.layer2(out)
                    out = self.b.layer3(out); out = self.b.layer4(out)
                    out = self.b.avgpool(out)
                    return self.b.fc(out.view(out.size(0), -1))

            wrapped = BackboneWithFC(backbone_llla)
            la_obj = Laplace(wrapped, "classification",
                             subset_of_weights="last_layer",
                             hessian_structure="kron")
            la_obj.load_state_dict(torch.load(llla_path, map_location="cpu"))
            la_obj.model.to(device)

            eu_llla, probs_llla, labels_llla = predict_llla(
                backbone_llla, la_obj, test_loader, device, T=args.T)
            if eu_llla is not None:
                baselines["LLLA (KFAC)"] = (eu_llla, probs_llla, labels_llla)
                print(f"  EU mean={eu_llla.mean():.4f}"
                      f"  acc={100*(probs_llla.argmax(-1)==labels_llla).mean():.2f}%")
        except ImportError:
            print("  [LLLA] laplace-torch not installed.")
        except Exception as e:
            print(f"  [LLLA] Error loading: {e}")
    else:
        if "llla" in args.methods:
            print(f"  [LLLA] checkpoint not found: {llla_path}")

    # ---- SGLD ----
    sgld_path = os.path.join(baseline_dir, "sgld_samples.pt")
    sgld_samples = None
    sgld_backbone = None
    if "sgld" in args.methods and os.path.isfile(sgld_path):
        print("\n[Loading SGLD]")
        ckpt = torch.load(sgld_path, map_location="cpu")
        sgld_backbone = cifar_resnet18()
        sgld_backbone.load_state_dict(ckpt["backbone_state_dict"])
        sgld_samples = ckpt["samples"]
        print(f"  Loaded {len(sgld_samples)} SGLD samples")

        eu_sgld, probs_sgld, labels_sgld = predict_sgld(
            sgld_backbone, sgld_samples, test_loader, device)
        baselines["SGLD"] = (eu_sgld, probs_sgld, labels_sgld)
        print(f"  EU mean={eu_sgld.mean():.4f}"
              f"  acc={100*(probs_sgld.argmax(-1)==labels_sgld).mean():.2f}%")
    else:
        if "sgld" in args.methods:
            print(f"  [SGLD] checkpoint not found: {sgld_path}")

    # Also load student (if available) for comparison
    student_path = os.path.join(args.save_dir, "student.pt")
    student_eu_np = None
    student_probs_np = None
    if os.path.isfile(student_path):
        print("\n[Loading student for comparison]")
        from evaluate_student import load_student, predict_student
        student = load_student(args.save_dir, device)
        stu_probs_t, stu_eu_t = predict_student(student, test_loader, device)
        student_eu_np    = stu_eu_t.numpy()
        student_probs_np = stu_probs_t.numpy()

    if not baselines:
        print("\nNo baselines found. Run train_baselines.py first.")
        return

    # -----------------------------------------------------------------------
    # Print metric tables
    # -----------------------------------------------------------------------

    print(f"\n{'='*80}")
    print(f"  CIFAR-10 Baseline Evaluation Summary")
    print(f"{'='*80}")

    # Helper: print calibration block
    print(f"\n{'='*80}")
    print(f"  Accuracy & Calibration (clean CIFAR-10 test set)")
    print(f"{'='*80}")
    print(f"  {'Method':<30} {'Acc%':>8} {'ECE-15':>8} {'NLL':>8} {'Brier':>8} "
          f"{'EU-Pear':>8} {'EU-Spear':>9}")
    print(f"  {'-'*80}")

    tea_ece   = compute_ece(teacher_probs, true_labels)
    tea_nll   = compute_nll(teacher_probs, true_labels)
    tea_brier = compute_brier(teacher_probs, true_labels)
    tea_acc   = 100.0 * (teacher_probs.argmax(-1) == true_labels).mean()
    tea_eu_t  = torch.from_numpy(teacher_eu_t).float()
    print(f"  {'Teacher (ensemble)':<30} {tea_acc:>8.2f} {tea_ece:>8.4f} {tea_nll:>8.4f} "
          f"{tea_brier:>8.4f} {'—':>8} {'—':>9}")

    if student_probs_np is not None:
        s_ece   = compute_ece(student_probs_np, true_labels)
        s_nll   = compute_nll(student_probs_np, true_labels)
        s_brier = compute_brier(student_probs_np, true_labels)
        s_acc   = 100.0 * (student_probs_np.argmax(-1) == true_labels).mean()
        s_eu_t  = torch.from_numpy(student_eu_np).float()
        s_rp    = pearson_corr(s_eu_t, tea_eu_t)
        s_rs    = spearman_corr(s_eu_t, tea_eu_t)
        print(f"  {'Student (distilled)':<30} {s_acc:>8.2f} {s_ece:>8.4f} {s_nll:>8.4f} "
              f"{s_brier:>8.4f} {s_rp:>8.4f} {s_rs:>9.4f}")

    for name, (eu_np, probs_np, _) in baselines.items():
        if name.endswith("(aleatoric)"):
            continue  # shown separately below
        ece   = compute_ece(probs_np, true_labels)
        nll   = compute_nll(probs_np, true_labels)
        brier = compute_brier(probs_np, true_labels)
        acc   = 100.0 * (probs_np.argmax(-1) == true_labels).mean()
        eu_t  = torch.from_numpy(eu_np).float()
        rp    = pearson_corr(eu_t, tea_eu_t)
        rs    = spearman_corr(eu_t, tea_eu_t)
        print_row(name, acc, ece, nll, brier, rp, rs)

    # -----------------------------------------------------------------------
    # OOD Detection AUROC
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  OOD Detection AUROC (ID = Clean CIFAR-10 test)")
    print(f"{'='*80}")

    all_ood_items = list(SEEN_OOD) + list(UNSEEN_OOD)
    header_methods = ["Teacher EU"]
    if student_probs_np is not None:
        header_methods += ["Student EU", "Stu Entropy"]
    for name in [n for n in baselines if not n.endswith("(aleatoric)")]:
        header_methods.append(name)

    col_w = 11
    header_row = f"  {'OOD Dataset':<20} {'Type':<7}"
    for m in header_methods:
        header_row += f" {m[:col_w]:>{col_w}}"
    print(header_row)
    print(f"  {'-'*80}")

    for ood_id, cache_key, display_name in all_ood_items:
        if cache_key not in data:
            continue
        ood_type = "seen" if (ood_id, cache_key, display_name) in SEEN_OOD else "unseen"
        try:
            ood_loader = get_ood_loader(ood_id, args.data_dir,
                                         args.batch_size, args.num_workers)
        except Exception as e:
            print(f"  {display_name:<20} SKIPPED ({e})")
            continue

        ood_tea_eu = data[cache_key]
        n_ood = len(ood_tea_eu)

        # Run each baseline on OOD data
        row = f"  {display_name:<20} {ood_type:<7}"

        # Teacher EU AUROC
        a_tea = auroc(teacher_eu_t[:10000], ood_tea_eu[:n_ood])
        row += f" {a_tea:{col_w}.4f}"

        if student_probs_np is not None:
            ood_stu_probs, ood_stu_eu = _infer_student_ood(
                student_path, args.save_dir, ood_loader, device)
            if ood_stu_eu is not None:
                a_stu = auroc(student_eu_np, ood_stu_eu)
                ood_ent = -(ood_stu_probs * np.log(ood_stu_probs + EPS)).sum(-1)
                id_ent  = -(student_probs_np * np.log(student_probs_np + EPS)).sum(-1)
                a_ent   = auroc(id_ent, ood_ent)
                row += f" {a_stu:{col_w}.4f} {a_ent:{col_w}.4f}"
            else:
                row += f" {'N/A':>{col_w}} {'N/A':>{col_w}}"

        for name, (id_eu, id_probs, _) in baselines.items():
            if name.endswith("(aleatoric)"):
                continue
            ood_eu_bl, ood_probs_bl = _infer_baseline_ood(
                name, baseline_dir, ood_loader, device, args.T)
            if ood_eu_bl is not None:
                a = auroc(id_eu, ood_eu_bl)
                row += f" {a:{col_w}.4f}"
            else:
                row += f" {'N/A':>{col_w}}"

        print(row)

    # -----------------------------------------------------------------------
    # AURC selective prediction
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  Selective Prediction AURC (Clean CIFAR-10 test)")
    print(f"{'='*80}")
    print(f"  {'Method':<30} {'AURC↓':>10} {'OracleGap↓':>12} {'@90%↑':>8} {'@80%↑':>8}")
    print(f"  {'-'*70}")

    errors = (teacher_probs.argmax(-1) != true_labels).astype(float)

    if student_probs_np is not None:
        _print_aurc("Teacher EU", errors, teacher_eu_t)
        _print_aurc("Student EU (distilled)", errors, student_eu_np)
        id_ent_stu = -(student_probs_np * np.log(student_probs_np + EPS)).sum(-1)
        _print_aurc("Student entropy", errors, id_ent_stu)

    for name, (eu_np, probs_np, _) in baselines.items():
        if name.endswith("(aleatoric)"):
            continue
        bl_errors = (probs_np.argmax(-1) != true_labels).astype(float)
        _print_aurc(name + " EU", bl_errors, eu_np)
        bl_ent = -(probs_np * np.log(probs_np + EPS)).sum(-1)
        _print_aurc(name + " entropy", bl_errors, bl_ent)

    print(f"  {'Oracle':<30} {'—':>10} {'0.000000':>12} {'—':>8} {'—':>8}")
    print(f"  {'Random':<30} {errors.mean():>10.6f} {'—':>12} {'—':>8} {'—':>8}")

    # -----------------------------------------------------------------------
    # Throughput
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  Inference Throughput (device={device}, bs=256, 50 batches)")
    print(f"{'='*80}")
    print(f"  {'Method':<38} {'Samples/sec':>14}")
    print(f"  {'-'*54}")

    # Load ensemble members for reference
    K_members = 0
    all_members = []
    for midx in range(20):
        mpath = os.path.join(args.save_dir, f"member_{midx}.pt")
        if not os.path.exists(mpath):
            break
        mckpt = torch.load(mpath, map_location=device, weights_only=True)
        mcfg = mckpt.get("member_config", {})
        mem = cifar_resnet18(num_classes=10,
                              dropout_rate=mcfg.get("dropout_rate", 0.0)).to(device)
        mem.load_state_dict(mckpt["model_state_dict"])
        mem.eval()
        all_members.append(mem)

    if all_members:
        K_members = len(all_members)
        def _ens_fn(x):
            return torch.stack([F.softmax(m(x), -1) for m in all_members]).mean(0)
        ens_tp = measure_throughput(_ens_fn, img_shape, device,
                                    batch_size=256, n_batches=50)
        print(f"  {'Ensemble (K='+str(K_members)+', reference)':<38} {ens_tp:>14,.0f}")

    if os.path.isfile(student_path):
        from evaluate_student import load_student
        stu = load_student(args.save_dir, device)
        stu_tp = measure_throughput(lambda x: stu(x), img_shape, device,
                                    batch_size=256, n_batches=50)
        print(f"  {'Student (single pass)':<38} {stu_tp:>14,.0f}")

    mc_bl_path = os.path.join(baseline_dir, "mc_dropout.pt")
    if os.path.isfile(mc_bl_path) and "mc_dropout" in args.methods:
        ckpt = torch.load(mc_bl_path, map_location=device)
        mc_m = MCDropoutResNet(dropout_p=ckpt.get("dropout_p", 0.1)).to(device)
        mc_m.load_state_dict(ckpt["model_state_dict"])
        T = args.T
        mc_tp = _mc_throughput(mc_m, T, img_shape, device)
        print(f"  {'MC Dropout (T='+str(T)+')':<38} {mc_tp:>14,.0f}")

    if sgld_backbone is not None and sgld_samples is not None:
        sgld_backbone.to(device).eval()
        T = len(sgld_samples)
        sg_tp = _sgld_throughput(sgld_backbone, sgld_samples, img_shape, device)
        print(f"  {'SGLD (T='+str(T)+')':<38} {sg_tp:>14,.0f}")

    print(f"\nBaseline evaluation complete.")


# ---------------------------------------------------------------------------
# Deferred OOD inference helpers (to avoid circular dependency)
# ---------------------------------------------------------------------------

def _infer_student_ood(student_path, save_dir, ood_loader, device):
    """Run student on OOD loader; returns (probs_np, eu_np) or (None, None)."""
    try:
        from evaluate_student import load_student, predict_student
        student = load_student(save_dir, device)
        probs_t, eu_t = predict_student(student, ood_loader, device)
        return probs_t.numpy(), eu_t.numpy()
    except Exception:
        return None, None


def _infer_baseline_ood(name, baseline_dir, ood_loader, device, T=16):
    """Run the named baseline on an OOD loader; returns (eu_np, probs_np) or (None, None)."""
    try:
        if name == "MC Dropout":
            ckpt = torch.load(os.path.join(baseline_dir, "mc_dropout.pt"),
                              map_location="cpu")
            mc_m = MCDropoutResNet(dropout_p=ckpt.get("dropout_p", 0.1))
            mc_m.load_state_dict(ckpt["model_state_dict"])
            eu, probs, _ = predict_mc_dropout(mc_m, ood_loader, device, T=T)
            return eu, probs

        elif name == "EDL":
            ckpt = torch.load(os.path.join(baseline_dir, "edl_head.pt"),
                              map_location="cpu")
            bb = cifar_resnet18()
            bb.load_state_dict(ckpt["backbone_state_dict"])
            head = EDLHead(feat_dim=FEAT_DIM, num_classes=NUM_CLASSES)
            head.load_state_dict(ckpt["edl_head_state_dict"])
            eu, _, probs, _ = predict_edl(bb, head, ood_loader, device)
            return eu, probs

        elif name == "LLLA (KFAC)":
            from laplace import Laplace
            bb_ckpt = torch.load(os.path.join(baseline_dir, "llla_backbone.pt"),
                                 map_location="cpu")
            bb = cifar_resnet18()
            bb.load_state_dict(bb_ckpt["backbone_state_dict"])
            # la_obj was built in the calling scope — pass via module-level ref
            eu, probs, _ = predict_llla(bb, _LLLA_OBJ, ood_loader, device, T=T)
            return eu, probs

        elif name == "SGLD":
            ckpt = torch.load(os.path.join(baseline_dir, "sgld_samples.pt"),
                              map_location="cpu")
            bb = cifar_resnet18()
            bb.load_state_dict(ckpt["backbone_state_dict"])
            eu, probs, _ = predict_sgld(bb, ckpt["samples"], ood_loader, device)
            return eu, probs

    except Exception as e:
        print(f"    [{name} OOD] error: {e}")
        return None, None

    return None, None


# Module-level LLLA object reference for OOD inference
_LLLA_OBJ = None


def _print_aurc(name, errors, scores):
    a, _, gap, a90, a80 = compute_aurc(errors, np.asarray(scores))
    print(f"  {name:<30} {a:>10.6f} {gap:>12.6f} {a90:>8.4f} {a80:>8.4f}")


if __name__ == "__main__":
    main()
