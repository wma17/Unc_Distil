"""
Evaluate BNN baselines for MNIST and compare with the distilled student.

Metrics aligned with evaluate_student.py:
  - Accuracy, ECE-15, NLL, Brier Score
  - EU Pearson / Spearman (vs teacher ensemble EU)
  - OOD AUROC (seen + unseen)
  - AURC + selective prediction
  - Inference throughput

Usage:
    python evaluate_baselines.py --save_dir ./checkpoints --gpu 1
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import mnist_convnet, mnist_convnet_student
from train_baselines import (MCDropoutConvNet, EDLHead, FEAT_DIM, NUM_CLASSES,
                              MNIST_MEAN, MNIST_STD, EPS)
from evaluate_student import (
    get_ood_loader, compute_ece, compute_nll, compute_brier, compute_aurc,
    auroc, pearson_corr, spearman_corr, measure_throughput,
    SEEN_OOD, UNSEEN_OOD,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_test_loader(data_dir, batch_size, num_workers=4):
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    ds = datasets.MNIST(data_dir, train=False, download=False, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# EU helpers
# ---------------------------------------------------------------------------

def mutual_information(probs_stack):
    """EU = H[E[p]] - E[H[p]] from (T, N, K) array."""
    mean_probs = probs_stack.mean(axis=0)
    h_mean = -(mean_probs * np.log(mean_probs + EPS)).sum(axis=-1)
    h_each = -(probs_stack * np.log(probs_stack + EPS)).sum(axis=-1)
    eu     = np.maximum(h_mean - h_each.mean(axis=0), 0.0)
    return eu, mean_probs


# ---------------------------------------------------------------------------
# Baseline inference functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_mc_dropout(mc_model, loader, device, T=16):
    mc_model.to(device).train()  # activates Dropout2d
    all_imgs, all_labels = [], []
    for imgs, labels in loader:
        all_imgs.append(imgs); all_labels.append(labels)
    all_imgs = torch.cat(all_imgs).to(device)

    sample_probs = []
    bs = 256
    for _ in range(T):
        chunks = []
        for i in range(0, len(all_imgs), bs):
            chunks.append(F.softmax(mc_model(all_imgs[i:i+bs]), -1).cpu().numpy())
        sample_probs.append(np.concatenate(chunks))

    eu, mean_probs = mutual_information(np.stack(sample_probs))
    return eu, mean_probs, torch.cat(all_labels).numpy()


@torch.no_grad()
def predict_edl(backbone, edl_head, loader, device):
    backbone.to(device).eval(); edl_head.to(device).eval()
    all_eu_epi, all_eu_ale, all_probs, all_labels = [], [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        out  = backbone.features(imgs)
        out  = backbone.avgpool(out)
        feat = out.view(out.size(0), -1)

        alpha = edl_head(feat)
        S     = alpha.sum(dim=-1, keepdim=True)
        p_hat = (alpha / S).cpu()

        u_ale = (torch.digamma(S + 1)
                 - (alpha / S * torch.digamma(alpha + 1)).sum(dim=-1, keepdim=True)
                 ).cpu().squeeze(-1)
        h_p   = -(p_hat * torch.log(p_hat + EPS)).sum(dim=-1)
        u_epi = (h_p - u_ale).clamp(min=0.0)

        all_eu_epi.append(u_epi.numpy())
        all_eu_ale.append(u_ale.numpy())
        all_probs.append(p_hat.numpy())
        all_labels.append(labels.numpy())

    return (np.concatenate(all_eu_epi), np.concatenate(all_eu_ale),
            np.concatenate(all_probs), np.concatenate(all_labels))


@torch.no_grad()
def predict_llla(backbone, la, loader, device, T=16):
    all_imgs, all_labels = [], []
    for imgs, labels in loader:
        all_imgs.append(imgs); all_labels.append(labels.numpy())
    all_imgs = torch.cat(all_imgs)

    sample_probs = []
    bs = 128
    for i in range(0, len(all_imgs), bs):
        batch = all_imgs[i:i+bs].to(device)
        logit_samps = la.predictive_samples(batch, pred_type="glm", n_samples=T)
        sample_probs.append(F.softmax(logit_samps, -1).cpu().numpy())
    probs_stack = np.concatenate(sample_probs, axis=1)
    eu, mean_probs = mutual_information(probs_stack)
    return eu, mean_probs, np.concatenate(all_labels)


@torch.no_grad()
def predict_sgld(backbone, samples, loader, device):
    backbone.to(device).eval()
    all_imgs, all_labels = [], []
    for imgs, labels in loader:
        all_imgs.append(imgs); all_labels.append(labels.numpy())
    all_imgs = torch.cat(all_imgs).to(device)

    bs = 256
    feat_chunks = []
    for i in range(0, len(all_imgs), bs):
        out = backbone.features(all_imgs[i:i+bs])
        out = backbone.avgpool(out)
        feat_chunks.append(out.view(out.size(0), -1).cpu())
    feats = torch.cat(feat_chunks)

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
# Helpers
# ---------------------------------------------------------------------------

def _print_aurc(name, errors, scores):
    a, _, gap, a90, a80 = compute_aurc(errors, np.asarray(scores))
    print(f"  {name:<32} {a:>10.6f} {gap:>12.6f} {a90:>8.4f} {a80:>8.4f}")


def _load_baseline(name, baseline_dir, device):
    """Load a baseline and return (eu, probs, labels) on test set using cache."""
    # This is called during main(); actual loading happens per-baseline below.
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir",    default="./checkpoints")
    p.add_argument("--data_dir",    default="/home/maw6/maw6/unc_regression/data")
    p.add_argument("--gpu",         type=int, default=1)
    p.add_argument("--batch_size",  type=int, default=256)
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
    npz = np.load(os.path.join(args.save_dir, "teacher_targets.npz"))
    teacher_eu_np  = npz["test_eu"]
    true_labels    = npz["test_labels"]
    teacher_probs  = npz["test_probs"]
    tea_eu_t       = torch.from_numpy(teacher_eu_np).float()

    test_loader = get_test_loader(args.data_dir, args.batch_size, args.num_workers)
    img_shape   = (3, 28, 28)

    # -----------------------------------------------------------------------
    # Load and run all baselines on the test set
    # -----------------------------------------------------------------------
    baselines = {}   # name -> (eu_np, probs_np, labels_np)

    # MC Dropout
    mc_path = os.path.join(baseline_dir, "mc_dropout.pt")
    if "mc_dropout" in args.methods and os.path.isfile(mc_path):
        print("\n[MC Dropout]")
        ckpt = torch.load(mc_path, map_location="cpu")
        mc_m = MCDropoutConvNet(dropout_p=ckpt.get("dropout_p", 0.1))
        mc_m.load_state_dict(ckpt["model_state_dict"])
        eu, probs, labels = predict_mc_dropout(mc_m, test_loader, device, T=args.T)
        baselines["MC Dropout"] = (eu, probs, labels)
        print(f"  EU={eu.mean():.4f}  acc={100*(probs.argmax(-1)==labels).mean():.2f}%")

    # EDL
    edl_path = os.path.join(baseline_dir, "edl_head.pt")
    if "edl" in args.methods and os.path.isfile(edl_path):
        print("\n[EDL]")
        ckpt = torch.load(edl_path, map_location="cpu")
        bb   = mnist_convnet()
        bb.load_state_dict(ckpt["backbone_state_dict"])
        head = EDLHead(feat_dim=FEAT_DIM, num_classes=NUM_CLASSES)
        head.load_state_dict(ckpt["edl_head_state_dict"])
        eu_epi, eu_ale, probs, labels = predict_edl(bb, head, test_loader, device)
        baselines["EDL"] = (eu_epi, probs, labels)
        print(f"  EU_epi={eu_epi.mean():.4f}  EU_ale={eu_ale.mean():.4f}"
              f"  acc={100*(probs.argmax(-1)==labels).mean():.2f}%")

    # LLLA
    la_obj = None
    llla_path   = os.path.join(baseline_dir, "llla.pt")
    llla_bb_path = os.path.join(baseline_dir, "llla_backbone.pt")
    if "llla" in args.methods and os.path.isfile(llla_path):
        print("\n[LLLA]")
        try:
            from laplace import Laplace

            bb_llla = mnist_convnet()
            bb_llla.load_state_dict(
                torch.load(llla_bb_path, map_location="cpu")["backbone_state_dict"])

            class BackboneFC(nn.Module):
                def __init__(self, b):
                    super().__init__(); self.b = b
                def forward(self, x):
                    out = self.b.features(x); out = self.b.avgpool(out)
                    return self.b.fc(out.view(out.size(0), -1))

            wrapped = BackboneFC(bb_llla)
            la_obj = Laplace(wrapped, "classification",
                             subset_of_weights="last_layer",
                             hessian_structure="kron")
            la_obj.load_state_dict(torch.load(llla_path, map_location="cpu"))
            la_obj.model.to(device)
            eu, probs, labels = predict_llla(bb_llla, la_obj, test_loader, device, T=args.T)
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
        sgld_backbone = mnist_convnet()
        sgld_backbone.load_state_dict(ckpt["backbone_state_dict"])
        sgld_samples = ckpt["samples"]
        eu, probs, labels = predict_sgld(sgld_backbone, sgld_samples, test_loader, device)
        baselines["SGLD"] = (eu, probs, labels)
        print(f"  EU={eu.mean():.4f}  acc={100*(probs.argmax(-1)==labels).mean():.2f}%")

    # Student (for comparison)
    student_eu_np = None; student_probs_np = None
    student_path = os.path.join(args.save_dir, "student.pt")
    if os.path.isfile(student_path):
        print("\n[Student for comparison]")
        from evaluate_student import load_student, predict_student
        stu = load_student(args.save_dir, device)
        stu_probs_t, stu_eu_t = predict_student(stu, test_loader, device)
        student_eu_np    = stu_eu_t.numpy()
        student_probs_np = stu_probs_t.numpy()

    if not baselines:
        print("\nNo baselines found. Run train_baselines.py first.")
        return

    # -----------------------------------------------------------------------
    # Summary table: Accuracy + Calibration + EU correlation
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  MNIST Baseline Evaluation Summary")
    print(f"{'='*80}")
    print(f"\n  Accuracy & Calibration (clean MNIST test set)")
    print(f"  {'Method':<30} {'Acc%':>8} {'ECE-15':>8} {'NLL':>8} {'Brier':>8} "
          f"{'EU-Pear':>8} {'EU-Spear':>9}")
    print(f"  {'-'*80}")

    tea_ece    = compute_ece(teacher_probs, true_labels)
    tea_nll    = compute_nll(teacher_probs, true_labels)
    tea_brier  = compute_brier(teacher_probs, true_labels)
    tea_acc    = 100.0 * (teacher_probs.argmax(-1) == true_labels).mean()
    print(f"  {'Teacher (ensemble)':<30} {tea_acc:>8.2f} {tea_ece:>8.4f} "
          f"{tea_nll:>8.4f} {tea_brier:>8.4f} {'—':>8} {'—':>9}")

    if student_probs_np is not None:
        s_ece  = compute_ece(student_probs_np, true_labels)
        s_nll  = compute_nll(student_probs_np, true_labels)
        s_br   = compute_brier(student_probs_np, true_labels)
        s_acc  = 100.0 * (student_probs_np.argmax(-1) == true_labels).mean()
        s_eu_t = torch.from_numpy(student_eu_np).float()
        rp     = pearson_corr(s_eu_t, tea_eu_t)
        rs     = spearman_corr(s_eu_t, tea_eu_t)
        print(f"  {'Student (distilled)':<30} {s_acc:>8.2f} {s_ece:>8.4f} "
              f"{s_nll:>8.4f} {s_br:>8.4f} {rp:>8.4f} {rs:>9.4f}")

    for name, (eu_np, probs_np, _) in baselines.items():
        ece   = compute_ece(probs_np, true_labels)
        nll   = compute_nll(probs_np, true_labels)
        brier = compute_brier(probs_np, true_labels)
        acc   = 100.0 * (probs_np.argmax(-1) == true_labels).mean()
        eu_t  = torch.from_numpy(eu_np).float()
        rp    = pearson_corr(eu_t, tea_eu_t)
        rs    = spearman_corr(eu_t, tea_eu_t)
        print(f"  {name:<30} {acc:>8.2f} {ece:>8.4f} {nll:>8.4f} "
              f"{brier:>8.4f} {rp:>8.4f} {rs:>9.4f}")

    # -----------------------------------------------------------------------
    # OOD Detection AUROC
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  OOD Detection AUROC (ID = Clean MNIST test)")
    print(f"{'='*80}")

    all_ood_items = list(SEEN_OOD) + list(UNSEEN_OOD)
    method_names  = (["Teacher EU", "Student EU"] if student_probs_np is not None
                     else ["Teacher EU"])
    method_names += [n for n in baselines]

    col_w = 12
    hdr = f"  {'OOD Dataset':<22} {'Type':<8}"
    for m in method_names:
        hdr += f" {m[:col_w]:>{col_w}}"
    print(hdr); print(f"  {'-'*80}")

    for ood_id, cache_key, display_name in all_ood_items:
        if cache_key not in npz:
            continue
        ood_type = "seen" if (ood_id, cache_key, display_name) in SEEN_OOD else "unseen"
        try:
            ood_loader = get_ood_loader(ood_id, args.data_dir,
                                         args.batch_size, args.num_workers)
        except Exception as e:
            print(f"  {display_name:<22} SKIPPED ({e})")
            continue

        ood_tea_eu = npz[cache_key]
        row = f"  {display_name:<22} {ood_type:<8}"

        # Teacher
        row += f" {auroc(teacher_eu_np, ood_tea_eu[:len(ood_tea_eu)]):{col_w}.4f}"

        # Student
        if student_probs_np is not None:
            try:
                from evaluate_student import predict_student, load_student
                stu = load_student(args.save_dir, device)
                ood_stu_probs_t, ood_stu_eu_t = predict_student(stu, ood_loader, device)
                row += f" {auroc(student_eu_np, ood_stu_eu_t.numpy()):{col_w}.4f}"
            except Exception:
                row += f" {'N/A':>{col_w}}"

        # Each baseline
        for name, (id_eu, _, _) in baselines.items():
            try:
                if name == "MC Dropout":
                    ckpt = torch.load(os.path.join(baseline_dir, "mc_dropout.pt"),
                                      map_location="cpu")
                    mm = MCDropoutConvNet(dropout_p=ckpt.get("dropout_p", 0.1))
                    mm.load_state_dict(ckpt["model_state_dict"])
                    ood_eu, _, _ = predict_mc_dropout(mm, ood_loader, device, T=args.T)
                elif name == "EDL":
                    ckpt = torch.load(os.path.join(baseline_dir, "edl_head.pt"),
                                      map_location="cpu")
                    bb2  = mnist_convnet()
                    bb2.load_state_dict(ckpt["backbone_state_dict"])
                    hd2  = EDLHead(feat_dim=FEAT_DIM, num_classes=NUM_CLASSES)
                    hd2.load_state_dict(ckpt["edl_head_state_dict"])
                    ood_eu, _, _, _ = predict_edl(bb2, hd2, ood_loader, device)
                elif name == "LLLA (KFAC)" and la_obj is not None:
                    bb3  = mnist_convnet()
                    bb3.load_state_dict(
                        torch.load(llla_bb_path, map_location="cpu")["backbone_state_dict"])
                    ood_eu, _, _ = predict_llla(bb3, la_obj, ood_loader, device, T=args.T)
                elif name == "SGLD" and sgld_backbone is not None:
                    ood_eu, _, _ = predict_sgld(sgld_backbone, sgld_samples,
                                                 ood_loader, device)
                else:
                    row += f" {'N/A':>{col_w}}"
                    continue
                row += f" {auroc(id_eu, ood_eu):{col_w}.4f}"
            except Exception as e:
                row += f" {'N/A':>{col_w}}"
        print(row)

    # -----------------------------------------------------------------------
    # AURC
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  Selective Prediction AURC (clean MNIST test)")
    print(f"{'='*80}")
    print(f"  {'Method':<32} {'AURC↓':>10} {'OracleGap↓':>12} {'@90%↑':>8} {'@80%↑':>8}")
    print(f"  {'-'*72}")

    errors_tea = (teacher_probs.argmax(-1) != true_labels).astype(float)
    _print_aurc("Teacher EU", errors_tea, teacher_eu_np)
    if student_probs_np is not None:
        errors_stu = (student_probs_np.argmax(-1) != true_labels).astype(float)
        _print_aurc("Student EU (distilled)", errors_stu, student_eu_np)
        id_ent = -(student_probs_np * np.log(student_probs_np + EPS)).sum(-1)
        _print_aurc("Student entropy", errors_stu, id_ent)

    for name, (eu_np, probs_np, _) in baselines.items():
        bl_errors = (probs_np.argmax(-1) != true_labels).astype(float)
        _print_aurc(f"{name} EU", bl_errors, eu_np)
        bl_ent = -(probs_np * np.log(probs_np + EPS)).sum(-1)
        _print_aurc(f"{name} entropy", bl_errors, bl_ent)

    # -----------------------------------------------------------------------
    # Throughput
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  Inference Throughput (device={device}, bs=256, 50 batches)")
    print(f"{'='*80}")
    print(f"  {'Method':<38} {'Samples/sec':>14}")
    print(f"  {'-'*54}")

    all_members = []
    for midx in range(20):
        mpath = os.path.join(args.save_dir, f"member_{midx}.pt")
        if not os.path.exists(mpath): break
        mckpt = torch.load(mpath, map_location=device, weights_only=True)
        mem   = mnist_convnet(dropout_rate=mckpt.get("member_config", {}).get("dropout_rate", 0.0))
        mem.load_state_dict(mckpt["model_state_dict"])
        mem.to(device).eval()
        all_members.append(mem)

    if all_members:
        K = len(all_members)
        def _ens(x): return torch.stack([F.softmax(m(x), -1) for m in all_members]).mean(0)
        tp_ens = measure_throughput(_ens, img_shape, device, batch_size=256, n_batches=50)
        tp_sgl = measure_throughput(lambda x: all_members[0](x), img_shape, device,
                                    batch_size=256, n_batches=50)
        print(f"  {'Ensemble (K='+str(K)+')':<38} {tp_ens:>14,.0f}")
        print(f"  {'Single member':<38} {tp_sgl:>14,.0f}")

    if os.path.isfile(student_path):
        from evaluate_student import load_student
        stu = load_student(args.save_dir, device)
        tp_stu = measure_throughput(lambda x: stu(x), img_shape, device,
                                    batch_size=256, n_batches=50)
        print(f"  {'Student (single pass)':<38} {tp_stu:>14,.0f}")

    mc_bl_path = os.path.join(baseline_dir, "mc_dropout.pt")
    if os.path.isfile(mc_bl_path) and "mc_dropout" in args.methods:
        ckpt = torch.load(mc_bl_path, map_location=device)
        mc_m = MCDropoutConvNet(dropout_p=ckpt.get("dropout_p", 0.1)).to(device)
        mc_m.load_state_dict(ckpt["model_state_dict"])
        T = args.T
        mc_m.train()
        tp_mc = measure_throughput(lambda x: _multi_forward(mc_m, x, T),
                                    img_shape, device, batch_size=256, n_batches=50)
        print(f"  {'MC Dropout (T='+str(T)+')':<38} {tp_mc:>14,.0f}")

    if sgld_backbone is not None and sgld_samples is not None:
        sgld_backbone.to(device).eval()
        T = len(sgld_samples)
        ws = [s["weight"].to(device) for s in sgld_samples]
        bs_list = [s["bias"].to(device) for s in sgld_samples]
        dummy = torch.randn(256, *img_shape, device=device)
        with torch.no_grad():
            out  = sgld_backbone.features(dummy)
            feat = sgld_backbone.avgpool(out).view(256, -1)
        def _sgld_fn(x):
            with torch.no_grad():
                o  = sgld_backbone.features(x)
                f2 = sgld_backbone.avgpool(o).view(x.size(0), -1)
            return torch.stack([F.softmax(f2 @ w.T + b, -1) for w, b in zip(ws, bs_list)])
        tp_sg = measure_throughput(_sgld_fn, img_shape, device, batch_size=256, n_batches=50)
        print(f"  {'SGLD (T='+str(T)+')':<38} {tp_sg:>14,.0f}")

    print(f"\nBaseline evaluation complete.")


def _multi_forward(model, x, T):
    return torch.stack([F.softmax(model(x), -1) for _ in range(T)]).mean(0)


def _print_aurc(name, errors, scores):
    a, _, gap, a90, a80 = compute_aurc(errors, np.asarray(scores))
    print(f"  {name:<32} {a:>10.6f} {gap:>12.6f} {a90:>8.4f} {a80:>8.4f}")


if __name__ == "__main__":
    main()
