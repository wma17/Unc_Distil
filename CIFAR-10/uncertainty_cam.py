"""
UncertaintyCAM: Attribution maps for epistemic uncertainty and prediction.

Attribution methods:
- **GradCAM**: Spatial heatmap from layer4 activations (interpretable, fast)
- **Integrated Gradients (IG)**: Axiomatically-grounded pixel attribution; interpolates
  baseline→input and integrates gradients. Better than vanilla gradient saliency.

1. **Teacher EU**: GradCAM or IG of ensemble epistemic uncertainty
2. **Student EU**: GradCAM or IG of student EU head (full attribution: backbone + probs path)
3. **Teacher / Student prediction**: Gradient or IG saliency of predicted-class logit

Output: 5 panels — Original | Teacher EU | Student EU | Teacher Pred | Student Pred

Usage:
    python uncertainty_cam.py --save_dir ./checkpoints --out_dir ./uncertainty_cam --gpu 0
    python uncertainty_cam.py --attribution_method ig --ig_steps 50  # Use Integrated Gradients
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import cifar_resnet18, cifar_resnet18_student
from evaluate_uncertainty import entropy, compute_uncertainty
from evaluate_student import load_student
from cache_ensemble_targets import apply_corruption, CORRUPTION_TYPES, CORRUPTION_SEED


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
EPS = 1e-8

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# OOD datasets: (ood_id, display_name)
OOD_DATASETS = [
    ("svhn", "SVHN"),
    ("cifar100", "CIFAR-100"),
]


def load_ensemble(save_dir, device, num_classes=10):
    """Load ensemble members, skipping corrupted checkpoints."""
    members = []
    ckpt_files = sorted(f for f in os.listdir(save_dir) if f.startswith("member_") and f.endswith(".pt"))
    for fname in ckpt_files:
        path = os.path.join(save_dir, fname)
        if os.path.getsize(path) == 0:
            print(f"  Skipping {fname} (empty file)")
            continue
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
            model = cifar_resnet18(num_classes=num_classes).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            members.append(model)
            print(f"  Loaded {fname}  acc={ckpt.get('test_acc', 0):.2f}%")
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
    if not members:
        raise FileNotFoundError(f"No valid member checkpoints in {save_dir}")
    print(f"Ensemble size: {len(members)}")
    return members


# ---------------------------------------------------------------------------
# Teacher UncertaintyCAM
# ---------------------------------------------------------------------------

def compute_teacher_uncertainty_cam(members, x, device):
    """
    Compute GradCAM for teacher ensemble epistemic uncertainty.

    EU = H(p̄) - (1/M) Σ_m H(p_m). We backprop EU through all members and
    aggregate GradCAM from each member's layer4 activations.

    Args:
        members: list of CIFARResNet ensemble models
        x: (1, 3, 32, 32) input tensor, requires_grad=True
    Returns:
        cam: (32, 32) numpy array, normalized [0, 1]
    """
    M = len(members)
    activations = {}
    gradients = {}

    def make_forward_hook(member_idx):
        def hook(module, input, output):
            activations[member_idx] = output.detach()
        return hook

    def make_backward_hook(member_idx):
        def hook(module, grad_input, grad_output):
            gradients[member_idx] = grad_output[0].detach()
        return hook

    handles = []
    for m, model in enumerate(members):
        h_fwd = model.layer4.register_forward_hook(make_forward_hook(m))
        h_bwd = model.layer4.register_full_backward_hook(make_backward_hook(m))
        handles.append((h_fwd, h_bwd))

    try:
        x = x.to(device)
        if not x.requires_grad:
            x.requires_grad_(True)

        # Forward all members
        all_logits = []
        for model in members:
            logits = model(x)
            all_logits.append(logits)

        # Stack: (1, M, C)
        all_logits = torch.stack(all_logits, dim=1)
        all_probs = F.softmax(all_logits, dim=-1)

        mean_probs = all_probs.mean(dim=1)  # (1, C)
        tu = entropy(mean_probs)  # (1,)
        au = entropy(all_probs).mean(dim=1)  # (1,)
        eu = (tu - au).squeeze()  # scalar

        eu.backward()

        # GradCAM per member, then aggregate
        cam_maps = []
        for m in range(M):
            A = activations[m]   # (1, 512, H, W)
            g = gradients[m]     # (1, 512, H, W)
            if g is None:
                continue
            # Grad-CAM: weights = global average of gradients
            weights = g.mean(dim=(2, 3))  # (1, 512)
            cam = (weights.unsqueeze(-1).unsqueeze(-1) * A).sum(dim=1, keepdim=True)  # (1, 1, H, W)
            cam = F.relu(cam)
            cam_maps.append(cam)

        if not cam_maps:
            return np.zeros((32, 32), dtype=np.float32)

        # Aggregate: average across members (all contribute to EU)
        cam = torch.cat(cam_maps, dim=0).mean(dim=0, keepdim=True)  # (1, 1, H, W)
        cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)  # (1, 1, 32, 32)
        cam = cam.squeeze().cpu().numpy()  # (32, 32)
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)

    finally:
        for h_fwd, h_bwd in handles:
            h_fwd.remove()
            h_bwd.remove()


# ---------------------------------------------------------------------------
# Student UncertaintyCAM
# ---------------------------------------------------------------------------

def _student_forward_no_detach(model, x):
    """Forward pass without detaching probs, so gradients flow through prediction path."""
    out = F.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.layer4(out)
    out = model.avgpool(out)
    feat = out.view(out.size(0), -1)
    logits = model.fc(feat)
    probs = F.softmax(logits, dim=-1)  # NO detach — full gradient flow
    eu_in = torch.cat([feat, probs], dim=-1)
    eu = model.eu_act(model.eu_fc2(F.relu(model.eu_fc1(eu_in)))).squeeze(-1)
    return out, logits, eu


def compute_student_uncertainty_cam(model, x, device, include_probs_path=True):
    """
    Compute GradCAM for student epistemic uncertainty head.

    By default (include_probs_path=True): Full attribution — gradients flow through
    both backbone features AND the prediction (logits→probs) path. The trained
    model uses probs.detach() for Phase 2; here we do a separate forward without
    detach to capture how prediction uncertainty affects EU.

    If include_probs_path=False: Only backbone path (matches trained forward).

    Args:
        model: CIFARResNetStudent
        x: (1, 3, 32, 32) input tensor, requires_grad=True
        include_probs_path: if True, probs not detached so prediction path contributes
    Returns:
        cam: (32, 32) numpy array, normalized [0, 1]
    """
    activation = None
    gradient = None

    def fwd_hook(module, input, output):
        nonlocal activation
        activation = output.detach()

    def bwd_hook(module, grad_input, grad_output):
        nonlocal gradient
        gradient = grad_output[0].detach()

    h_fwd = model.layer4.register_forward_hook(fwd_hook)
    h_bwd = model.layer4.register_full_backward_hook(bwd_hook)

    try:
        x = x.to(device)
        if not x.requires_grad:
            x.requires_grad_(True)

        if include_probs_path:
            _, _, eu = _student_forward_no_detach(model, x)
        else:
            logits, eu = model(x)
        eu = eu.squeeze()
        eu.backward()

        if activation is None or gradient is None:
            return np.zeros((32, 32), dtype=np.float32)

        A = activation   # (1, 512, H, W)
        g = gradient     # (1, 512, H, W)
        weights = g.mean(dim=(2, 3))
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)

    finally:
        h_fwd.remove()
        h_bwd.remove()


# ---------------------------------------------------------------------------
# Traditional saliency: gradient of predicted-class logit w.r.t. input
# ---------------------------------------------------------------------------

def compute_teacher_prediction_saliency(members, x, pred_class, device):
    """Saliency for teacher: gradient of mean logit for pred_class w.r.t. input."""
    x = x.to(device)
    if not x.requires_grad:
        x.requires_grad_(True)
    all_logits = torch.stack([m(x) for m in members], dim=1)  # (1, M, C)
    mean_logits = all_logits.mean(dim=1)  # (1, C)
    score = mean_logits[0, pred_class]
    score.backward()
    sal = x.grad.abs().sum(dim=1).squeeze(0).cpu().numpy()
    if sal.max() > 0:
        sal = sal / sal.max()
    return sal.astype(np.float32)


def compute_student_prediction_saliency(model, x, pred_class, device):
    """Saliency for student: gradient of logit for pred_class w.r.t. input."""
    x = x.to(device)
    if not x.requires_grad:
        x.requires_grad_(True)
    logits, _ = model(x)
    score = logits[0, pred_class]
    score.backward()
    sal = x.grad.abs().sum(dim=1).squeeze(0).cpu().numpy()
    if sal.max() > 0:
        sal = sal / sal.max()
    return sal.astype(np.float32)


# ---------------------------------------------------------------------------
# Integrated Gradients (Sundararajan et al.)
# ---------------------------------------------------------------------------

def get_baseline(image, baseline_type="black"):
    """Baseline for IG: black image in normalized CIFAR space."""
    if baseline_type == "black":
        bl = torch.zeros_like(image, device=image.device)
        for c in range(3):
            bl[:, c, :, :] = -CIFAR10_MEAN[c] / CIFAR10_STD[c]
        return bl
    elif baseline_type == "zero":
        return torch.zeros_like(image, device=image.device)
    return torch.zeros_like(image, device=image.device)


def integrated_gradients(scalar_fn, image, baseline, device, steps=50):
    """
    Integrated Gradients: attribution of scalar_fn(x) w.r.t. input.
    IG(x) = (x - baseline) * integral_0^1 grad_scalar_fn(baseline + a*(x-baseline)) da

    Args:
        scalar_fn: callable(x) -> scalar tensor
        image: (1, C, H, W)
        baseline: (1, C, H, W)
        device, steps
    Returns:
        attr: (H, W) numpy, normalized [0, 1]
    """
    accumulated = torch.zeros_like(image, device=device)
    for step in range(steps + 1):
        alpha = step / steps
        interp = baseline + alpha * (image - baseline)
        interp = interp.detach().requires_grad_(True)
        out = scalar_fn(interp)
        out.backward()
        if interp.grad is not None:
            accumulated += interp.grad
    ig = (image - baseline) * accumulated / steps
    sal = ig.abs().sum(dim=1).squeeze(0).cpu().numpy()
    if sal.max() > 0:
        sal = sal / sal.max()
    return sal.astype(np.float32)


def compute_teacher_eu_ig(members, x, device, baseline=None, steps=50):
    """IG of teacher ensemble EU w.r.t. input."""
    def fn(x_in):
        logits = torch.stack([m(x_in) for m in members], dim=1)
        probs = F.softmax(logits, dim=-1)
        mean_p = probs.mean(dim=1)
        tu = entropy(mean_p)
        au = entropy(probs).mean(dim=1)
        return (tu - au).squeeze()
    x = x.to(device)
    baseline = baseline if baseline is not None else get_baseline(x).to(device)
    return integrated_gradients(fn, x, baseline, device, steps)


def compute_student_eu_ig(model, x, device, baseline=None, steps=50):
    """IG of student EU w.r.t. input (full path, no detach)."""
    def fn(x_in):
        _, _, eu = _student_forward_no_detach(model, x_in)
        return eu.squeeze()
    x = x.to(device)
    baseline = baseline if baseline is not None else get_baseline(x).to(device)
    return integrated_gradients(fn, x, baseline, device, steps)


def compute_teacher_pred_ig(members, x, pred_class, device, baseline=None, steps=50):
    """IG of teacher mean logit for pred_class w.r.t. input."""
    def fn(x_in):
        logits = torch.stack([m(x_in) for m in members], dim=1).mean(dim=1)
        return logits[0, pred_class]
    x = x.to(device)
    baseline = baseline if baseline is not None else get_baseline(x).to(device)
    return integrated_gradients(fn, x, baseline, device, steps)


def compute_student_pred_ig(model, x, pred_class, device, baseline=None, steps=50):
    """IG of student logit for pred_class w.r.t. input."""
    def fn(x_in):
        logits, _ = model(x_in)
        return logits[0, pred_class]
    x = x.to(device)
    baseline = baseline if baseline is not None else get_baseline(x).to(device)
    return integrated_gradients(fn, x, baseline, device, steps)


# ---------------------------------------------------------------------------
# Data & visualization
# ---------------------------------------------------------------------------

def denormalize(img_tensor):
    """Convert normalized CIFAR tensor to [0,1] RGB."""
    img = img_tensor.cpu().numpy()
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    mean = np.array(CIFAR10_MEAN)
    std = np.array(CIFAR10_STD)
    img = img * std + mean
    return np.clip(img, 0, 1)


def select_cases_by_uncertainty(teacher_eu, n_cases=100):
    """
    Select n_cases indices spanning the full range of epistemic uncertainty.

    Rank by teacher EU, then sample evenly across percentiles.
    """
    ranked_idx = np.argsort(teacher_eu)  # ranked_idx[i] = original index of i-th smallest EU
    n = len(ranked_idx)
    if n_cases >= n:
        return ranked_idx
    # Evenly spaced positions in sorted order (0, n/(n_cases-1), 2*n/(n_cases-1), ..., n-1)
    positions = np.linspace(0, n - 1, n_cases, dtype=int)
    return ranked_idx[positions]


def plot_single_case(original_img, teacher_cam, student_cam, teacher_sal, student_sal,
                    teacher_eu, student_eu, pred_teacher, pred_student,
                    save_path, true_label_str=None, method_label="GradCAM"):
    """Create one figure: Original | Teacher EU | Student EU | Teacher Pred | Student Pred."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))

    # Original
    axes[0].imshow(original_img)
    orig_title = "Original"
    if true_label_str:
        orig_title += f"\n{true_label_str}"
    axes[0].set_title(orig_title, fontsize=9)
    axes[0].axis("off")

    # Teacher EU
    axes[1].imshow(original_img)
    axes[1].imshow(teacher_cam, cmap="jet", alpha=0.5)
    axes[1].set_title(f"Teacher EU ({method_label})\nEU={teacher_eu:.4f} Pred:{CIFAR10_CLASSES[pred_teacher]}", fontsize=9)
    axes[1].axis("off")

    # Student EU
    axes[2].imshow(original_img)
    axes[2].imshow(student_cam, cmap="jet", alpha=0.5)
    axes[2].set_title(f"Student EU ({method_label})\nEU={student_eu:.4f} Pred:{CIFAR10_CLASSES[pred_student]}", fontsize=9)
    axes[2].axis("off")

    # Teacher prediction
    axes[3].imshow(original_img)
    axes[3].imshow(teacher_sal, cmap="hot", alpha=0.5)
    axes[3].set_title(f"Teacher Pred ({method_label})\nlogit[{CIFAR10_CLASSES[pred_teacher]}]", fontsize=9)
    axes[3].axis("off")

    # Student prediction
    axes[4].imshow(original_img)
    axes[4].imshow(student_sal, cmap="hot", alpha=0.5)
    axes[4].set_title(f"Student Pred ({method_label})\nlogit[{CIFAR10_CLASSES[pred_student]}]", fontsize=9)
    axes[4].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def run_dataset_cam(members, student, dataset_name, all_images, teacher_eu, student_eu_all,
                    pred_teacher, pred_student, label_str_fn, out_dir, n_cases, device,
                    attribution_method="gradcam", ig_steps=50):
    """
    Run attribution for one dataset.
    attribution_method: "gradcam" or "ig"
    """
    indices = select_cases_by_uncertainty(teacher_eu, n_cases=n_cases)
    print(f"  {dataset_name}: {len(indices)} cases, method={attribution_method}"
          f"{f', ig_steps={ig_steps}' if attribution_method == 'ig' else ''}")

    ds_out = os.path.join(out_dir, dataset_name)
    os.makedirs(ds_out, exist_ok=True)
    summary_rows = []
    use_ig = attribution_method.lower() == "ig"
    method_label = "IG" if use_ig else "GradCAM"

    for idx, glob_idx in enumerate(indices):
        pt, ps = int(pred_teacher[glob_idx]), int(pred_student[glob_idx])
        x = all_images[glob_idx : glob_idx + 1].clone().to(device)

        if use_ig:
            teacher_cam = compute_teacher_eu_ig(members, x, device, steps=ig_steps)
            x2 = all_images[glob_idx : glob_idx + 1].clone().to(device)
            student_cam = compute_student_eu_ig(student, x2, device, steps=ig_steps)
            x3 = all_images[glob_idx : glob_idx + 1].clone().to(device)
            teacher_sal = compute_teacher_pred_ig(members, x3, pt, device, steps=ig_steps)
            x4 = all_images[glob_idx : glob_idx + 1].clone().to(device)
            student_sal = compute_student_pred_ig(student, x4, ps, device, steps=ig_steps)
        else:
            x_t = x.clone()
            x_t.requires_grad_(True)
            teacher_cam = compute_teacher_uncertainty_cam(members, x_t, device)
            x_s = all_images[glob_idx : glob_idx + 1].clone().to(device)
            x_s.requires_grad_(True)
            student_cam = compute_student_uncertainty_cam(student, x_s, device, include_probs_path=True)
            x_ts = all_images[glob_idx : glob_idx + 1].clone().to(device)
            x_ts.requires_grad_(True)
            teacher_sal = compute_teacher_prediction_saliency(members, x_ts, pt, device)
            x_ss = all_images[glob_idx : glob_idx + 1].clone().to(device)
            x_ss.requires_grad_(True)
            student_sal = compute_student_prediction_saliency(student, x_ss, ps, device)

        orig = denormalize(all_images[glob_idx])
        fname = f"uncertainty_cam_{idx:03d}.png"
        true_str = label_str_fn(glob_idx) if label_str_fn else None
        plot_single_case(
            orig, teacher_cam, student_cam, teacher_sal, student_sal,
            float(teacher_eu[glob_idx]), float(student_eu_all[glob_idx]),
            pt, ps,
            os.path.join(ds_out, fname), true_label_str=true_str, method_label=method_label,
        )

        true_lab = label_str_fn(glob_idx) if label_str_fn else "OOD"
        summary_rows.append([
            idx, glob_idx, f"{teacher_eu[glob_idx]:.6f}", f"{student_eu_all[glob_idx]:.6f}",
            true_lab,
            CIFAR10_CLASSES[int(pred_teacher[glob_idx])],
            CIFAR10_CLASSES[int(pred_student[glob_idx])],
            fname,
        ])

        if (idx + 1) % 20 == 0:
            print(f"    Saved {idx + 1}/{len(indices)}")

    summary_path = os.path.join(ds_out, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "idx", "teacher_eu", "student_eu", "true_label", "pred_teacher", "pred_student", "filename"])
        writer.writerows(summary_rows)
    return len(indices)


def main():
    parser = argparse.ArgumentParser(description="UncertaintyCAM visualization")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--out_dir", type=str, default="./uncertainty_cam")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_per_dataset", type=int, default=35,
                        help="Number of cases per dataset (ID, each shifted, each OOD)")
    parser.add_argument("--datasets", type=str, nargs="+", default=["id", "shifted", "ood"],
                        help="Which datasets to run: id, shifted, ood (run separately if memory constrained)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for uncertainty computation (smaller = less GPU memory)")
    parser.add_argument("--attribution_method", type=str, default="ig", choices=["gradcam", "ig"],
                        help="Attribution: gradcam (fast) or ig (Integrated Gradients, better)")
    parser.add_argument("--ig_steps", type=int, default=50, help="Integration steps for IG (higher=more accurate, slower)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Attribution method: {args.attribution_method}" + (f" (steps={args.ig_steps})" if args.attribution_method == "ig" else ""))

    # Load models
    print("Loading ensemble (teacher)...")
    members = load_ensemble(args.save_dir, device)
    print("Loading student...")
    student = load_student(args.save_dir, device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    rgb_ood_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    total_saved = 0

    # ---- ID: Clean CIFAR-10 test ----
    if "id" in args.datasets:
        print("\n--- ID (Clean CIFAR-10 test) ---")
        test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        all_images, all_labels = [], []
        all_member_probs_list = []
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            all_images.append(inputs.cpu())
            all_labels.append(targets.cpu())
            with torch.no_grad():
                batch_probs = torch.stack([F.softmax(m(inputs), dim=-1).cpu() for m in members], dim=0).permute(1, 0, 2)
            all_member_probs_list.append(batch_probs)

        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_member_probs = torch.cat(all_member_probs_list, dim=0)
        unc = compute_uncertainty(all_member_probs)
        teacher_eu = unc["epistemic_uncertainty"].numpy()
        pred_teacher = unc["pred_labels"].numpy()

        student_eu_list, student_pred_list = [], []
        for i in range(0, len(all_images), args.batch_size):
            batch = all_images[i : i + args.batch_size].to(device)
            with torch.no_grad():
                logits, eu = student(batch)
            student_eu_list.append(eu.cpu())
            student_pred_list.append(logits.argmax(dim=1).cpu())
        student_eu_all = torch.cat(student_eu_list).numpy()
        student_pred_all = torch.cat(student_pred_list).numpy()

        def label_fn(i):
            return f"True: {CIFAR10_CLASSES[int(all_labels[i])]}"

        n = run_dataset_cam(
            members, student, "id",
            all_images, teacher_eu, student_eu_all, pred_teacher, student_pred_all,
            label_fn, args.out_dir, args.n_per_dataset, device,
            attribution_method=args.attribution_method, ig_steps=args.ig_steps,
        )
        total_saved += n

    # ---- Shifted ID: Corrupted CIFAR-10 ----
    if "shifted" in args.datasets:
        test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        clean_imgs = torch.cat([x for x, _ in test_loader], 0)
        clean_labels = torch.cat([y for _, y in test_loader], 0)

        for ctype in CORRUPTION_TYPES:
            print(f"\n--- Shifted ID ({ctype}) ---")
            corrupted = apply_corruption(clean_imgs, ctype, seed=CORRUPTION_SEED)

            all_member_probs_list = []
            for model in members:
                probs_list = []
                for i in range(0, len(corrupted), args.batch_size):
                    batch = corrupted[i : i + args.batch_size].to(device)
                    with torch.no_grad():
                        probs_list.append(F.softmax(model(batch), dim=-1).cpu())
                all_member_probs_list.append(torch.cat(probs_list, dim=0))
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            all_member_probs = torch.stack(all_member_probs_list, dim=0).permute(1, 0, 2)
            unc = compute_uncertainty(all_member_probs)
            teacher_eu = unc["epistemic_uncertainty"].numpy()
            pred_teacher = unc["pred_labels"].numpy()

            student_eu_list, student_pred_list = [], []
            for i in range(0, len(corrupted), args.batch_size):
                batch = corrupted[i : i + args.batch_size].to(device)
                with torch.no_grad():
                    logits, eu = student(batch)
                student_eu_list.append(eu.cpu())
                student_pred_list.append(logits.argmax(dim=1).cpu())
            student_eu_all = torch.cat(student_eu_list).numpy()
            student_pred_all = torch.cat(student_pred_list).numpy()

            def label_fn(i):
                return f"True: {CIFAR10_CLASSES[int(clean_labels[i])]}"

            dataset_name = f"shifted_{ctype}"
            n = run_dataset_cam(
                members, student, dataset_name,
                corrupted, teacher_eu, student_eu_all, pred_teacher, student_pred_all,
                label_fn, args.out_dir, args.n_per_dataset, device,
                attribution_method=args.attribution_method, ig_steps=args.ig_steps,
            )
            total_saved += n

    # ---- OOD: SVHN, CIFAR-100 ----
    if "ood" in args.datasets:
        for ood_id, display_name in OOD_DATASETS:
            print(f"\n--- OOD ({display_name}) ---")
            if ood_id == "svhn":
                ood_set = datasets.SVHN(root=os.path.join(args.data_dir, "svhn"), split="test",
                                        download=True, transform=rgb_ood_transform)
            elif ood_id == "cifar100":
                ood_set = datasets.CIFAR100(root=args.data_dir, train=False, download=True,
                                            transform=rgb_ood_transform)
            else:
                continue

            ood_loader = DataLoader(ood_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            all_images = torch.cat([x for x, _ in ood_loader], 0)

            all_member_probs_list = []
            for model in members:
                probs_list = []
                for i in range(0, len(all_images), args.batch_size):
                    batch = all_images[i : i + args.batch_size].to(device)
                    with torch.no_grad():
                        probs_list.append(F.softmax(model(batch), dim=-1).cpu())
                all_member_probs_list.append(torch.cat(probs_list, dim=0))
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            all_member_probs = torch.stack(all_member_probs_list, dim=0).permute(1, 0, 2)
            unc = compute_uncertainty(all_member_probs)
            teacher_eu = unc["epistemic_uncertainty"].numpy()
            pred_teacher = unc["pred_labels"].numpy()

            student_eu_list, student_pred_list = [], []
            for i in range(0, len(all_images), args.batch_size):
                batch = all_images[i : i + args.batch_size].to(device)
                with torch.no_grad():
                    logits, eu = student(batch)
                student_eu_list.append(eu.cpu())
                student_pred_list.append(logits.argmax(dim=1).cpu())
            student_eu_all = torch.cat(student_eu_list).numpy()
            student_pred_all = torch.cat(student_pred_list).numpy()

            def label_fn(i):
                return f"OOD ({display_name})"

            dataset_name = f"ood_{ood_id}"
            n = run_dataset_cam(
                members, student, dataset_name,
                all_images, teacher_eu, student_eu_all, pred_teacher, student_pred_all,
                label_fn, args.out_dir, args.n_per_dataset, device,
                attribution_method=args.attribution_method, ig_steps=args.ig_steps,
            )
            total_saved += n

    print(f"\nDone! Saved {total_saved} UncertaintyCAM figures to {args.out_dir}")


if __name__ == "__main__":
    main()
