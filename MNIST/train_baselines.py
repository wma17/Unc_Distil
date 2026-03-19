"""
Train BNN baselines for MNIST uncertainty quantification.

All baselines start from member_0.pt (first ensemble member).

  1. MC Dropout  — Dropout2d after block1 (pre-MaxPool) and block2; fine-tune 20 epochs.
  2. EDL         — Replace FC with Dirichlet head; fine-tune head only 50 epochs.
  3. LLLA        — Last-layer Laplace (KFAC) via laplace-torch; no training.
  4. SGLD        — Last-layer SGLD; 200-step burn-in + 16 samples (thinning 10).

Usage:
    python train_baselines.py --save_dir ./checkpoints --gpu 1
    python train_baselines.py --save_dir ./checkpoints --gpu 1 --methods mc_dropout edl
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import mnist_convnet


MNIST_MEAN  = (0.1307, 0.1307, 0.1307)
MNIST_STD   = (0.3081, 0.3081, 0.3081)
NUM_CLASSES = 10
FEAT_DIM    = 64
EPS         = 1e-8


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _train_tfm():
    return transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])


def _test_tfm():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])


def get_train_loader(data_dir, batch_size, num_workers=4):
    ds = datasets.MNIST(data_dir, train=True, download=False, transform=_train_tfm())
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)


def get_test_loader(data_dir, batch_size, num_workers=4):
    ds = datasets.MNIST(data_dir, train=False, download=False, transform=_test_tfm())
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# 1. MC Dropout model
# ---------------------------------------------------------------------------

class MCDropoutConvNet(nn.Module):
    """MNISTConvNet with Dropout2d injected after each MaxPool block.

    The original `features` Sequential (14 children, indices 0–13) is split:
      features1: indices 0–6 (conv+bn+relu + conv+bn+relu + MaxPool)
      drop1:     Dropout2d(p)
      features2: indices 7–13 (conv+bn+relu + conv+bn+relu + MaxPool)
      drop2:     Dropout2d(p)

    This matches the recommended spatial MC Dropout placement — after each
    resolution-reducing block so feature maps are regularised before the
    next block.
    """

    def __init__(self, dropout_p=0.1, num_classes=10):
        super().__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.drop1 = nn.Dropout2d(p=dropout_p)
        self.features2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.drop2     = nn.Dropout2d(p=dropout_p)
        self.avgpool   = nn.AdaptiveAvgPool2d((1, 1))
        self.fc        = nn.Linear(FEAT_DIM, num_classes)

    def forward(self, x):
        out = self.drop1(self.features1(x))
        out = self.drop2(self.features2(out))
        out = self.avgpool(out)
        return self.fc(out.view(out.size(0), -1))


def load_mc_dropout_from_member(member_path, dropout_p=0.1):
    """Build MCDropoutConvNet and copy weights from standard MNISTConvNet ckpt.

    Key mapping:
        features.N  → features1.N  if N < 7
        features.N  → features2.(N-7)  if N >= 7
    """
    mc_model = MCDropoutConvNet(dropout_p=dropout_p)
    ckpt = torch.load(member_path, map_location="cpu")
    src  = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    dst_state = mc_model.state_dict()
    mapped = {}
    for k, v in src.items():
        if k.startswith("features."):
            rest = k[len("features."):]
            idx_str = rest.split(".")[0]
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            suffix = rest[len(idx_str):]
            if idx < 7:
                new_key = f"features1.{idx}{suffix}"
            else:
                new_key = f"features2.{idx - 7}{suffix}"
            if new_key in dst_state and dst_state[new_key].shape == v.shape:
                mapped[new_key] = v
        elif k in dst_state and dst_state[k].shape == v.shape:
            mapped[k] = v

    missing = set(dst_state.keys()) - set(mapped.keys())
    if missing:
        print(f"  [MC Dropout] Unmatched keys (random init): {missing}")
    dst_state.update(mapped)
    mc_model.load_state_dict(dst_state)
    return mc_model


def train_mc_dropout(mc_model, train_loader, device, epochs=20, lr=0.01):
    """Fine-tune MC Dropout model (train mode enables Dropout2d)."""
    mc_model.to(device)
    optimizer = optim.SGD(mc_model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        mc_model.train()
        correct, total = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            F.cross_entropy(mc_model(imgs), labels).backward()
            optimizer.step()
            with torch.no_grad():
                correct += mc_model(imgs).argmax(1).eq(labels).sum().item()
                total   += imgs.size(0)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [MC Dropout] epoch {epoch}/{epochs}  acc={100*correct/total:.1f}%")
    return mc_model


# ---------------------------------------------------------------------------
# 2. EDL head (identical structure to CIFAR-10 version)
# ---------------------------------------------------------------------------

class EDLHead(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, hidden=64, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, feat):
        evidence = F.softplus(self.fc2(F.relu(self.fc1(feat))))
        return evidence + 1.0  # α


def edl_loss(alpha, labels, num_classes, lambda_kl=0.1):
    S   = alpha.sum(dim=-1, keepdim=True)
    p   = alpha / S
    K   = num_classes
    y   = F.one_hot(labels, K).float()

    l_err = ((y - p) ** 2).sum(dim=-1)
    l_var = (p * (1.0 - p) / (S + 1.0)).sum(dim=-1)

    alpha_tilde = y + (1.0 - y) * alpha
    S_tilde = alpha_tilde.sum(dim=-1)
    kl = (torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(float(K)))
          - torch.lgamma(alpha_tilde).sum(dim=-1)
          + ((alpha_tilde - 1.0) * (
              torch.digamma(alpha_tilde) - torch.digamma(S_tilde.unsqueeze(-1))
          )).sum(dim=-1))

    return (l_err + l_var + lambda_kl * kl).mean()


def extract_features(backbone, loader, device):
    """Run backbone in eval/no_grad and return (features, labels)."""
    backbone.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out  = backbone.features(imgs)
            out  = backbone.avgpool(out)
            feat = out.view(out.size(0), -1)
            all_feats.append(feat.cpu())
            all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


def train_edl(backbone, train_loader, device, epochs=50,
              lr=1e-3, lambda_kl=0.1, hidden=64):
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    edl_head = EDLHead(feat_dim=FEAT_DIM, hidden=hidden,
                       num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(edl_head.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("  [EDL] Extracting backbone features...")
    feats, labels_all = extract_features(backbone, train_loader, device)
    feat_loader = DataLoader(
        torch.utils.data.TensorDataset(feats, labels_all),
        batch_size=256, shuffle=True, num_workers=0)

    for epoch in range(1, epochs + 1):
        edl_head.train()
        total_loss, correct, total = 0.0, 0, 0
        for feat, labels in feat_loader:
            feat, labels = feat.to(device), labels.to(device)
            optimizer.zero_grad()
            alpha = edl_head(feat)
            loss  = edl_loss(alpha, labels, NUM_CLASSES, lambda_kl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feat.size(0)
            correct    += alpha.argmax(1).eq(labels).sum().item()
            total      += feat.size(0)
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [EDL] epoch {epoch}/{epochs}  loss={total_loss/total:.4f}"
                  f"  acc={100*correct/total:.1f}%")
    return edl_head


# ---------------------------------------------------------------------------
# 3. LLLA — KFAC
# ---------------------------------------------------------------------------

def train_llla(backbone, train_loader, device):
    try:
        from laplace import Laplace
    except ImportError:
        print("  [LLLA] laplace-torch not installed.")
        return None

    class BackboneFC(nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b
        def forward(self, x):
            out = self.b.features(x)
            out = self.b.avgpool(out)
            return self.b.fc(out.view(out.size(0), -1))

    backbone.to(device).eval()
    wrapped = BackboneFC(backbone)

    print("  [LLLA] Fitting KFAC Laplace on last layer...")
    la = Laplace(wrapped, "classification",
                 subset_of_weights="last_layer",
                 hessian_structure="kron")
    la.fit(train_loader)
    print("  [LLLA] Optimizing prior precision...")
    la.optimize_prior_precision(method="marglik")
    print(f"  [LLLA] Prior precision: {la.prior_precision.item():.4f}")
    return la


# ---------------------------------------------------------------------------
# 4. SGLD — last layer
# ---------------------------------------------------------------------------

def train_sgld(backbone, train_loader, device,
               step_size=1e-5, burn_in=200, n_samples=16, thin=10,
               prior_sigma=1.0):
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    fc_weight = backbone.fc.weight.detach().clone().requires_grad_(True)
    fc_bias   = backbone.fc.bias.detach().clone().requires_grad_(True)
    params    = [fc_weight, fc_bias]

    N = len(train_loader.dataset)
    total_steps = burn_in + n_samples * thin
    data_iter   = iter(train_loader)
    samples     = []
    t0 = time.time()

    for step in range(total_steps):
        try:
            imgs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            imgs, labels = next(data_iter)

        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            out  = backbone.features(imgs)
            out  = backbone.avgpool(out)
            feat = out.view(out.size(0), -1)

        logits   = feat @ fc_weight.T + fc_bias
        loss_ce  = F.cross_entropy(logits, labels)
        loss_reg = (fc_weight.pow(2).sum() + fc_bias.pow(2).sum()) / (2 * prior_sigma ** 2)
        U = N * loss_ce + loss_reg

        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        U.backward()

        with torch.no_grad():
            for p in params:
                noise = torch.randn_like(p) * (step_size ** 0.5)
                p.data -= 0.5 * step_size * p.grad + noise

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append({
                "weight": fc_weight.detach().cpu().clone(),
                "bias":   fc_bias.detach().cpu().clone(),
            })

        if step % 50 == 0:
            print(f"  [SGLD] step {step}/{total_steps}  "
                  f"U={U.item():.3f}  samples_collected={len(samples)}")

    print(f"  [SGLD] Done. {len(samples)} samples in {time.time()-t0:.1f}s")
    return samples


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
    p.add_argument("--methods", nargs="+",
                   default=["mc_dropout", "edl", "llla", "sgld"],
                   choices=["mc_dropout", "edl", "llla", "sgld"])
    p.add_argument("--mc_epochs",    type=int,   default=20)
    p.add_argument("--mc_lr",        type=float, default=0.01)
    p.add_argument("--mc_dropout_p", type=float, default=0.1)
    p.add_argument("--edl_epochs",   type=int,   default=50)
    p.add_argument("--edl_lr",       type=float, default=1e-3)
    p.add_argument("--edl_lambda_kl",type=float, default=0.1)
    p.add_argument("--sgld_step",    type=float, default=1e-5)
    p.add_argument("--sgld_burn_in", type=int,   default=200)
    p.add_argument("--sgld_samples", type=int,   default=16)
    p.add_argument("--sgld_thin",    type=int,   default=10)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    baseline_dir = os.path.join(args.save_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)

    member0_path = os.path.join(args.save_dir, "member_0.pt")
    if not os.path.isfile(member0_path):
        raise FileNotFoundError(f"member_0.pt not found in {args.save_dir}")

    train_loader = get_train_loader(args.data_dir, args.batch_size, args.num_workers)

    # ---- MC Dropout ----
    if "mc_dropout" in args.methods:
        print("\n=== MC Dropout ===")
        mc_model = load_mc_dropout_from_member(member0_path, args.mc_dropout_p)
        mc_model  = train_mc_dropout(mc_model, train_loader, device,
                                     epochs=args.mc_epochs, lr=args.mc_lr)
        out_path = os.path.join(baseline_dir, "mc_dropout.pt")
        torch.save({"model_state_dict": mc_model.cpu().state_dict(),
                    "dropout_p": args.mc_dropout_p}, out_path)
        print(f"  Saved -> {out_path}")

    # ---- EDL ----
    if "edl" in args.methods:
        print("\n=== Evidential Deep Learning ===")
        ckpt     = torch.load(member0_path, map_location="cpu")
        backbone = mnist_convnet()
        src      = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        backbone.load_state_dict(src)

        edl_head = train_edl(backbone, train_loader, device,
                             epochs=args.edl_epochs, lr=args.edl_lr,
                             lambda_kl=args.edl_lambda_kl)
        out_path = os.path.join(baseline_dir, "edl_head.pt")
        torch.save({
            "backbone_state_dict": backbone.cpu().state_dict(),
            "edl_head_state_dict": edl_head.cpu().state_dict(),
        }, out_path)
        print(f"  Saved -> {out_path}")

    # ---- LLLA ----
    if "llla" in args.methods:
        print("\n=== Last-Layer Laplace (KFAC) ===")
        ckpt     = torch.load(member0_path, map_location="cpu")
        backbone = mnist_convnet()
        src      = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        backbone.load_state_dict(src)

        la = train_llla(backbone, train_loader, device)
        if la is not None:
            out_path = os.path.join(baseline_dir, "llla.pt")
            torch.save(la.state_dict(), out_path)
            torch.save({"backbone_state_dict": backbone.cpu().state_dict()},
                       os.path.join(baseline_dir, "llla_backbone.pt"))
            print(f"  Saved -> {out_path}")

    # ---- SGLD ----
    if "sgld" in args.methods:
        print("\n=== SGLD (last-layer) ===")
        ckpt     = torch.load(member0_path, map_location="cpu")
        backbone = mnist_convnet()
        src      = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        backbone.load_state_dict(src)

        samples = train_sgld(backbone, train_loader, device,
                             step_size=args.sgld_step,
                             burn_in=args.sgld_burn_in,
                             n_samples=args.sgld_samples,
                             thin=args.sgld_thin)
        out_path = os.path.join(baseline_dir, "sgld_samples.pt")
        torch.save({
            "backbone_state_dict": backbone.cpu().state_dict(),
            "samples": samples,
            "n_samples": len(samples),
        }, out_path)
        print(f"  Saved -> {out_path}  ({len(samples)} samples)")

    print(f"\nAll done. Baselines saved in {baseline_dir}/")


if __name__ == "__main__":
    main()
