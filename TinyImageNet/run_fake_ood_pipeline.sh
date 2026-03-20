#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/home/maw6/maw6/unc_regression/data}"
SAVE_DIR="${SAVE_DIR:-$ROOT/checkpoints_fake_ood_ft}"
GPU="${GPU:-0}"

NUM_MEMBERS="${NUM_MEMBERS:-16}"
ENSEMBLE_EPOCHS="${ENSEMBLE_EPOCHS:-40}"
ENSEMBLE_BATCH="${ENSEMBLE_BATCH:-64}"
DISTILL_BATCH="${DISTILL_BATCH:-32}"
WORKERS="${WORKERS:-4}"
FAKE_OOD_MIXUP_FRAC="${FAKE_OOD_MIXUP_FRAC:-0.5}"
FAKE_OOD_PATCHSHUFFLE_FRAC="${FAKE_OOD_PATCHSHUFFLE_FRAC:-0.0}"
FAKE_OOD_CUTPASTE_FRAC="${FAKE_OOD_CUTPASTE_FRAC:-0.0}"
BACKBONE_LR_FACTOR="${BACKBONE_LR_FACTOR:-0.01}"
DISTILL_P1_EPOCHS="${DISTILL_P1_EPOCHS:-50}"
DISTILL_P1_LR="${DISTILL_P1_LR:-1e-4}"
DISTILL_P1_WD="${DISTILL_P1_WD:-0.05}"
DISTILL_WARMUP="${DISTILL_WARMUP:-5}"
DISTILL_P1_UNFREEZE_BLOCKS="${DISTILL_P1_UNFREEZE_BLOCKS:-2}"
DISTILL_P1_BACKBONE_LR_FACTOR="${DISTILL_P1_BACKBONE_LR_FACTOR:-0.01}"
DISTILL_LLRD="${DISTILL_LLRD:-0.75}"
DISTILL_P1_AUGMENTATIONS="${DISTILL_P1_AUGMENTATIONS:-randaugment+colorjitter+erasing}"
DISTILL_P1_LABEL_SMOOTH="${DISTILL_P1_LABEL_SMOOTH:-0.05}"
DISTILL_MIXUP_ALPHA="${DISTILL_MIXUP_ALPHA:-0.2}"
DISTILL_CUTMIX_ALPHA="${DISTILL_CUTMIX_ALPHA:-1.0}"
DISTILL_EMA_DECAY="${DISTILL_EMA_DECAY:-0.999}"
DISTILL_GRAD_CLIP="${DISTILL_GRAD_CLIP:-1.0}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.7}"
DISTILL_TAU="${DISTILL_TAU:-2.0}"
DISTILL_P2_EPOCHS="${DISTILL_P2_EPOCHS:-80}"
DISTILL_P2_LR="${DISTILL_P2_LR:-0.003}"
DISTILL_RANK_WEIGHT="${DISTILL_RANK_WEIGHT:-0.1}"

mkdir -p "$DATA_DIR" "$SAVE_DIR"
cd "$ROOT"
export DATA_DIR SAVE_DIR GPU
export NUM_MEMBERS ENSEMBLE_EPOCHS ENSEMBLE_BATCH DISTILL_BATCH WORKERS FAKE_OOD_MIXUP_FRAC FAKE_OOD_PATCHSHUFFLE_FRAC FAKE_OOD_CUTPASTE_FRAC BACKBONE_LR_FACTOR
export DISTILL_P1_EPOCHS DISTILL_P1_LR DISTILL_P1_WD DISTILL_WARMUP DISTILL_P1_UNFREEZE_BLOCKS DISTILL_P1_BACKBONE_LR_FACTOR DISTILL_LLRD
export DISTILL_P1_AUGMENTATIONS DISTILL_P1_LABEL_SMOOTH DISTILL_MIXUP_ALPHA DISTILL_CUTMIX_ALPHA DISTILL_EMA_DECAY DISTILL_GRAD_CLIP
export DISTILL_ALPHA DISTILL_TAU DISTILL_P2_EPOCHS DISTILL_P2_LR DISTILL_RANK_WEIGHT
export PYTHONUNBUFFERED=1

echo "ROOT=$ROOT"
echo "DATA_DIR=$DATA_DIR"
echo "SAVE_DIR=$SAVE_DIR"
echo "GPU=$GPU"
echo "BACKBONE_LR_FACTOR=$BACKBONE_LR_FACTOR"
echo "FAKE_OOD_MIXUP_FRAC=$FAKE_OOD_MIXUP_FRAC"
echo "FAKE_OOD_PATCHSHUFFLE_FRAC=$FAKE_OOD_PATCHSHUFFLE_FRAC"
echo "FAKE_OOD_CUTPASTE_FRAC=$FAKE_OOD_CUTPASTE_FRAC"
echo "DISTILL_P1_EPOCHS=$DISTILL_P1_EPOCHS"
echo "DISTILL_P2_EPOCHS=$DISTILL_P2_EPOCHS"
echo "DISTILL_P1_AUGMENTATIONS=$DISTILL_P1_AUGMENTATIONS"

python - <<'PY'
import os
from torchvision import datasets

root = os.environ["DATA_DIR"]

downloads = [
    ("SVHN", lambda: datasets.SVHN(f"{root}/svhn", split="test", download=True)),
    ("CIFAR10", lambda: datasets.CIFAR10(root, train=False, download=True)),
    ("CIFAR100", lambda: datasets.CIFAR100(root, train=False, download=True)),
    ("STL10", lambda: datasets.STL10(root, split="test", download=True)),
    ("DTD", lambda: datasets.DTD(root, split="test", download=True)),
    ("FashionMNIST", lambda: datasets.FashionMNIST(root, train=False, download=True)),
    ("MNIST", lambda: datasets.MNIST(root, train=False, download=True)),
]

for name, fn in downloads:
    try:
        fn()
        print(f"[ok] {name}")
    except Exception as exc:
        print(f"[skip] {name}: {exc}")
PY

python train_ensemble.py \
  --save_dir "$SAVE_DIR" \
  --data_dir "$DATA_DIR" \
  --num_members "$NUM_MEMBERS" \
  --epochs "$ENSEMBLE_EPOCHS" \
  --batch_size "$ENSEMBLE_BATCH" \
  --workers "$WORKERS" \
  --backbone_lr_factor "$BACKBONE_LR_FACTOR" \
  --gpu "$GPU"

python cache_ensemble_targets.py \
  --save_dir "$SAVE_DIR" \
  --data_dir "$DATA_DIR" \
  --batch_size 128 \
  --gpu "$GPU" \
  --p2_data_mode fake_ood \
  --fake_ood_mixup_frac "$FAKE_OOD_MIXUP_FRAC" \
  --fake_ood_patchshuffle_frac "$FAKE_OOD_PATCHSHUFFLE_FRAC" \
  --fake_ood_cutpaste_frac "$FAKE_OOD_CUTPASTE_FRAC"

python distill.py \
  --save_dir "$SAVE_DIR" \
  --data_dir "$DATA_DIR" \
  --batch_size "$DISTILL_BATCH" \
  --workers "$WORKERS" \
  --gpu "$GPU" \
  --p1_epochs "$DISTILL_P1_EPOCHS" \
  --p1_lr "$DISTILL_P1_LR" \
  --p1_wd "$DISTILL_P1_WD" \
  --warmup "$DISTILL_WARMUP" \
  --p1_unfreeze_blocks "$DISTILL_P1_UNFREEZE_BLOCKS" \
  --p1_backbone_lr_factor "$DISTILL_P1_BACKBONE_LR_FACTOR" \
  --llrd "$DISTILL_LLRD" \
  --p1_augmentations "$DISTILL_P1_AUGMENTATIONS" \
  --p1_label_smoothing "$DISTILL_P1_LABEL_SMOOTH" \
  --mixup_alpha "$DISTILL_MIXUP_ALPHA" \
  --cutmix_alpha "$DISTILL_CUTMIX_ALPHA" \
  --ema_decay "$DISTILL_EMA_DECAY" \
  --grad_clip "$DISTILL_GRAD_CLIP" \
  --alpha "$DISTILL_ALPHA" \
  --tau "$DISTILL_TAU" \
  --p2_epochs "$DISTILL_P2_EPOCHS" \
  --p2_lr "$DISTILL_P2_LR" \
  --rank_weight "$DISTILL_RANK_WEIGHT"

python evaluate_student.py \
  --save_dir "$SAVE_DIR" \
  --data_dir "$DATA_DIR" \
  --batch_size 128 \
  --gpu "$GPU"
