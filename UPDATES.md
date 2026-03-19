# Uncertainty Distillation — Implementation Updates

> Target venue: NeurIPS 2026
> Last updated: 2026-03-18

---

## What Was Implemented

### Part 1 — Retrofit Existing Experiments (E1 MNIST, E2 CIFAR-10, E3 TinyImageNet)

#### 1a. New Evaluation Metrics (all three datasets)

Added to each `evaluate_student.py`:

| Metric | Description |
|---|---|
| **ECE-15** | 15-bin Expected Calibration Error |
| **NLL** | Mean negative log-likelihood |
| **Brier Score** | Mean sum-of-squared errors vs one-hot |
| **AURC** | Area Under Risk-Coverage Curve (selective prediction) |
| **Oracle AURC / Gap** | Lower bound + student-vs-oracle gap |
| **Acc @ 90% / 80% coverage** | Accuracy when abstaining on most uncertain 10%/20% |
| **Inference Throughput** | Samples/sec: ensemble vs single member vs student |

#### 1b. Ablation CLI Flags Added

**`CIFAR-10/distill.py`** — new arguments:

| Flag | Values | Purpose |
|---|---|---|
| `--curriculum` | A1, A2, A3 | Phase 2 data tiers (A1=clean only, A2=+corrupted, A3=+OOD) |
| `--loss_mode` | mse, log_mse, ranking, combined | Phase 2 loss variant |
| `--joint` | flag | Joint training (backbone+head simultaneously) |
| `--joint_gamma` | float (default 1.0) | EU loss weight in joint mode |

**`CIFAR-10/cache_ensemble_targets.py`** — new argument:

| Flag | Purpose |
|---|---|
| `--subset_size K` | Use only first K ensemble members (Ablation C) |

#### 1c. Bash Pipeline Scripts

Each dataset has a `run_experiments.sh` that runs all evaluations and ablations in one command.

---

### Part 2 — BNN Baselines (all three datasets)

Four BNN baselines implemented. All use `member_0.pt` as the pretrained backbone.

| Method | Training | EU source |
|---|---|---|
| **MC Dropout** | Fine-tune 20 epochs (CIFAR-10/MNIST) / 10 epochs (TinyImageNet) with spatial dropout enabled | MI from T=16 stochastic passes |
| **EDL** | Freeze backbone; train Dirichlet head 50 epochs | `U_epi = H[p̂] − U_ale` (digamma formula) |
| **LLLA** | No training; KFAC Laplace on last FC layer via laplace-torch | MI from T=16 posterior samples |
| **SGLD** | Last FC layer only, 200-step burn-in, thin=10, collect 16 samples | MI from 16 weight samples |

> **TinyImageNet special case:** EDL and LLLA applied to prediction head only (CLS features from frozen DeiT backbone). SGLD uses step_size=1e-6 (vs 1e-5 for smaller models). MC Dropout injects `attn_drop`/`proj_drop` into DeiT attention blocks.

New files per dataset:

```
CIFAR-10/train_baselines.py
CIFAR-10/evaluate_baselines.py
MNIST/train_baselines.py
MNIST/evaluate_baselines.py
TinyImageNet/train_baselines.py
TinyImageNet/evaluate_baselines.py
```

---

## File Map (all new / modified files)

```
unc_regression/
├── CIFAR-10/
│   ├── evaluate_student.py      [MODIFIED] +ECE/NLL/Brier/AURC/Throughput
│   ├── distill.py               [MODIFIED] +curriculum/loss_mode/joint flags
│   ├── cache_ensemble_targets.py[MODIFIED] +--subset_size K
│   ├── run_experiments.sh       [NEW] full ablation pipeline
│   ├── train_baselines.py       [NEW] MC Dropout / EDL / LLLA / SGLD
│   └── evaluate_baselines.py    [NEW] aligned metric comparison
│
├── MNIST/
│   ├── evaluate_student.py      [MODIFIED] +ECE/NLL/Brier/AURC/Throughput
│   ├── run_experiments.sh       [NEW] evaluation pipeline
│   ├── train_baselines.py       [NEW] MC Dropout / EDL / LLLA / SGLD
│   └── evaluate_baselines.py    [NEW] aligned metric comparison
│
└── TinyImageNet/
    ├── evaluate_student.py      [MODIFIED] +ECE/NLL/Brier/AURC/Throughput
    ├── run_experiments.sh       [NEW] evaluation pipeline
    ├── train_baselines.py       [NEW] MC Dropout / EDL / LLLA / SGLD
    └── evaluate_baselines.py    [NEW] aligned metric comparison
```

---

## Commands to Run Everything

> **Prerequisites:** Ensemble members must be trained first (`train_ensemble.py`).
> For Ablation C (ensemble size), CIFAR-10 needs 16 members trained.
> Teacher targets must be cached (`cache_ensemble_targets.py`).
> Student must be distilled (`distill.py`) before evaluate_student runs.

### MNIST (GPU 1)

```bash
cd /home/maw6/maw6/unc_regression/MNIST

# 0. Cache teacher targets (if not done)
python cache_ensemble_targets.py --save_dir ./checkpoints --gpu 1

# 1. Distill student (if not done)
python distill.py --save_dir ./checkpoints --gpu 1

# 2. Evaluate student (all metrics)
python evaluate_student.py --save_dir ./checkpoints --gpu 1

# 3. Train BNN baselines
python train_baselines.py --save_dir ./checkpoints --gpu 1

# 4. Evaluate baselines
python evaluate_baselines.py --save_dir ./checkpoints --gpu 1

# --- OR run everything via the pipeline script ---
bash run_experiments.sh                       # evaluates main + fake_ood checkpoints
GPU=1 bash run_experiments.sh                 # specify GPU
```

### CIFAR-10 (GPU 0)

```bash
cd /home/maw6/maw6/unc_regression/CIFAR-10

# 0. Cache teacher targets
python cache_ensemble_targets.py --save_dir ./checkpoints --gpu 0

# 1. Distill student
python distill.py --save_dir ./checkpoints --gpu 0

# 2. Evaluate student
python evaluate_student.py --save_dir ./checkpoints --gpu 0

# 3. Run all ablations (A=curriculum, B=loss, C=ensemble size, D=OOD source, E=joint)
bash run_experiments.sh                       # all sections
SECTIONS="0 A B" bash run_experiments.sh      # selected sections only
GPU=0 bash run_experiments.sh

# 4. Train BNN baselines
python train_baselines.py --save_dir ./checkpoints --gpu 0

# 5. Evaluate baselines
python evaluate_baselines.py --save_dir ./checkpoints --gpu 0

# --- Ablation C requires 16 ensemble members ---
python train_ensemble.py --num_members 16 --epochs 200 --gpu 0
# Then re-run cache + distill per ablation dir (handled by run_experiments.sh Section C)
```

### TinyImageNet (GPU 0)

```bash
cd /home/maw6/maw6/unc_regression/TinyImageNet

# 0. Cache teacher targets (if not done)
python cache_ensemble_targets.py --save_dir ./checkpoints --gpu 0

# 1. Distill student (if not done)
python distill.py --save_dir ./checkpoints --gpu 0

# 2. Evaluate student (all known checkpoint variants, best first)
bash run_experiments.sh
GPU=0 SAVE_DIR=./checkpoints_fake_ood_ft_wd005_blr1e3 bash run_experiments.sh  # single dir

# 3. Evaluate student directly
python evaluate_student.py --save_dir ./checkpoints_fake_ood_ft_wd005_blr1e3 --gpu 0

# 4. Train BNN baselines
python train_baselines.py --save_dir ./checkpoints_fake_ood_ft_wd005_blr1e3 --gpu 0 --batch_size 64

# 5. Evaluate baselines
python evaluate_baselines.py --save_dir ./checkpoints_fake_ood_ft_wd005_blr1e3 --gpu 0 --batch_size 64
```

### BNN Baselines — Method-Specific Options

```bash
# Train only specific methods
python train_baselines.py --save_dir ./checkpoints --gpu 0 --methods mc_dropout edl

# Skip LLLA if laplace-torch not installed
python train_baselines.py --save_dir ./checkpoints --gpu 0 --methods mc_dropout edl sgld

# Adjust MC Dropout fine-tune epochs
python train_baselines.py --save_dir ./checkpoints --gpu 0 --mc_epochs 30 --mc_dropout_p 0.15

# Adjust SGLD
python train_baselines.py --save_dir ./checkpoints --gpu 0 --sgld_burn_in 500 --sgld_samples 32

# Evaluate with T=32 samples instead of default 16
python evaluate_baselines.py --save_dir ./checkpoints --gpu 0 --T 32
```

### Install laplace-torch (for LLLA)

```bash
pip install laplace-torch
```

---

## CIFAR-10 Ablation Reference

| Section | What it tests | CLI | Output dir |
|---|---|---|---|
| 0 | Main model evaluation | — | `checkpoints/` |
| A | Phase 2 curriculum | `--curriculum A1/A2/A3` | `checkpoints_abl_A_A1/` etc. |
| B | Phase 2 loss mode | `--loss_mode mse/log_mse/ranking/combined` | `checkpoints_abl_B_*/` |
| C | Ensemble size K | `--subset_size K` in cache | `checkpoints_abl_C_K3/`, `K5/` |
| D | OOD source | `--p2_data_mode fake_ood/ood` | `checkpoints_abl_D_*/` |
| E | Joint vs two-phase | `--joint` | `checkpoints_abl_E_joint/` |

Run selected ablations:
```bash
SECTIONS="0 A B C" bash run_experiments.sh   # skip D (needs real OOD data) and E (slow)
```

---

## What Remains (Part 2 & 3 of Original Plan)

### Part 2 — Pascal VOC 2012 Semantic Segmentation (E4)
Not yet started. Directory: `VOC/`

Files to create:
- `models.py` — SegFormer-B2 teacher (LoRA r=16) + student with spatial EU head
- `data.py` — VOC 2012 + SBD dataloader
- `lora.py` — LoRA adapters (adapt from TinyImageNet)
- `train_ensemble.py` — 5-member ensemble
- `cache_ensemble_targets.py` — per-pixel EU as float16 .npz shards
- `distill.py` — two-phase distillation
- `evaluate_student.py` — segmentation-specific metrics

### Part 3 — SST-2 Sentiment (E5)
Not yet started. Directory: `SST2/`

Files to create:
- `models.py` — BERT-base teacher (LoRA r=8) + DistilBERT student with EU head
- `data.py` — SST-2 + OOD loaders (IMDB, Yelp, Amazon, AG News, 20NG)
- `train_ensemble.py` — 5-member LoRA BERT ensemble
- `cache_ensemble_targets.py` — per-sentence EU targets
- `distill.py` — two-phase distillation
- `evaluate_student.py` — full evaluation + 5 OOD datasets

Dependencies to install for E5:
```bash
pip install transformers datasets nlpaug peft
```
