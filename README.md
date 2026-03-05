# Uncertainty Distillation

A two-phase knowledge distillation framework that compresses a **deep ensemble teacher** into a **single student model** while preserving both classification accuracy and **epistemic uncertainty (EU)** estimation.

The student produces two outputs simultaneously: classification logits and a scalar EU score — making ensemble-quality uncertainty estimates available at single-model inference cost.

---

## Overview

Deep ensembles are the gold standard for uncertainty quantification, but are expensive (5× parameters, 5× compute). This project distills the ensemble's uncertainty signal into a lightweight student with a dedicated EU regression head, trained in two phases:

```
Phase 1 — Classification KD:
    L₁ = (1−α)·CE(y, softmax(z_S)) + α·τ²·KL(softmax(z_T/τ) ∥ softmax(z_S/τ))

Phase 2 — EU head regression (backbone frozen, mixed data):
    L₂ = log1p_MSE(EU_S, EU_T) + β·PairwiseRankingLoss(EU_S, EU_T)
    Training data: 50% ID + 25% shifted ID + 25% OOD (or fake OOD)
```

Experiments are provided for **CIFAR-10**, **MNIST**, and **TinyImageNet**.

---

## Repository Structure

```
unc_regression/
├── CIFAR-10/
│   ├── models.py                  # Teacher (CIFARResNet18) + Student (dual-head)
│   ├── train_ensemble.py          # Train diverse deep ensemble teacher
│   ├── cache_ensemble_targets.py  # Cache teacher soft labels + EU targets (OOD or fake OOD)
│   ├── distill.py                 # Two-phase distillation (Phase 1 + Phase 2)
│   ├── evaluate_student.py        # Accuracy, EU correlation, OOD AUROC
│   ├── evaluate_uncertainty.py    # Teacher ensemble uncertainty analysis
│   ├── plot_eu.py                 # Visualize EU distributions
│   ├── uncertainty_cam.py        # Uncertainty-aware CAM visualization
│   ├── uncertainty_cam_math.md   # EU-CAM derivation notes
│   ├── figures/                   # Generated plots
│   └── uncertainty_cam/           # CAM output directory
├── MNIST/
│   ├── models.py
│   ├── train_ensemble.py
│   ├── cache_ensemble_targets.py
│   ├── distill.py
│   ├── evaluate_student.py
│   └── plot_eu.py
├── TinyImageNet/
│   ├── models.py
│   ├── data.py                    # TinyImageNet dataset loader
│   ├── lora.py                    # LoRA adapter support
│   ├── train_ensemble.py
│   ├── cache_ensemble_targets.py
│   ├── distill.py
│   ├── evaluate_student.py
│   └── plot_eu.py
├── README.md
└── requirements.txt
```

---

## Method

### Teacher: Deep Ensemble

Each ensemble member is a CIFAR-adapted ResNet18 (3×3 stem, stride 1) trained with randomly sampled diversity strategies:

| Strategy | Range |
|---|---|
| Random seed | Independent init + data order |
| Augmentation mix | Subset of {AutoAugment, ColorJitter, Grayscale, Rotation, Cutout, RandErasing} |
| Label smoothing | Uniform [0, 0.05] |
| Dropout (before FC) | {0, 0.05, 0.1} |
| Data fraction (bagging) | Uniform [0.8, 1.0] |
| Learning rate | Uniform [0.05, 0.15] |
| LR schedule | {Cosine, Step} |
| Head init scale | Log-uniform [0.5, 1.5] |

Epistemic uncertainty is computed as the mean entropy minus the entropy of the mean (mutual information):

```
EU = H[E_θ[p(y|x,θ)]] − E_θ[H[p(y|x,θ)]]
```

### Student: Dual-Head ResNet18

The student shares the same ResNet18 backbone but adds a dedicated EU head:

```
feat (512-d) ──┬──► fc ──► logits (10-d)
               │
               ├── probs = softmax(logits).detach()
               │
               └── [feat ⊕ probs] ──► EU_fc1 ──► ReLU ──► EU_fc2 ──► Softplus ──► EU (scalar)
```

The EU head receives both backbone features and the model's own confidence (detached softmax), which is strongly correlated with epistemic uncertainty. The `Softplus` output enforces EU ≥ 0.

### Phase 2 Mixed Training Data

To teach the student to produce high EU across different input regimes, Phase 2 uses a mixed dataset with fixed ratios:

- **50% ID (clean)** — low EU baseline
- **25% shifted ID (corrupted)** — medium EU (Gaussian noise, blur, low contrast)
- **25% OOD or fake OOD** — high EU

**Real OOD** (when available): SVHN + CIFAR-100

**Fake OOD** (when OOD datasets are unavailable): mixup + masked samples, both derived from CIFAR-10 train only:
- *Mixup* — λ ∈ {0.2, 0.4, 0.6, 0.8}
- *Masked* — styles: random block, random pixel, center crop; rates: 0.1, 0.3, 0.5

Use `--p2_data_mode fake_ood` in `cache_ensemble_targets.py` to enable fake OOD.

### Loss Functions

**log1p-MSE**: MSE in log(1+x) space compresses the long-tailed EU distribution (95% near-zero, 5% large) into a balanced one, preventing the model from ignoring the low-EU regime.

**Pairwise Ranking Loss**: Directly optimizes the ordering of EU scores across random pairs, complementing the magnitude regression.

---

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

```
torch>=2.0
torchvision>=0.15
numpy>=1.24
```

### Full Pipeline (CIFAR-10)

**Step 1 — Train the teacher ensemble**
```bash
cd CIFAR-10
python train_ensemble.py --num_members 5 --epochs 200 --gpu 0
```

**Step 2 — Cache teacher targets** (soft labels + EU scores for all splits + OOD datasets)
```bash
# With real OOD (SVHN, CIFAR-100)
python cache_ensemble_targets.py --save_dir ./checkpoints --gpu 0

# With fake OOD only (mixup + masked, no external datasets)
python cache_ensemble_targets.py --save_dir ./checkpoints --gpu 0 --p2_data_mode fake_ood
```

**Step 3 — Distill into student**
```bash
# Full two-phase distillation
python distill.py --save_dir ./checkpoints --gpu 0

# Phase 2 only (if Phase 1 checkpoint already exists)
python distill.py --save_dir ./checkpoints --gpu 0 --phase2_only
```

**Step 4 — Evaluate**
```bash
python evaluate_student.py --save_dir ./checkpoints --gpu 0
```

**Step 5 — Plot EU distributions**
```bash
python plot_eu.py --save_dir ./checkpoints
```

**Optional — Uncertainty-aware CAM**
```bash
python uncertainty_cam.py --save_dir ./checkpoints --out_dir ./uncertainty_cam
```

### Key Arguments

| Script | Argument | Default | Description |
|---|---|---|---|
| `train_ensemble.py` | `--num_members` | 5 | Ensemble size |
| `train_ensemble.py` | `--no_diversity` | False | Ablation: seed-only diversity |
| `cache_ensemble_targets.py` | `--p2_data_mode` | ood | `ood` or `fake_ood` |
| `cache_ensemble_targets.py` | `--fake_ood_mixup_frac` | 0.5 | Fraction of fake OOD from mixup (rest masked) |
| `distill.py` | `--alpha` | 0.7 | KD weight for Phase 1 |
| `distill.py` | `--tau` | 4.0 | Temperature for Phase 1 |
| `distill.py` | `--rank_weight` | 1.0 | β weight for ranking loss in Phase 2 |
| `distill.py` | `--p1_epochs` | 200 | Phase 1 training epochs |
| `distill.py` | `--p2_epochs` | 100 | Phase 2 training epochs |

---

## Evaluation

`evaluate_student.py` reports:

1. **Accuracy** — Student vs teacher ensemble on clean CIFAR-10 test
2. **Correctness agreement** — Fraction of samples where student and teacher agree
3. **EU correlation** — Pearson + Spearman on clean, corrupted, and OOD splits
4. **OOD detection AUROC** — Student EU vs teacher EU vs softmax baselines
   - Seen OOD (in Phase 2 training): SVHN, CIFAR-100
   - Unseen OOD (held out): MNIST, FashionMNIST, STL10, DTD
5. **Uncertainty decomposition** — TU / EU / AU comparison (teacher vs student)
6. **Decomposed AUROC** — EU vs AU vs TU for OOD detection

---

## Sample Results

Example output figures saved in `CIFAR-10/figures/`:

| Figure | Description |
|---|---|
| `uncertainty_decomposition.png` | TU / EU / AU distributions for teacher and student |
| `eu_violin_comparison.png` | EU distributions across clean / corrupted / OOD |
| `eu_correlation.png` | Scatter plot of student EU vs teacher EU |
| `eu_dist_teacher.png` | Teacher EU histogram |
| `eu_dist_student.png` | Student EU histogram |
| `teacher_tu_eu_au_dist.png` | Teacher TU / EU / AU distributions |
| `student_tu_eu_au_dist.png` | Student TU / EU / AU distributions |

---

## Data

Datasets are downloaded automatically by torchvision. By default they are stored in `../data/` (one level above the experiment folder). All standard datasets (CIFAR-10, CIFAR-100, MNIST, SVHN, STL10, DTD, FashionMNIST) are fetched via `torchvision.datasets`.

---

## References

- Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, NeurIPS 2017
- Fort et al., *Deep Ensembles: A Loss Landscape Perspective*, 2019
- Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015
- Malinin & Gales, *Predictive Uncertainty Estimation via Prior Networks*, NeurIPS 2018
