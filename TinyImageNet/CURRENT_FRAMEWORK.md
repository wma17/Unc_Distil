# TinyImageNet Current Framework

This document summarizes the current TinyImageNet pipeline in this directory and records the recent code changes made for the next teacher retraining and full student redistillation run.

## Goal

The current TinyImageNet version targets:

- a DeiT-Small LoRA ensemble teacher with stronger member diversity
- distillation of both class prediction and epistemic uncertainty (EU)
- fake-OOD-based EU supervision rather than relying on real OOD in the training pipeline

The main training path is now:

1. train a diverse teacher ensemble
2. cache teacher predictions and EU targets
3. run full student distillation from Phase 1 and Phase 2
4. evaluate classification, calibration, and OOD uncertainty behavior

## Core Model Design

### Teacher ensemble

Teacher members are defined in `models.py` and `lora.py`.

- Base model: pretrained `deit_small_patch16_224`
- Output head: new 200-class linear head for Tiny-ImageNet
- Adaptation: LoRA modules injected into attention and optionally MLP linears
- Fine-tuning: most pretrained weights remain frozen, with optional unfreezing of the last few transformer blocks

Each saved member checkpoint stores only the trainable state needed to reconstruct the member:

- LoRA adapter weights
- classification head weights
- optionally unfrozen trainable backbone weights

### Student

The student is also DeiT-Small, but with an extra scalar EU head.

- Backbone feature: CLS token feature, dimension 384
- Class head output: 200 logits
- EU head input: `[CLS feature || softmax(logits)]`, dimension `384 + 200 = 584`
- EU head output: one scalar per sample

The student is optimized in two stages:

- Phase 1: classification distillation
- Phase 2: EU regression

## Current Pipeline

### 1. Data and transforms

The data pipeline lives in `data.py`.

- ID dataset: Tiny-ImageNet train/val
- Validation transform: resize to 256, center crop 224, normalize
- Training transform: resize to 256, random crop 224, horizontal flip, plus optional augmentations

Supported image-level augmentation tokens:

- `randaugment`
- `autoaugment`
- `colorjitter`
- `perspective`
- `erasing`

Supported batch-level augmentation tokens:

- `mixup`
- `cutmix`

### 2. Teacher training

Teacher training is implemented in `train_ensemble.py`.

Each ensemble member is sampled independently with its own configuration. The current code now uses a wider heterogeneity range than the older homogeneous setup.

Per-member variation now includes:

- LoRA rank: `4, 8, 16, 32`
- LoRA alpha: sampled relative to rank
- LoRA dropout: `0.0, 0.05, 0.1`
- LoRA targets:
  - `qkv_only`
  - `qkv+proj`
  - `qkv+proj+mlp`
- unfrozen backbone blocks: `0, 1, 2, 4`
- augmentation bundle: sampled per member
- label smoothing
- bagging fraction
- learning rate
- weight decay
- warmup epochs
- backbone LR factor

Teacher intent:

- keep pretrained DeiT stability
- make member behaviors less correlated
- improve teacher EU diversity rather than only top-1 accuracy

Artifacts written into a save directory:

- `member_*.pt`
- `ensemble_configs.json`

### 3. Teacher target caching

Teacher target generation is implemented in `cache_ensemble_targets.py`.

This stage loads all teacher members and caches:

- mean class probabilities on train and val
- total uncertainty (TU)
- aleatoric uncertainty (AU)
- epistemic uncertainty (EU)
- corrupted-ID targets
- fake-OOD targets
- evaluation OOD targets

The main file produced here is:

- `teacher_targets.npz`

This file is the bridge between teacher training and student distillation.

### 4. Student distillation

Student distillation is implemented in `distill.py`.

#### Phase 1: classification KD

Phase 1 trains the DeiT student backbone and class head using:

- CE with hard labels
- KL to teacher probabilities
- MixUp / CutMix
- label smoothing
- EMA
- gradient clipping
- partial unfreezing with layer-wise LR decay

Phase 1 best checkpoint:

- `student_phase1.pt`

#### Phase 2: EU regression

Phase 2 freezes the student backbone and class head, reinitializes the EU head, and trains only the EU head using:

- MSE to teacher EU
- pairwise ranking loss

The Phase 2 data mix is currently:

- `50%` clean ID
- `25%` corrupted ID
- `25%` fake OOD

That clean-ID fraction stays at 50% by construction and was kept specifically to avoid degrading the model on normal in-distribution data.

Phase 2 best checkpoint:

- `student.pt`

### 5. Evaluation

Evaluation is implemented in `evaluate_student.py`.

The evaluation writes a markdown report to:

- `../result.md` relative to the checkpoint directory

It reports:

- teacher vs student val accuracy
- calibration metrics
- correctness agreement
- AUROC on OOD detection
- EU and TU behavior
- throughput

## Recent Code Changes

The following changes were added to support the next full retraining cycle.

### 1. Teacher diversity was increased

File:

- `train_ensemble.py`

Changes:

- widened per-member search space for LoRA rank, dropout, targets, unfreezing, weight decay, warmup, and backbone LR factor
- replaced the more uniform augmentation recipe with a more heterogeneous per-member augmentation sampler
- kept the low-LR backbone update pattern, but made the backbone LR factor member-specific

Why:

- the previous 16-member ensemble looked too correlated
- more members alone do not help if they collapse toward similar solutions
- this change aims to increase functional disagreement and improve EU quality

### 2. Fake OOD generation was expanded

File:

- `data.py`

New fake-OOD families:

- `mixup`
- `masked`
- `patchshuffle`
- `cutpaste`

New functionality:

- deterministic metadata generation for all fake-OOD families
- dataset rebuild support from cached specs
- on-the-fly fake sample reconstruction from `teacher_targets.npz`

Why:

- the old fake-OOD setup used only `mixup` and `masked`
- that was likely too narrow and too easy
- richer fake OOD should better cover synthetic near-OOD behavior without using real OOD in training

### 3. Fake-OOD caching now supports explicit family fractions

File:

- `cache_ensemble_targets.py`

New arguments:

- `--fake_ood_mixup_frac`
- `--fake_ood_patchshuffle_frac`
- `--fake_ood_cutpaste_frac`

Behavior:

- masked fake OOD gets the remaining fraction
- the sum of explicit fractions must be at most `1.0`
- all selected fake families get their own cached EU arrays

Why:

- this makes fake-OOD composition controlled and reproducible
- it also lets the student Phase 2 replay the same family mix used during teacher target construction

### 4. Phase 2 now consumes all cached fake-OOD families

File:

- `distill.py`

Changes:

- fake-OOD Phase 2 loader now reconstructs all available fake families from `teacher_targets.npz`
- family fractions are read from cached metadata
- the requested OOD budget is split across active fake families
- fallback to real OOD remains available if no fake-OOD family is present

Important invariant:

- clean ID still remains `50%` of Phase 2 data

### 5. Full retrain wrapper was added

Files:

- `run_fake_ood_pipeline.sh`
- `run_rich_fake_ood_pipeline.sh`

`run_fake_ood_pipeline.sh` now:

- trains the teacher
- caches fake-OOD targets
- reruns full student distillation from Phase 1
- runs final evaluation

`run_rich_fake_ood_pipeline.sh` is a convenience wrapper with a recommended recipe:

- `12` members by default
- richer fake-OOD mix
- stronger student Phase 1 defaults
- same end-to-end pipeline entrypoint

## Current Fake-OOD Design

The current fake-OOD families are:

### MixUp

- interpolates two ID images in raw pixel space
- controlled by predetermined lambdas

### Masked

- masks content with several styles:
  - random block
  - random pixel
  - center crop mask
  - multi-block mask
  - border mask

### Patch shuffle

- divides the image into a grid
- shuffles patch locations
- preserves local texture while breaking global structure

### CutPaste

- pastes a patch from image B into image A
- creates local semantic inconsistencies

These families are intended to cover a broader range of abnormal structure while still remaining synthetic and reproducible.

## Recommended End-to-End Run

The current recommended launch path is:

```bash
cd /home/maw6/maw6/unc_regression/TinyImageNet
nohup env GPU=0 \
  SAVE_DIR=/home/maw6/maw6/unc_regression/TinyImageNet/checkpoints_rich_fake_ood_12m \
  NUM_MEMBERS=12 \
  ENSEMBLE_EPOCHS=50 \
  FAKE_OOD_MIXUP_FRAC=0.25 \
  FAKE_OOD_PATCHSHUFFLE_FRAC=0.25 \
  FAKE_OOD_CUTPASTE_FRAC=0.25 \
  DISTILL_P1_EPOCHS=50 \
  DISTILL_P1_UNFREEZE_BLOCKS=12 \
  DISTILL_P1_AUGMENTATIONS=basic \
  DISTILL_P1_LABEL_SMOOTH=0.1 \
  DISTILL_P1_WD=0.1 \
  DISTILL_P2_EPOCHS=80 \
  DISTILL_P2_LR=0.003 \
  bash /home/maw6/maw6/unc_regression/TinyImageNet/run_rich_fake_ood_pipeline.sh \
  > /home/maw6/maw6/unc_regression/TinyImageNet/retrain_rich_fake_ood.log 2>&1 &
```

This command runs:

1. teacher retraining
2. teacher target rebuild
3. full student distillation
4. final evaluation

## Current Framework Summary

The current TinyImageNet version should be understood as:

- a DeiT-Small LoRA teacher ensemble with explicit diversity knobs
- a two-phase DeiT-Small student for class prediction plus EU prediction
- a fake-OOD-based EU distillation pipeline
- a Phase 2 design that preserves at least half of the data budget for clean ID
- an end-to-end shell entrypoint for full reruns

The immediate purpose of the current codebase is not just to improve teacher accuracy. It is to improve the quality and transferability of teacher EU, then distill that EU into a single student under a reviewer-safer fake-OOD setup.

## Known Practical Notes

- `result.md` is overwritten by evaluation runs because `evaluate_student.py` writes to the parent of the checkpoint directory.
- Real OOD loaders are still used for evaluation and remain available as a fallback if fake-OOD Phase 2 sources are absent.
- The main open research question is still whether the richer fake-OOD mix and stronger teacher heterogeneity are enough to recover the EU quality lost in the previous 16-member run.
