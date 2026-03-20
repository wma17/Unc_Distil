# Uncertainty Distillation

This repository is for ensemble-teacher uncertainty distillation: train a diverse teacher ensemble, cache its predictive uncertainty, and distill a single student that predicts both class logits and epistemic uncertainty (EU).

The current project direction is:

- use a teacher ensemble as the uncertainty source
- distill both prediction and uncertainty into one student
- train the uncertainty head with fake OOD synthesized from ID data rather than true OOD
- evaluate accuracy, calibration, selective prediction, OOD behavior, and teacher-student EU fidelity

## Canonical Project Docs

These are the docs that should be treated as current:

- [`UPDATES.md`](/home/maw6/maw6/unc_regression/UPDATES.md): implemented features, commands, and experiment status
- [`additional_experiment_plan.md`](/home/maw6/maw6/unc_regression/additional_experiment_plan.md): broader experiment roadmap
- [`TinyImageNet/CURRENT_FRAMEWORK.md`](/home/maw6/maw6/unc_regression/TinyImageNet/CURRENT_FRAMEWORK.md): active TinyImageNet design and recommended retrain recipe

## Current Status

- `MNIST/`: completed image-classification pipeline with fake-OOD and baseline evaluation support
- `CIFAR-10/`: completed image-classification pipeline with ablation and BNN baseline support
- `TinyImageNet/`: active working area for the current richer fake-OOD teacher retraining and student redistillation cycle

For TinyImageNet, the recommended workflow is:

1. train a heterogeneous DeiT-S LoRA teacher ensemble
2. cache teacher soft labels and uncertainty targets
3. distill a DeiT-S student in two phases
4. evaluate prediction quality, calibration, throughput, and OOD uncertainty behavior

## Method Summary

Phase 1 distills prediction quality:

```text
L_p1 = (1 - alpha) * CE(y, z_s) + alpha * tau^2 * KL(softmax(z_T / tau) || softmax(z_s / tau))
```

Phase 2 distills epistemic uncertainty with the backbone/classifier frozen:

```text
L_p2 = log-MSE(EU_s, EU_T) + beta * ranking_loss(EU_s, EU_T)
```

The Phase 2 data mix is intentionally reviewer-safer than using true OOD for training:

- `50%` clean ID
- `25%` shifted/corrupted ID
- `25%` fake OOD built from ID samples

Current fake-OOD families in TinyImageNet:

- `mixup`
- `masked`
- `patchshuffle`
- `cutpaste`

## Repository Layout

```text
unc_regression/
├── README.md
├── UPDATES.md
├── additional_experiment_plan.md
├── cleanup_artifacts.sh          # remove stale logs/checkpoints safely
├── data/                         # shared downloaded datasets
├── MNIST/
│   ├── train_ensemble.py
│   ├── cache_ensemble_targets.py
│   ├── distill.py
│   ├── evaluate_student.py
│   ├── train_baselines.py
│   ├── evaluate_baselines.py
│   └── run_experiments.sh
├── CIFAR-10/
│   ├── train_ensemble.py
│   ├── cache_ensemble_targets.py
│   ├── distill.py
│   ├── evaluate_student.py
│   ├── train_baselines.py
│   ├── evaluate_baselines.py
│   └── run_experiments.sh
└── TinyImageNet/
    ├── data.py
    ├── lora.py
    ├── models.py
    ├── train_ensemble.py
    ├── cache_ensemble_targets.py
    ├── distill.py
    ├── evaluate_student.py
    ├── train_baselines.py
    ├── evaluate_baselines.py
    ├── run_fake_ood_pipeline.sh
    ├── run_rich_fake_ood_pipeline.sh
    ├── run_experiments.sh
    └── CURRENT_FRAMEWORK.md
```

## Active Artifact Policy

The tree gets cluttered quickly because training leaves behind many large ignored artifacts. The working convention is:

- keep source code, docs, and the current best or active checkpoint directories
- keep old ablation outputs only when they are still needed immediately
- delete one-off logs, incomplete trial checkpoints, and recreateable ablation directories

At the moment the important TinyImageNet checkpoint directories are:

- `TinyImageNet/checkpoints_rich_fake_ood_12m/`: current retraining workspace
- `TinyImageNet/checkpoints_fake_ood_ft_wd005_blr1e3/`: best complete fake-OOD TinyImageNet run kept for evaluation/baselines

The cleanup helper keeps those and removes older incomplete trial outputs by default.

## Recommended Commands

Install dependencies:

```bash
cd /home/maw6/maw6/unc_regression
pip install -r requirements.txt
```

Run the current TinyImageNet end-to-end retrain:

```bash
cd /home/maw6/maw6/unc_regression/TinyImageNet
bash run_rich_fake_ood_pipeline.sh
```

Evaluate active TinyImageNet checkpoints:

```bash
cd /home/maw6/maw6/unc_regression/TinyImageNet
bash run_experiments.sh
```

Evaluate legacy TinyImageNet checkpoints as well:

```bash
cd /home/maw6/maw6/unc_regression/TinyImageNet
INCLUDE_LEGACY=1 bash run_experiments.sh
```

Clean stale artifacts:

```bash
cd /home/maw6/maw6/unc_regression
DRY_RUN=1 bash cleanup_artifacts.sh
DRY_RUN=0 bash cleanup_artifacts.sh
```

## Notes

- `TinyImageNet/run_experiments.sh` now prioritizes the current rich fake-OOD retrain directory and the best complete fake-OOD checkpoint.
- Most large checkpoint payloads are ignored by git; small metadata files from pruned checkpoint folders still show up in git diffs.
- `evaluate_student.py` writes `result.md` in the parent of the checkpoint directory, so the most recent evaluation overwrites the previous summary for that dataset.
