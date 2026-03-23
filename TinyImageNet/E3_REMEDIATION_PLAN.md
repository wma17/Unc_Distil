# E3 TinyImageNet Remediation Plan

**Date:** 2026-03-22  
**Scope:** Recover student accuracy and improve true-OOD separation for TinyImageNet uncertainty distillation.

## 1. Goal

The immediate goal is to produce a single-pass student that:

- keeps TinyImageNet accuracy at or above the current strong baseline
- preserves good clean-ID epistemic-uncertainty correlation
- separates clean ID from true OOD, especially `SVHN` and `STL10`, more reliably

This plan is based on direct inspection of:

- `TinyImageNet/distill.py`
- `TinyImageNet/models.py`
- `TinyImageNet/cache_ensemble_targets.py`
- `TinyImageNet/data.py`
- `TinyImageNet/E3_ANALYSIS.md`
- `TinyImageNet/result.md`
- saved artifacts in `TinyImageNet/checkpoints_v3/`
- saved artifacts in `TinyImageNet/checkpoints_rich_fake_ood_12m/`

## 2. Executive Summary

There are two different problems, and they should not be optimized together at first.

### Problem A: student accuracy regressed in v3

This is a Phase 1 issue.

- `checkpoints_v3/student_phase1.pt` peaks at `85.67%`
- `checkpoints_rich_fake_ood_12m/student_phase1.pt` peaks at `87.51%`

The v3 student started Phase 2 from a weaker classifier, so its later uncertainty results are confounded by a degraded backbone.

### Problem B: true-OOD separation is still weak for SVHN and somewhat weak for STL10

This is mainly a Phase 2 issue.

The current EU head:

- only sees frozen `[CLS feature || softmax probs]`
- is trained on pre-extracted features
- groups all fake OOD into one coarse tier
- is strongly penalized for clean-ID over-prediction with `suppress_weight=2.0`

That setup improved clean-ID EU calibration, but it also suppresses the head too aggressively on truly OOD inputs.

## 3. Confirmed Findings

## 3.1 The current v3 diagnosis in `E3_ANALYSIS.md` is only partially correct

The file correctly identifies the Phase 1 accuracy regression and the suppression-loss issue, but one important diagnosis is stale:

- the claim that fake OOD tops out around `0.13` no longer matches the active `checkpoints_v3` targets

I inspected `checkpoints_v3/teacher_targets.npz`. The current rich fake-OOD families already cover a much wider EU range:

| Split | Mean teacher EU |
|------|------------------|
| clean val | `0.062` |
| fake mixup | `0.120` |
| fake masked | `0.131` |
| fake patchshuffle | `0.165` |
| fake cutpaste | `0.104` |
| fake heavy_noise | `0.272` |
| fake pixel_permute | `0.209` |
| real SVHN | `0.343` |
| real STL10 | `0.151` |

So the current bottleneck is not simply "the student never sees targets above 0.13". The richer fake-OOD cache already contains high-EU synthetic samples.

## 3.2 Phase 1 in `distill.py` is more aggressive than the summary suggests

In `TinyImageNet/distill.py`, Phase 1 does not hard-freeze earlier blocks. It assigns them a small but nonzero learning rate through `_build_llrd_param_groups()`.

This means the v3 setup was effectively:

- high base LR: `2e-4`
- 4 active top blocks
- low-LR updates to deeper backbone layers as well
- 100 epochs

That is materially more aggressive than the baseline and is consistent with the observed accuracy drop.

## 3.3 Phase 2 cannot adapt the backbone in the current implementation

This matters because one proposed remedy in `E3_ANALYSIS.md` is "unfreeze top 1-2 blocks in Phase 2".

That is not possible without code changes, because:

- `run_phase2()` in `TinyImageNet/distill.py` pre-extracts Phase 2 inputs once and trains on cached tensors
- `DeiTStudent.forward()` in `TinyImageNet/models.py` detaches both `feat` and `probs` before the EU head

So current Phase 2 is strictly head-only.

## 3.4 The frozen Phase 1 representation already contains more OOD signal than the current EU head uses

I ran a quick probe on the v3 Phase 1 checkpoint using feature-space density proxies.

For OOD detection AUROC:

| Dataset | Student entropy | nearest-centroid L2 | 1 - max cosine to centroid |
|--------|------------------|---------------------|----------------------------|
| SVHN | `0.746` | `0.809` | `0.826` |
| STL10 | `0.691` | `0.707` | `0.710` |
| CIFAR-10 | `0.776` | `0.823` | `0.820` |
| CIFAR-100 | `0.788` | `0.829` | `0.830` |

This strongly suggests:

- the backbone feature geometry already contains extra ID-vs-OOD information
- the current EU head and loss do not exploit that information well enough

## 3.5 The v3 Phase 2 loss likely suppresses OOD too aggressively

Current Phase 2 combines:

- tier-asymmetric log-MSE
- pairwise ranking
- clean-ID suppression loss with `suppress_weight=2.0`
- tier margin loss with `margin_weight=0.5`

Observed behavior:

- clean-ID student EU mean improved from `0.084` to `0.074`
- SVHN student EU mean dropped from `0.213` to `0.116`
- SVHN AUROC fell from `0.897` to `0.844`

That is consistent with over-regularization of the head.

## 4. Root-Cause Diagnosis

## 4.1 Root cause of the accuracy regression

Primary cause:

- v3 Phase 1 over-updated the student backbone

Most likely mechanism:

- 4-block unfreeze plus `2e-4` base LR moved the pretrained ViT away from the useful ImageNet prior faster than TinyImageNet supervision could recover

## 4.2 Root cause of poor SVHN and mediocre STL10 separation

Primary causes:

- Phase 2 uses a frozen representation and cannot repair representation mistakes
- Phase 2 collapses heterogeneous fake OOD into one broad tier
- the suppression term is too strong for the current head capacity
- the head has no explicit feature-density or class-manifold signal

Secondary cause:

- the optimization target is mostly clean-val Spearman, which does not guarantee good true-OOD ordering

## 4.3 What is not the main bottleneck right now

These are not the first issues to attack:

- teacher weakness: the teacher already separates SVHN very well
- lack of any high-EU fake targets: the rich cache already provides them
- model capacity alone: the 3-layer head is not obviously the limiting factor

## 5. Strategy Principles

The next experiments should follow four rules.

### Rule 1: Decouple accuracy recovery from OOD-head recovery

Do not keep changing Phase 1 and Phase 2 at the same time.

### Rule 2: Use the strongest available classifier as the Phase 2 base

Phase 2 ablations should start from `checkpoints_rich_fake_ood_12m/student_phase1.pt`, not `checkpoints_v3/student_phase1.pt`.

### Rule 3: Optimize true-OOD behavior with metrics that actually reflect it

For Phase 2, do not choose checkpoints only by clean-ID Spearman.

Track:

- `SVHN` AUROC
- `STL10` AUROC
- clean-ID mean EU
- clean-vs-SVHN mean-EU ratio
- clean-ID Spearman

### Rule 4: Prefer low-risk architectural changes before reworking the whole pipeline

Before attempting Phase 2 backbone unfreezing, first test:

- loss simplification
- fake-OOD composition changes
- explicit feature-density inputs to the EU head

## 6. Concrete Plan

## Stage 0: Lock the baselines

Use these as the comparison anchors.

### Baseline A: stronger classifier, acceptable Phase 2

Directory:

- `TinyImageNet/checkpoints_rich_fake_ood_12m/`

Why:

- best known Phase 1 accuracy: `87.51%`
- Phase 2 Spearman: `0.8407`
- SVHN AUROC: `0.8974`

### Baseline B: current v3

Directory:

- `TinyImageNet/checkpoints_v3/`

Why keep it:

- it demonstrates the regression introduced by the new loss recipe

## Stage 1: Re-run Phase 2 ablations on the strong Phase 1 checkpoint

### Objective

Answer one question cleanly:

> Is the OOD problem mainly caused by the new Phase 2 loss design, independent of the weaker v3 backbone?

### Setup

Start from:

- teacher targets from `checkpoints_rich_fake_ood_12m`
- Phase 1 checkpoint from `checkpoints_rich_fake_ood_12m/student_phase1.pt`

Do not retrain the teacher.

Do not retrain Phase 1 yet.

### Experiments

Run three Phase 2-only variants:

1. `suppress_weight=0.0`, `margin_weight=0.0`
2. `suppress_weight=0.25`, `margin_weight=0.1`
3. `suppress_weight=0.5`, `margin_weight=0.1`

Keep:

- `rank_weight=1.0`
- `eu_sample_alpha=10`
- same cache and same Phase 1 backbone

### Why this comes first

If SVHN recovers immediately on the stronger Phase 1 checkpoint, then the current issue is mostly the Phase 2 loss, not fake-OOD coverage.

### Commands

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maw6
cd /home/maw6/maw6/unc_regression/TinyImageNet

mkdir -p checkpoints_p2_ablate_s0
cp checkpoints_rich_fake_ood_12m/{teacher_targets.npz,ensemble_configs.json,student_phase1.pt} checkpoints_p2_ablate_s0/
cp checkpoints_rich_fake_ood_12m/member_*.pt checkpoints_p2_ablate_s0/

python distill.py --save_dir ./checkpoints_p2_ablate_s0 \
  --phase2_only \
  --p2_epochs 200 \
  --p2_lr 0.003 \
  --rank_weight 1.0 \
  --suppress_weight 0.0 \
  --margin_weight 0.0 \
  --eu_sample_alpha 10.0 \
  --gpu 0 2>&1 | tee checkpoints_p2_ablate_s0/p2.log

python evaluate_student.py --save_dir ./checkpoints_p2_ablate_s0 \
  --gpu 0 2>&1 | tee checkpoints_p2_ablate_s0/eval.log
```

Repeat with:

- `checkpoints_p2_ablate_s025`
- `checkpoints_p2_ablate_s05`

and the corresponding suppression and margin weights.

### Success criteria

This stage is successful if at least one run achieves:

- accuracy still inherited from Phase 1 at around `87.5%`
- clean Spearman `>= 0.82`
- `SVHN AUROC >= 0.89`
- `STL10 AUROC >= 0.76`
- clean/SVHN mean-EU ratio clearly improved over v3

## Stage 2: Re-cache a higher-tail fake-OOD mixture

### Objective

Test whether better fake-OOD composition improves true-OOD separation without changing the teacher or the classifier.

### Rationale

The rich cache has useful high-EU families, but the current composition still allocates substantial budget to weak families such as:

- mixup
- cutpaste
- multi_corrupt

The higher-EU families are more likely to help true-OOD separation:

- heavy_noise
- pixel_permute
- patchshuffle
- masked

### Proposed composition

Recommended starting point:

- mixup: `0.05`
- patchshuffle: `0.25`
- cutpaste: `0.05`
- heavy_noise: `0.35`
- multi_corrupt: `0.00`
- pixel_permute: `0.20`
- masked: remainder `0.10`

### Commands

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maw6
cd /home/maw6/maw6/unc_regression/TinyImageNet

mkdir -p checkpoints_p2_hightail
cp checkpoints_rich_fake_ood_12m/{ensemble_configs.json,student_phase1.pt} checkpoints_p2_hightail/
cp checkpoints_rich_fake_ood_12m/member_*.pt checkpoints_p2_hightail/

python cache_ensemble_targets.py --save_dir ./checkpoints_p2_hightail \
  --data_dir /home/maw6/maw6/unc_regression/data \
  --batch_size 128 \
  --gpu 0 \
  --p2_data_mode fake_ood \
  --fake_ood_mixup_frac 0.05 \
  --fake_ood_patchshuffle_frac 0.25 \
  --fake_ood_cutpaste_frac 0.05 \
  --fake_ood_heavy_noise_frac 0.35 \
  --fake_ood_multi_corrupt_frac 0.00 \
  --fake_ood_pixel_permute_frac 0.20

python distill.py --save_dir ./checkpoints_p2_hightail \
  --phase2_only \
  --p2_epochs 200 \
  --p2_lr 0.003 \
  --rank_weight 1.0 \
  --suppress_weight 0.25 \
  --margin_weight 0.1 \
  --eu_sample_alpha 15.0 \
  --gpu 0 2>&1 | tee checkpoints_p2_hightail/p2.log

python evaluate_student.py --save_dir ./checkpoints_p2_hightail \
  --gpu 0 2>&1 | tee checkpoints_p2_hightail/eval.log
```

### Success criteria

This stage is successful if it improves at least one of:

- `SVHN AUROC`
- `STL10 AUROC`
- clean/SVHN mean-EU ratio

without materially degrading:

- clean accuracy
- clean-ID Spearman

## Stage 3: Add explicit feature-density signals to the EU head

### Objective

Make the EU head use geometric OOD information that is already present in the frozen Phase 1 representation.

### Why this is important

The feature probe showed that simple manifold-distance scores outperform student entropy on SVHN and STL10. That means the current EU head input is too weak.

### Proposed new EU-head inputs

Augment `[feat || softmax(logits)]` with:

- nearest-class-centroid squared L2 distance
- maximum cosine similarity to class centroids
- energy score
- entropy
- logit margin between top-1 and top-2 classes

### Implementation sketch

Modify:

- `TinyImageNet/models.py`
- `TinyImageNet/distill.py`

Add:

- centroid computation from train features after Phase 1
- cached scalar density features for Phase 2
- concatenation of density features into the EU head input

### Expected effect

This should help the EU head distinguish:

- hard ID examples
- corrupted ID
- class-manifold-outside OOD examples such as SVHN

without needing to unfreeze the backbone.

### Decision rule

Only do this if Stages 1 and 2 fail to recover SVHN meaningfully.

## Stage 4: Retrain Phase 1 conservatively

### Objective

Recover classification accuracy without destabilizing the representation.

### Recommended first configuration

Try one of:

1. `p1_unfreeze_blocks=2`, `p1_lr=1e-4`, `p1_epochs=60`
2. `p1_unfreeze_blocks=3`, `p1_lr=1e-4`, `p1_epochs=80`

Do not start with 4-block unfreeze again.

### Why this is later, not earlier

If Stage 1 already restores OOD behavior using the stronger existing Phase 1 checkpoint, then Phase 2 is the main issue and no immediate Phase 1 retraining is needed.

### Commands

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maw6
cd /home/maw6/maw6/unc_regression/TinyImageNet

mkdir -p checkpoints_p1_recover
cp checkpoints_rich_fake_ood_12m/{teacher_targets.npz,ensemble_configs.json} checkpoints_p1_recover/
cp checkpoints_rich_fake_ood_12m/member_*.pt checkpoints_p1_recover/

python distill.py --save_dir ./checkpoints_p1_recover \
  --p1_epochs 80 \
  --p1_lr 1e-4 \
  --p1_wd 0.05 \
  --warmup 5 \
  --p1_unfreeze_blocks 3 \
  --p1_backbone_lr_factor 0.01 \
  --llrd 0.75 \
  --alpha 0.7 \
  --tau 2.0 \
  --gpu 0 2>&1 | tee checkpoints_p1_recover/train.log
```

If this recovers accuracy, then rerun the best Phase 2 recipe from Stages 1-3 on top of this checkpoint.

### Success criteria

Stage 4 is successful if:

- Phase 1 best val accuracy reaches at least `87.5%`
- ideally reaches `>= 88.0%`
- train-vs-val gap stays modest

## Stage 5: Optional architectural change, only if needed

### Objective

Allow slight backbone adaptation during Phase 2.

### Important constraint

This is not possible in the current codebase without refactoring, because:

- Phase 2 trains on pre-extracted tensors
- the EU path detaches backbone features

### Required code changes

To support Phase 2 partial unfreezing, the following must change:

1. stop pre-extracting all Phase 2 features
2. train on image batches directly in Phase 2
3. remove `feat.detach()` for the EU branch when Phase 2 unfreezing is enabled
4. add a very small LR for the top 1-2 blocks, for example `1e-5`

### When to do this

Only after:

- Stage 1 loss ablations
- Stage 2 fake-OOD composition changes
- Stage 3 density-feature augmentation

have all failed to recover the desired OOD behavior.

## 7. Monitoring Instructions

## 7.1 What to monitor during Phase 1

The key metric is:

- validation accuracy

Secondary metrics:

- training accuracy
- train loss
- whether the best checkpoint arrives early and then degrades

### Good signs

- val accuracy exceeds `87.5%`
- train loss decreases smoothly
- no strong late-epoch overfitting

### Bad signs

- val accuracy stalls below `87.5%`
- train accuracy rises while val accuracy stays flat
- best checkpoint appears very early and later epochs only drift

### Simple interpretation guide

- low train and low val: underfitting or LR too small
- high train and flat val: overfitting or too much backbone drift
- unstable val: LR too high or too many active blocks

## 7.2 What to monitor during Phase 2

Do not monitor only clean-ID Spearman.

Track all of:

- clean-ID Spearman
- clean-ID Pearson
- clean mean EU
- corrupted mean EU
- fake-OOD mean EU
- SVHN mean EU
- STL10 mean EU
- clean-vs-SVHN AUROC
- clean-vs-STL10 AUROC
- clean/SVHN mean-EU ratio

### Good signs

- clean mean EU stays close to teacher
- SVHN mean EU increases relative to v3
- SVHN AUROC improves without collapsing clean Spearman

### Bad signs

- clean mean EU keeps falling while OOD means also fall
- Spearman improves but SVHN AUROC worsens
- OOD means collapse toward clean ID

### Core lesson

For this task, Phase 2 model selection should be based on a multi-metric gate, not on Spearman alone.

## 8. Evaluation Instructions

## 8.1 Standard evaluation command

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maw6
cd /home/maw6/maw6/unc_regression/TinyImageNet

python evaluate_student.py --save_dir ./CHECKPOINT_DIR \
  --gpu 0 2>&1 | tee ./CHECKPOINT_DIR/eval.log
```

The report is also written to:

- `TinyImageNet/result.md`

If multiple runs are being compared, copy it immediately after each evaluation so it is not overwritten.

Example:

```bash
cp /home/maw6/maw6/unc_regression/TinyImageNet/result.md \
  /home/maw6/maw6/unc_regression/TinyImageNet/checkpoints_p2_ablate_s0/result_snapshot.md
```

## 8.2 Minimum comparison table to maintain

For every run, record:

| Run | P1 acc | Spearman | clean EU mean | SVHN EU mean | clean/SVHN ratio | SVHN AUROC | STL10 AUROC | CIFAR-10 AUROC | DTD AUROC |
|-----|--------|----------|---------------|--------------|------------------|------------|-------------|----------------|-----------|

This table should be updated after each experiment.

## 8.3 Recommended gating criteria for selecting the next default

Promote a run to "new default candidate" only if all are reasonably satisfied:

- Phase 1 accuracy `>= 87.5%`
- clean-ID Spearman `>= 0.82`
- `SVHN AUROC >= 0.89`
- `STL10 AUROC >= 0.76`
- clean mean EU not obviously over-suppressed
- no major regression on CIFAR-10, CIFAR-100, DTD

## 9. Recommended Execution Order

Use this exact order.

1. Stage 1: Phase 2-only ablations on `checkpoints_rich_fake_ood_12m`
2. Stage 2: higher-tail fake-OOD recache plus Phase 2-only retraining
3. Stage 3: density-feature augmentation to the EU head
4. Stage 4: conservative Phase 1 retraining
5. Stage 5: true Phase 2 partial-unfreezing refactor only if still needed

## 10. Final Recommendation

The most likely fastest path is:

1. revert to the stronger Phase 1 backbone from `checkpoints_rich_fake_ood_12m`
2. simplify Phase 2 regularization
3. emphasize higher-tail fake-OOD families
4. if needed, add centroid or density features to the EU head

The current evidence suggests the main missed opportunity is not lack of any OOD signal in the teacher targets. It is that the student EU head is not using the OOD structure already present in the frozen representation.
