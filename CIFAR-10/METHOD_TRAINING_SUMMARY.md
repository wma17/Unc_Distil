# Bayesian Uncertainty Distillation on CIFAR-10

This note summarizes the method and training pipeline implemented in this folder, based on the actual code in `models.py`, `train_ensemble.py`, `cache_ensemble_targets.py`, `distill.py`, and the saved artifacts in `checkpoints/`.

## Paper-ready method summary

We distill Bayesian uncertainty from a deep ensemble teacher into a single ResNet18 student on CIFAR-10. The teacher is a diversified ensemble of independently trained CIFAR-ResNet18 models. For an input `x`, each member predicts a class distribution `p_m(y|x)`, and the ensemble predictive distribution is the mean `p_bar(y|x) = (1/M) sum_m p_m(y|x)`. Uncertainty is decomposed as total uncertainty `TU = H[p_bar(y|x)]`, aleatoric uncertainty `AU = (1/M) sum_m H[p_m(y|x)]`, and epistemic uncertainty `EU = TU - AU`. The student is trained in two phases. In Phase 1, the student backbone and classifier are distilled from the ensemble using a standard knowledge distillation loss that mixes hard-label cross-entropy and softened teacher targets. In Phase 2, the backbone and classifier are frozen, and a lightweight scalar uncertainty head is trained to regress the teacher EU on a mixed dataset containing clean CIFAR-10, corrupted CIFAR-10, and OOD-style samples. The Phase 2 objective combines log-space MSE with a pairwise ranking loss so the student learns both the scale and relative ordering of teacher EU. This yields a single forward-pass model that preserves most teacher accuracy while providing an explicit approximation of ensemble epistemic uncertainty.

## 1. Teacher: diversified deep ensemble

### Architecture

- Each teacher member is a CIFAR-adapted ResNet18.
- The ImageNet-style `7x7` stem and max-pool are replaced by a single `3x3` conv with stride 1 and padding 1.
- A dropout layer is inserted before the final FC classifier.

### Diversity strategy

Each member is trained independently with diversity introduced through:

- Different random seed.
- Different augmentation subset.
- Different label smoothing.
- Different dropout before the classifier.
- Different training-set fraction (bagging-style subsampling).
- Different learning rate and weight decay.
- Different LR schedule.
- Different classifier-head initialization scale.

The sampled ranges in `train_ensemble.py` are:

- Augmentations: random subset of `colorjitter`, `grayscale`, `rotation`, `cutout`, `randerasing`, or `autoaugment`.
- Label smoothing: uniform in `[0.0, 0.05]`.
- Dropout rate: one of `{0.0, 0.05, 0.1}`.
- Data fraction: uniform in `[0.8, 1.0]`.
- LR: uniform in `[0.05, 0.15]`.
- Weight decay: one of `{3e-4, 5e-4, 1e-3}`.
- LR schedule: `cosine` or `step`.
- Head-init scale: log-uniform in `[0.5, 1.5]`.

### Teacher training details

- Dataset: CIFAR-10 train/test.
- Base train transform: random crop with padding 4 + random horizontal flip.
- Normalization:
  - mean = `(0.4914, 0.4822, 0.4465)`
  - std = `(0.2470, 0.2435, 0.2616)`
- Optimizer: SGD with momentum `0.9`.
- Epochs: `200`.
- Batch size: `128`.
- Loss: cross-entropy with per-member label smoothing.

### Saved teacher artifacts in this folder

The current `checkpoints/` directory contains five teacher checkpoints:

- `member_0.pt`: `94.42%`
- `member_1.pt`: `95.56%`
- `member_2.pt`: `95.22%`
- `member_3.pt`: `94.41%`
- `member_4.pt`: `95.07%`

The cache/evaluation code loads **all** `member_*.pt` files, so the saved teacher used for distillation is effectively a **5-member ensemble**.

## 2. Teacher uncertainty targets

For each input, the ensemble produces per-member softmax probabilities. The cached targets are:

- `mean_probs`: ensemble-mean class probabilities.
- `TU = H(mean_probs)`.
- `AU = mean_m H(p_m)`.
- `EU = TU - AU`.

These targets are cached by `cache_ensemble_targets.py` into `checkpoints/teacher_targets.npz`.

## 3. Student model

### Backbone and classifier

- The student uses the same CIFAR-ResNet18 backbone as the teacher.
- It has a standard classification head producing logits over 10 CIFAR-10 classes.

### Explicit EU head

The student also has a scalar EU head:

- Input to EU head: concatenation of
  - the final 512-d pooled feature vector, and
  - the student's own softmax probabilities.
- MLP structure:
  - `Linear(512 + 10 -> 128)`
  - `ReLU`
  - `Linear(128 -> 1)`
  - `Softplus`

Important implementation detail:

- The softmax probabilities are **detached** before entering the EU head.
- Therefore, the EU regression loss does **not** backpropagate into the classifier.

## 4. Two-phase distillation

### Phase 1: classification distillation

Phase 1 trains the full student on clean CIFAR-10 using teacher soft labels and hard labels together:

`L_p1 = (1 - alpha) * CE(y, z_s) + alpha * tau^2 * KL(softmax(z_t / tau) || softmax(z_s / tau))`

where:

- `alpha = 0.7`
- `tau = 4.0`

Phase 1 settings:

- Data: clean CIFAR-10 only.
- Train transform: random crop + horizontal flip.
- Optimizer: SGD with momentum `0.9`, weight decay `5e-4`.
- LR: `0.1`.
- Schedule: 10-epoch linear warmup, then cosine decay.
- Epochs: `200`.

### Phase 2: EU regression

After Phase 1:

- The backbone and classifier are frozen.
- Only the EU head is trainable.
- The EU head is re-initialized before training.

Phase 2 loss:

`L_p2 = log1p_MSE(EU_s, EU_t) + beta * PairwiseRankingLoss(EU_s, EU_t)`

with:

- `beta = 1.0`
- `log1p_MSE` applied in `log(1 + x)` space to stabilize the long-tailed EU distribution.
- Pairwise ranking loss applied on random sample pairs inside a batch, with:
  - `n_pairs = 256`
  - margin `= 0.05`

Phase 2 settings:

- Optimizer: Adam.
- LR: `1e-3`.
- Schedule: cosine decay.
- Epochs: `100`.
- Trainable parameters: `67,073 / 11,241,035`.

## 5. Phase 2 training distribution

The student uncertainty head is trained on a mixed dataset designed to teach EU across clean, shifted, and OOD-like inputs.

Target composition in code:

- `50%` clean CIFAR-10 train.
- `25%` corrupted CIFAR-10.
- `25%` OOD or fake OOD.

### Corrupted CIFAR-10

Three deterministic corruptions are applied:

- Gaussian noise.
- Gaussian blur.
- Low contrast.

The corruption seed is fixed at `2026`, so the same images can be regenerated instead of stored separately.

### Real OOD option supported by code

The code supports a real-OOD Phase 2 mode:

- SVHN test.
- CIFAR-100 test.

### Fake OOD option supported by code

The code also supports a synthetic Phase 2 mode:

- Mixup images.
- Masked images.

Fake OOD settings:

- Seed: `2027`.
- Mixup lambdas: `[0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]`.
- Mask styles:
  - `random_block`
  - `random_pixel`
  - `center_crop_mask`
  - `multi_block`
  - `border_mask`
- Mask rates: `[0.3, 0.5, 0.7, 0.8]`.

## 6. What the current saved run actually used

The current `checkpoints/teacher_targets.npz` shows:

- `p2_data_mode = fake_ood`
- `fake_ood_mixup_frac = 0.5`

So the **currently saved student** in this folder was trained with:

- `50,000` clean CIFAR-10 samples
- `24,999` corrupted CIFAR-10 samples
- `25,000` fake OOD samples
  - `12,500` mixup
  - `12,500` masked

Total Phase 2 training size:

- `99,999` samples

The shifted split is `24,999` rather than `25,000` because the code divides the corrupted portion evenly across three corruption types.

## 7. Current checkpoint metrics

### Phase 1 checkpoint

From `checkpoints/student_phase1.pt`:

- Best clean CIFAR-10 test accuracy: `95.81%`
- Best epoch saved: `192`

### Final Phase 2 checkpoint

From `checkpoints/student.pt`:

- Best checkpoint saved at epoch `95`
- Clean CIFAR-10 EU Pearson correlation: `0.7151`
- Clean CIFAR-10 EU Spearman correlation: `0.6902`

### Evaluation snapshot for current saved artifacts

From the saved fake-OOD run:

- Teacher ensemble accuracy on CIFAR-10 test: `96.37%`
- Distilled student accuracy on CIFAR-10 test: `95.81%`
- Clean CIFAR-10 test EU mean:
  - teacher: `0.0493`
  - student: `0.0697`

Examples of OOD detection AUROC with student EU:

- CIFAR-10 vs SVHN: `0.9768`
- CIFAR-10 vs CIFAR-100: `0.9094`
- CIFAR-10 vs MNIST: `0.9389`
- CIFAR-10 vs FashionMNIST: `0.9392`
- CIFAR-10 vs DTD: `0.9428`

This indicates the student preserves most classification accuracy while learning an explicit uncertainty score that is competitive with, and on some datasets slightly better than, the ensemble EU for OOD detection.

## 8. Reproducible training sequence

The intended pipeline is:

```bash
python train_ensemble.py --num_members 5 --epochs 200 --gpu 0
python cache_ensemble_targets.py --save_dir ./checkpoints --gpu 0 --p2_data_mode fake_ood
python distill.py --save_dir ./checkpoints --gpu 0
python evaluate_student.py --save_dir ./checkpoints --gpu 0
```

To use real OOD during Phase 2 instead:

```bash
python cache_ensemble_targets.py --save_dir ./checkpoints --gpu 0 --p2_data_mode ood
python distill.py --save_dir ./checkpoints --gpu 0
```

## 9. Important artifact note

There is a small reproducibility inconsistency in the current folder:

- `checkpoints/ensemble_configs.json` contains only **3** sampled configs.
- `checkpoints/` contains **5** teacher checkpoints.
- The cache/evaluation scripts use all `member_*.pt` files they find.

Also, `member_3.pt` and `member_4.pt` appear to carry duplicated config metadata from earlier members, which suggests leftover checkpoints from an earlier run. For the currently saved student and cached targets, the effective teacher is still the 5-member ensemble that the loader reads from disk. If exact reporting of teacher composition matters for the paper, clean retraining with a fresh checkpoint directory is recommended.

## 10. Suggested paper wording

Short version:

> We first train a diversified deep ensemble of CIFAR-ResNet18 classifiers and use the ensemble predictive mutual information as the teacher epistemic uncertainty target. We then distill the ensemble into a single student network using a two-stage procedure: classification distillation on clean CIFAR-10 followed by uncertainty-head regression on a mixture of clean, corrupted, and OOD-style samples. The student outputs both class probabilities and an explicit scalar estimate of epistemic uncertainty, enabling near-ensemble uncertainty behavior at single-model inference cost.

Longer version:

> Our teacher is a deep ensemble of independently trained CIFAR-ResNet18 models with diversity induced through randomized augmentation policies, label smoothing, dropout, bagging-style data subsampling, learning-rate settings, and classifier-head initialization. For each input, we compute the ensemble predictive distribution and decompose uncertainty into total, aleatoric, and epistemic terms, where epistemic uncertainty is the mutual information between predictions and model identity. A single CIFAR-ResNet18 student is distilled in two phases. In the first phase, the full student is trained with a mixture of hard-label cross-entropy and soft-label knowledge distillation from the ensemble. In the second phase, the backbone and classifier are frozen and only a lightweight uncertainty head is optimized to predict the teacher epistemic uncertainty. This second-stage training uses a mixture of clean CIFAR-10, distribution-shifted CIFAR-10 corruptions, and OOD-style samples, and combines log-space MSE with a pairwise ranking objective so that the student matches both the magnitude and ordering of teacher uncertainty. The final model retains high classification accuracy while approximating ensemble-based Bayesian uncertainty with a single forward pass.
