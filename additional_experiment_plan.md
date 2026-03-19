# Experiment Plan: Amortized Uncertainty Distillation
> Target venue: NeurIPS 2026 (oral-level)
> Hardware: 2× RTX 4090 (24 GB VRAM each)

---

## 0. GPU Feasibility Assessment

### Semantic Segmentation (Pascal VOC 2012, SegFormer-B2)
| Component | VRAM (per GPU) | Est. train time |
|---|---|---|
| SegFormer-B2 fine-tune (LoRA, bs=16) | ~8 GB | ~10 h/member |
| 5-member ensemble (2 GPUs, sequential) | ~8 GB | ~25 h total |
| Student Phase 1 + Phase 2 | ~10 GB | ~12 h total |
| **Total** | **within 24 GB** | **~2.5 days** |

**Verdict: 2× RTX 4090 is sufficient.** SegFormer-B2 with LoRA is memory-light. You can train 2 ensemble members in parallel (one per GPU) and still have headroom.

### SST-2 NLP (BERT-base + DistilBERT)
| Component | VRAM | Est. train time |
|---|---|---|
| BERT-base fine-tune (LoRA, bs=32) | ~5 GB | <1 h/member |
| 5-member ensemble | ~5 GB | ~4 h total |
| Student Phase 1 + Phase 2 | ~6 GB | ~3 h total |
| **Total** | **trivially within budget** | **<1 day** |

**Verdict: Trivially feasible. SST-2 is the easiest experiment in the entire plan.**

---

## 1. Full Experiment Suite Overview

| ID | Task Type | Dataset | Teacher Backbone | Student Backbone | Status |
|---|---|---|---|---|---|
| E1 | Image Classification | MNIST | ResNet18 (scratch) | ResNet18 (scratch) | ✅ Done |
| E2 | Image Classification | CIFAR-10 | ResNet18 (scratch) | ResNet18 (scratch) | ✅ Done |
| E3 | Image Classification | TinyImageNet | DeiT-S (pretrained+LoRA) | DeiT-S (pretrained) | ✅ Done |
| E4 | Semantic Segmentation | Pascal VOC 2012 | SegFormer-B2 (pretrained+LoRA) | SegFormer-B2 (pretrained) | 🔲 New |
| E5 | Text Classification | SST-2 | BERT-base (pretrained+LoRA) | DistilBERT (pretrained) | 🔲 New |

---

## 2. Existing Experiments — What to Add (E1, E2, E3)

You already have accuracy, EU correlation, OOD AUROC, and uncertainty decomposition. The following metrics are missing and **must be added for oral-level review**:

### 2.1 Metrics to retrofit onto E1/E2/E3

**Calibration (add to all):**
- **ECE-15**: 15-bin expected calibration error on the clean test set.
  Formula: `ECE = Σ_b (|B_b|/n) · |acc(B_b) − conf(B_b)|`
  Report for teacher ensemble and student separately.
- **NLL**: Mean negative log-likelihood on clean test set. `NLL = -1/n Σ log p_student(y_i | x_i)`
- **Brier Score**: `BS = 1/n Σ ||p_student(·|x_i) − e_{y_i}||²` (where `e_{y_i}` is the one-hot label)

**Selective Prediction (add to all):**
- **AURC** (Area Under Risk-Coverage Curve): Sort test samples in ascending order of student EU. For each coverage threshold τ ∈ [0, 1], compute accuracy on the top-τ fraction of samples (lowest EU). Plot and report area. Lower is better.
- **Sparsification curve**: Same as AURC but plot accuracy vs. fraction rejected. The curve should show monotone accuracy improvement as high-EU samples are removed. Also compute the **Oracle gap** (difference from oracle sparsification that rejects misclassified samples first).

**Inference Efficiency (add to E2 and E3 at minimum):**
- **Throughput** (samples/sec) on a single 4090 for: (a) full K=5 ensemble, (b) student single forward pass, (c) single ensemble member.
- Report speedup factor (student / ensemble).
- Measure with batch size 256, averaged over 100 batches after warmup.

---

## 3. New Experiment E4: Semantic Segmentation

### 3.1 Why Segmentation

All existing experiments use flat classification output. Segmentation forces the EU head to produce **spatially distributed uncertainty maps**, demonstrating that the framework generalizes to structured prediction tasks. It also directly motivates the medical imaging and autonomous driving applications mentioned in the introduction.

### 3.2 Dataset

**Pascal VOC 2012** (augmented split via SBD):
- Train: 10,582 images (VOC 2012 train + SBD augmentation)
- Val: 1,449 images (VOC 2012 val, no overlap with train)
- Classes: 20 foreground + 1 background = 21 classes
- Image size: resize shorter side to 512, random crop to 512×512 during training
- Source: standard `torchvision.datasets.VOCSegmentation` + SBD

**OOD datasets for image-level detection:**
- Near-OOD: **MS-COCO val2017** images containing only COCO-exclusive categories (filter out the 80 COCO categories that overlap with VOC 20; use only the ~30 COCO-exclusive ones, e.g., toaster, hair drier, toilet). This gives semantically similar images with novel object types.
- Far-OOD: **DTD** (Describable Textures Dataset) — purely textural, no objects, very different from VOC.

### 3.3 Teacher

- **Architecture**: SegFormer-B2, pretrained on ImageNet-22K (MIT-B2 backbone).
- **LoRA configuration**: Rank r=16, alpha=32, applied to all attention projection matrices (Q, K, V, Out) in the transformer blocks. Keep the MLP layers and decode head fully trainable.
- **Ensemble size**: K=5 members, each trained with a different random seed and a subset of the diversity strategies below.
- **Diversity strategy** (same spirit as E2 teacher):
  - Different random seed per member.
  - Different LoRA init scale (log-uniform in [0.5, 1.5]).
  - Different label smoothing (uniform in [0.0, 0.05]).
  - Different data augmentation subset: random horizontal flip + one of {color jitter, random scale, random rotation ±10°}.
  - Different weight decay: one of {1e-4, 5e-4, 1e-3}.
- **Training details**:
  - Optimizer: AdamW, lr=6e-5, weight decay=0.01
  - Schedule: poly LR with power 0.9, 40,000 iterations
  - Batch size: 16 (on single 4090)
  - Loss: cross-entropy with per-member label smoothing, ignore index=255 (boundary pixels)

**Teacher uncertainty computation (per pixel):**
For each pixel (i,j) and ensemble member m, compute softmax class distribution `p_m(y|x,i,j)`.
- Per-pixel TU: `TU(i,j) = H[mean_m p_m(y|x,i,j)]`
- Per-pixel AU: `AU(i,j) = mean_m H[p_m(y|x,i,j)]`
- Per-pixel EU: `EU(i,j) = TU(i,j) - AU(i,j)` (mutual information)

Cache these per-pixel targets for the full training set.

### 3.4 Student

- **Architecture**: SegFormer-B2 initialized from the **same ImageNet-22K pretrained weights** as the teacher (not from any teacher checkpoint — Phase 1 KD handles knowledge transfer).
- **EU head** (spatial, replaces scalar MLP):
  ```
  Input: frozen encoder output — feature map F of shape (B, 512, H/32, W/32)
  Head:
    Conv2d(512 → 128, kernel=1, bias=False) → BatchNorm2d → ReLU
    Conv2d(128 → 1, kernel=1) → Softplus
  Output: per-pixel EU map, upsampled bilinearly to (H, W)
  ```
  The head takes features from the last stage of the MIT-B2 encoder (before the decode head). The student's own softmax logit map (detached) is concatenated spatially if it fits in memory; otherwise use encoder features alone.

### 3.5 Two-Phase Distillation

**Phase 1 — Segmentation distillation:**
- Full student (backbone + decode head) trains on VOC 2012 train.
- Loss: `L_p1 = (1-α)·CE(y, z_s) + α·τ²·KL(softmax(z_T/τ) || softmax(z_s/τ))`
  - Applied per-pixel. Ignore boundary pixels (label=255).
  - α=0.7, τ=4.0 (same as E2).
- Optimizer: AdamW, lr=6e-5, poly schedule, 40,000 iterations, batch=16.

**Phase 2 — EU head training:**
- Freeze backbone and decode head. Only EU head is trainable.
- Reinitialize EU head before Phase 2.
- Training data (three-tier curriculum, per image):

| Tier | Fraction | Source |
|---|---|---|
| 1 — Clean | 50% | VOC 2012 train images |
| 2 — Corrupted | 25% | VOC train + Gaussian noise (σ=0.1), Gaussian blur (σ=2), low contrast (×0.5) |
| 3 — Synthetic OOD | 25% | Mixup of random VOC image pairs (λ ∈ {0.2, 0.5, 0.8}) + random block masking (rate ∈ {0.3, 0.5, 0.7}) |

- Loss per pixel: `L_unc = L_log + β·L_rank`
  `L_log = mean_{i,j} [log(1+EU_s(i,j)) - log(1+EU_T(i,j))]²`
  `L_rank`: pairwise ranking over randomly sampled pixel pairs within a batch (n_pairs=512 per image, margin=0.05)
- Optimizer: Adam, lr=1e-3, cosine schedule, 20,000 iterations, β=1.0.

### 3.6 Evaluation Metrics for E4

**Standard segmentation:**
| Metric | Description |
|---|---|
| mIoU | Mean intersection over union over 21 classes (teacher ensemble vs. student) |
| Pixel accuracy | Fraction of correctly labeled pixels |

**Uncertainty fidelity:**
| Metric | Description |
|---|---|
| Per-pixel EU Pearson | Pearson correlation between student per-pixel EU and teacher per-pixel EU on VOC val |
| Per-pixel EU Spearman | Spearman rank correlation (same) |
| Boundary vs. interior EU ratio | Mean EU at semantic boundary pixels (within 3px of class boundary) divided by mean EU at interior pixels. Report for both teacher and student. Should be >1 — higher EU at boundaries. |

**Selective segmentation:**
| Metric | Description |
|---|---|
| Coverage-mIoU curve | Sort pixels by student EU ascending. For coverage τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, report mIoU on accepted pixels. Plot curve. |
| AURC (segmentation) | Area under the coverage-mIoU curve. Compare student EU vs. teacher EU vs. softmax entropy (1-max probability) as the selection signal. |

**Image-level OOD detection:**
| Metric | Description |
|---|---|
| Image-level OOD AUROC | Use mean per-pixel EU as a scalar image score. ID = VOC val. OOD = COCO-exclusive images and DTD. Report AUROC for student EU vs. teacher EU vs. max softmax prob. |

**Calibration:**
| Metric | Description |
|---|---|
| Pixel-level ECE-15 | Bin pixels by max softmax confidence, measure alignment between confidence and accuracy. |
| Pixel-level NLL | Mean NLL per pixel on VOC val. |

---

## 4. New Experiment E5: NLP — SST-2 Sentiment Classification

### 4.1 Why SST-2

It is the simplest task that demonstrates cross-modal generality of the framework. BERT-based models on SST-2 are a well-understood benchmark. The LoRA ensemble approach mirrors E3 (TinyImageNet + DeiT), making the experimental design internally consistent.

### 4.2 Dataset

**In-distribution:**
- **SST-2** (Stanford Sentiment Treebank, binary: positive/negative)
- Train: 67,349 sentences. Dev: 872 sentences. Test labels not public — use dev as test.
- Source: `datasets` library (`glue`, `sst2`).

**Distribution-shifted (Phase 2 Tier 2):**
- Character-level perturbations on SST-2 train: random character swap (2%), random character insertion (2%), keyboard-adjacent substitution (2%). Use `nlpaug` library.
- Word-level perturbations: synonym substitution via WordNet for 15% of tokens per sentence.

**OOD datasets (evaluation only):**
| Dataset | Type | Description |
|---|---|---|
| IMDB | Near-OOD | Same task (binary sentiment), different domain (movie reviews, longer text) |
| Yelp Polarity | Near-OOD | Same task (binary sentiment), different domain (restaurant reviews) |
| Amazon Polarity | Near-OOD | Same task, e-commerce domain |
| AG News | Far-OOD | Topic classification (4 classes), unrelated to sentiment |
| 20 Newsgroups (binary subset) | Far-OOD | News text, very different distribution |

### 4.3 Teacher

- **Architecture**: BERT-base-uncased (110M params), pretrained from HuggingFace.
- **LoRA configuration**: Rank r=8, alpha=16, applied to Q and V projection matrices in all 12 attention layers. Keep [CLS] classification head fully trainable.
- **Ensemble size**: K=5.
- **Diversity strategy**:
  - Different random seed per member.
  - Different LoRA init scale.
  - Different label smoothing ∈ {0.0, 0.02, 0.05}.
  - Different dropout rate on BERT attention ∈ {0.1, 0.15}.
  - Different learning rate ∈ {1e-4, 2e-4, 3e-4}.
- **Training details**:
  - Optimizer: AdamW, lr sampled per member, weight_decay=0.01.
  - Schedule: linear warmup (10% of steps) + linear decay, 5 epochs.
  - Batch size: 32. Max sequence length: 128.
  - Loss: cross-entropy with per-member label smoothing.

**Teacher uncertainty computation:**
For each sentence, collect [CLS] softmax from each member. Compute:
- TU: `H[mean_m p_m(y|x)]`
- AU: `mean_m H[p_m(y|x)]`
- EU: `TU - AU`

Cache targets for SST-2 train + all perturbation variants.

### 4.4 Student

- **Architecture**: DistilBERT-base-uncased (66M params), initialized from pretrained HuggingFace checkpoint. DistilBERT has 6 layers vs. BERT's 12, making it a natural compression target.
- **EU head** (same structure as E2, operating on [CLS] representation):
  ```
  Input: [CLS] hidden state (768-d) concatenated with student softmax probs (2-d) [detached]
  Head:
    Linear(770 → 128) → ReLU
    Linear(128 → 1) → Softplus
  Output: scalar EU estimate
  ```

### 4.5 Two-Phase Distillation

**Phase 1 — Classification distillation:**
- Full student trains on SST-2 train.
- Loss: `L_p1 = (1-α)·CE(y, z_s) + α·τ²·KL(softmax(z_T/τ) || softmax(z_s/τ))`
  - α=0.7, τ=4.0.
- Optimizer: AdamW, lr=2e-4, linear warmup (10%) + cosine decay, 10 epochs, batch=32.

**Phase 2 — EU head training:**
- Freeze student backbone and classifier. Only EU head trainable.
- Training data (three-tier):

| Tier | Fraction | Source |
|---|---|---|
| 1 — Clean | 50% | SST-2 train |
| 2 — Corrupted | 25% | Character-level + word-level perturbations of SST-2 train |
| 3 — Synthetic OOD | 25% | Sentence mixup (interpolate BERT embeddings of two random SST-2 sentences with λ ∈ {0.2, 0.5, 0.8}) + random token masking (mask rate ∈ {0.3, 0.5, 0.7}) |

- Loss: `L_unc = L_log + β·L_rank` (same formulas as E2 Phase 2)
- Optimizer: Adam, lr=1e-3, cosine schedule, 50 epochs, β=1.0.

### 4.6 Evaluation Metrics for E5

| Metric | Description |
|---|---|
| Accuracy | On SST-2 dev set. Teacher (ensemble) vs. student. |
| ECE-15 | 15-bin calibration error on SST-2 dev. |
| NLL | Mean NLL on SST-2 dev. |
| Brier Score | On SST-2 dev. |
| EU Pearson/Spearman | Student vs. teacher EU on clean SST-2 dev, character-perturbed SST-2, and each OOD dataset. |
| OOD AUROC | Using student EU as score: SST-2 (neg) vs. each OOD dataset (pos). Compare: teacher EU, student EU, student entropy (1-max softmax), single BERT member entropy. |
| AURC | Selective prediction on SST-2 dev: sort by EU, report accuracy vs. coverage curve and its area. |
| Throughput | Sentences/sec for K=5 ensemble vs. DistilBERT student (single pass). |

---

## 5. Full Metrics Specification

This section consolidates all metrics across all experiments, specifying what is reported where.

### 5.1 Main Results Table (per experiment)

| Metric | E1 MNIST | E2 CIFAR-10 | E3 TinyImgNet | E4 VOC | E5 SST-2 |
|---|---|---|---|---|---|
| Accuracy (teacher) | ✓ | ✓ | ✓ | mIoU | ✓ |
| Accuracy (student) | ✓ | ✓ | ✓ | mIoU | ✓ |
| ECE (student) | ✓ | ✓ | ✓ | pixel-ECE | ✓ |
| NLL (student) | ✓ | ✓ | ✓ | pixel-NLL | ✓ |
| Brier Score (student) | — | ✓ | ✓ | — | ✓ |
| EU Pearson (clean) | ✓ | ✓ | ✓ | pixel ✓ | ✓ |
| EU Spearman (clean) | ✓ | ✓ | ✓ | pixel ✓ | ✓ |
| OOD AUROC (best OOD set) | ✓ | ✓ | ✓ | ✓ | ✓ |
| AURC | — | ✓ | ✓ | ✓ | ✓ |
| Throughput speedup | — | ✓ | ✓ | ✓ | ✓ |

### 5.2 OOD Detection Tables (per experiment)

Report AUROC for each method × OOD dataset pair. **Methods** (rows):
1. Teacher EU (full ensemble)
2. **Student EU** (ours — learned EU head)
3. Student softmax entropy `H[p_student(·|x)]`
4. Student 1 − max softmax probability
5. Single ensemble member entropy (baseline)
6. [UQ Baselines TBD by you: MC Dropout, EDL, Laplace]

**OOD datasets** (columns) — use the split below consistently:

| Experiment | Seen OOD (in Phase 2) | Unseen OOD | Shift |
|---|---|---|---|
| E2 CIFAR-10 | SVHN, CIFAR-100 | MNIST, FashionMNIST, STL10, DTD | Corrupted CIFAR-10 |
| E3 TinyImageNet | [define when run] | ImageNet-O, iNaturalist, Textures | TinyImageNet-C |
| E4 VOC | COCO-exclusive (image-level) | DTD | — |
| E5 SST-2 | IMDB, Yelp | Amazon, AG News, 20NG | Perturbed SST-2 |

### 5.3 Calibration Table

For E2, E3, E5, report a single table:

| Method | ECE ↓ | NLL ↓ | Brier ↓ |
|---|---|---|---|
| Teacher ensemble | | | |
| Student (ours) | | | |
| Single member | | | |
| [MC Dropout, EDL, Laplace — TBD] | | | |

### 5.4 Selective Prediction (AURC) Table

For E2, E3, E5, report:

| Method (selection score) | AURC ↓ | @90% coverage acc ↑ | @80% coverage acc ↑ |
|---|---|---|---|
| Teacher EU | | | |
| Student EU (ours) | | | |
| Student entropy | | | |
| 1-MaxProb | | | |
| Oracle | | | |
| Random | | | |

### 5.5 Efficiency Table

Report once, using E2 (CIFAR-10) and E3 (TinyImageNet) and E5 (SST-2):

| Model | Params | VRAM (inf) | Throughput (samples/sec) | FLOPs |
|---|---|---|---|---|
| Ensemble (K=5) | 5× | 5× | 1× (baseline) | 5× |
| Student (ours) | 1× + tiny head | ~1× | ~5× | ~1× |
| Single member | 1× | 1× | ~5× | ~1× |
| MC Dropout (K=5 passes) | 1× | 1× | ~5× (serial) | ~5× |

---

## 6. Ablation Studies

### 6.1 Scope

Run all ablations primarily on **E2 (CIFAR-10)** — it is fast, reproducible, and baselines already exist. For the **Phase 2 curriculum ablation** (the most important design choice), additionally validate on **E5 (SST-2)** to confirm it generalizes. You do **not** need to run every ablation on every dataset for an oral submission.

### 6.2 Ablation A: Phase 2 Training Curriculum

**Question**: Does the three-tier curriculum matter? Can you get away with less?

| Condition | Tier 1 (clean) | Tier 2 (corrupted) | Tier 3 (OOD synth) |
|---|---|---|---|
| A1 | ✓ | — | — |
| A2 | ✓ | ✓ | — |
| A3 (full, ours) | ✓ | ✓ | ✓ |

**Metrics**: EU Pearson (clean), OOD AUROC (SVHN, CIFAR-100, unseen average), AURC.
**Run on**: E2 (CIFAR-10) + E5 (SST-2) for cross-domain validation.

### 6.3 Ablation B: Phase 2 Loss Components

**Question**: Are both `L_log` and `L_rank` necessary?

| Condition | Loss |
|---|---|
| B1 | Standard MSE (no log transform) |
| B2 | `L_log` only (log-space MSE, no ranking) |
| B3 | `L_rank` only (pairwise ranking, no magnitude supervision) |
| B4 (ours) | `L_log + β·L_rank` |

**Metrics**: EU Pearson (clean), EU Spearman (clean), OOD AUROC, AURC.
**Run on**: E2 (CIFAR-10) only.

### 6.4 Ablation C: Ensemble Size K

**Question**: How does teacher quality (more ensemble members) affect distillation quality?

| K | 1 | 3 | 5 (ours) | 7 | 10 |
|---|---|---|---|---|---|

**Metrics**: EU Pearson, OOD AUROC (seen + unseen average), AURC.
**Run on**: E2 (CIFAR-10) only. Note: K=1 is degenerate (no epistemic uncertainty from a single model); use MC Dropout with K=1 model as teacher instead, or simply report it as a sanity check.

### 6.5 Ablation D: Phase 2 Data Mode — Real OOD vs. Synthetic OOD

**Question**: Does the self-contained synthetic OOD strategy (fake_ood) match using real external OOD data?

| Condition | Phase 2 Tier 3 source |
|---|---|
| D1 | fake_ood (mixup + masking, self-contained — ours) |
| D2 | real_ood (SVHN + CIFAR-100 as external OOD) |

**Metrics**: EU Pearson, OOD AUROC on **unseen** OOD datasets only (to avoid data leakage advantage for D2).
**Run on**: E2 (CIFAR-10) only.
**Expected result**: D1 ≈ D2 on unseen OOD, demonstrating the self-contained approach is not at a disadvantage.

### 6.6 Ablation E: Two-Phase Training vs. Joint Training

**Question**: Does the sequential two-phase approach matter? Can you train classification + EU jointly?

| Condition | Description |
|---|---|
| E-joint | Train backbone, classifier, and EU head simultaneously with `L_p1 + γ·L_unc` |
| E-twophase (ours) | Sequential: Phase 1 → freeze backbone → Phase 2 |

**Metrics**: Clean accuracy, EU Pearson, OOD AUROC.
**Run on**: E2 (CIFAR-10) only.

### 6.7 Ablation Summary Table

Present all ablations in a single compact table in the paper:

| Ablation | Variant | EU Pearson ↑ | OOD AUROC (avg) ↑ | AURC ↓ | Acc ↓ gap |
|---|---|---|---|---|---|
| A: Curriculum | Clean only | | | | |
| | + Corrupted | | | | |
| | + Synth OOD (ours) | | | | |
| B: Loss | MSE | | | | |
| | L_log only | | | | |
| | L_rank only | | | | |
| | L_log + L_rank (ours) | | | | |
| C: Ensemble K | K=3 | | | | |
| | K=5 (ours) | | | | |
| | K=10 | | | | |
| D: OOD source | Real OOD | | | | |
| | Synthetic (ours) | | | | |
| E: Training | Joint | | | | |
| | Two-phase (ours) | | | | |

---

## 7. Baselines

You mentioned you will add ensemble, MC Dropout, EDL, and Laplace later. For completeness, here is what each baseline should output for fair comparison:

| Baseline | Inference cost | What to report |
|---|---|---|
| Deep Ensemble (K=5) | K forward passes | Accuracy, ECE, NLL, OOD AUROC (using EU=mutual info), AURC |
| MC Dropout (K=50 passes) | K stochastic passes | Same. Note: needs test-time dropout enabled. |
| Evidential Deep Learning (EDL) | 1 forward pass | Same. Train from scratch with EDL loss (Prior Networks / PostNet). |
| Laplace Approximation | K posterior samples at test time | Same. Use `laplace-torch` library, last-layer Laplace. |
| **Our student (distilled)** | **1 forward pass** | Same. |
| Single ensemble member | 1 forward pass | Entropy and 1-MaxProb only (no EU). |

Report inference throughput for each — this is where our method wins decisively.

---

## 8. Implementation Notes

### 8.1 Segmentation EU Head — Per-Pixel Caching

The per-pixel EU targets are large. For VOC (10,582 train images, 512×512 resolution, 21 classes, K=5 members):
`10582 × 512 × 512 × 21 × 5 × 4 bytes ≈ 290 GB` — do NOT cache raw logits.
Instead, cache only the per-pixel EU scalar (float16):
`10582 × 512 × 512 × 2 bytes ≈ 5.5 GB` — fully manageable.
Compute EU targets on-the-fly during Phase 2 warm-up pass over the training set, save to `.npz` shards.

### 8.2 SST-2 Sentence Mixup

For synthetic OOD in NLP (Tier 3), implement **embedding-space mixup** rather than input-token mixup:
1. Forward two sentences through the frozen student backbone to get their [CLS] embeddings.
2. Linearly interpolate: `h_mix = λ·h_A + (1-λ)·h_B`.
3. Pass `h_mix` through the EU head (and the classifier head, detached, for the concatenated input).
4. EU target: teacher EU on the same mixed embedding (teacher also processes the mixed sentence pair — forward actual token sequences for each λ through teacher and interpolate predictions).

This avoids the out-of-vocabulary problem with token-level mixup.

### 8.3 Random Seeds

Fix all experiment seeds for reproducibility. Use seeds {0, 1, 2, 3, 4} for ensemble members, seed 42 for all student training.

### 8.4 ECE Implementation

Use `torchmetrics.CalibrationError(n_bins=15, norm='l1')` or equivalent. Always report on the held-out test/val set after training. Do not recalibrate with temperature scaling unless explicitly noted.

---

## 9. Priority and Estimated Timeline

| Priority | Experiment / Task | Estimated time (2× 4090) |
|---|---|---|
| 🔴 Critical | Add ECE, NLL, Brier, AURC to E2 (CIFAR-10) | 1 day (eval only) |
| 🔴 Critical | Add ECE, NLL, Brier, AURC to E3 (TinyImageNet) | 1 day (eval only) |
| 🔴 Critical | Throughput benchmark for E2, E3 | 0.5 day |
| 🟠 High | E5: SST-2 full pipeline | 2–3 days |
| 🟠 High | Ablation B (loss), D (OOD source), E (two-phase) on E2 | 2 days |
| 🟠 High | Ablation A (curriculum) on E2 + E5 | 2 days |
| 🟡 Medium | E4: Pascal VOC segmentation | 5–7 days |
| 🟡 Medium | Ablation C (ensemble size K) on E2 | 1 day |
| 🟢 Lower | Retrofit E1 (MNIST) with new metrics | 0.5 day (eval only) |
