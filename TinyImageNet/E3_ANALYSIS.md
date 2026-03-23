# E3 (TinyImageNet) — Problem Analysis & Experiment History

**Date:** 2026-03-22
**Goal:** Single-pass student that matches teacher ensemble's epistemic uncertainty (EU).
**Key problems:** (1) student accuracy below target, (2) SVHN and STL10 OOD AUROC lagging.

---

## Architecture

- **Teacher:** 16-member LoRA ViT-S/16 ensemble, EU = mutual information I[y;θ|x] between members
- **Student:** Single ViT-S/16 + shallow EU regression head (features || softmax probs → scalar EU)
- **Phase 1:** Classification KD from ensemble soft targets
- **Phase 2:** EU head regression — backbone frozen, head trained on cached (feature, EU_target) pairs

### Phase 2 training data (what we actually train on)

We **do not use any real OOD data**. Phase 2 uses:

| Tier | Source | EU target | Approx size |
|------|--------|-----------|-------------|
| Clean | TinyImageNet val (ID) | teacher EU ≈ 0.062 mean | 30k (v2) / 67k (v3) |
| Corrupted | 6 corruption types at medium severity | teacher EU ≈ 0.08–0.13 mean | 6×2500 (v2) / larger (v3) |
| Fake OOD | Mixup / patch-shuffle / cutpaste / masked / noise (v2); char/word perturbation style (v3) | teacher EU ≈ 0.13 mean (v2) | varies |

**Critical constraint:** The teacher assigns real OOD like SVHN a mean EU of **0.34**, but the highest fake-OOD EU targets in training are ~0.13. The head is never shown what "truly OOD" looks like.

---

## Run History

### v2 (baseline before improvement plan)

**Phase 1 config:** 2 unfreeze blocks, EMA=0.999, warmup 5 epochs, 50 epochs
**Phase 2 config:** log1p-MSE + pairwise ranking loss, symmetric asymmetric weight, 150 epochs
**EU head:** 2-layer: Linear(584+200→256) → ReLU → Linear(256→1) → Softplus

| Metric | Value |
|--------|-------|
| Phase 1 val acc (EMA) | **87.50%** |
| Phase 2 Spearman | **0.8407** |
| Phase 2 Pearson | 0.7880 |
| Student EU mean (clean) | 0.0841 (teacher: 0.0623) — **35% over-predicted** |
| Student EU mean (SVHN) | 0.2132 (teacher: 0.3435) — **38% under-predicted** |

**OOD AUROC (v2):**

| Dataset | Teacher EU | Student EU | Student TU |
|---------|-----------|-----------|------------|
| SVHN (seen) | 0.9795 | **0.8974** | 0.8456 |
| CIFAR-10 (unseen) | 0.8719 | 0.8035 | 0.8353 |
| CIFAR-100 (seen) | 0.8673 | 0.8262 | 0.8400 |
| STL10 (unseen) | 0.8115 | 0.7438 | 0.8308 |
| DTD | 0.9092 | 0.9074 | 0.9291 |
| FashionMNIST | 0.9451 | 0.8874 | 0.9097 |
| MNIST | 0.8951 | 0.8796 | 0.8027 |

AURC: 0.026 (teacher) vs — (not recorded)
Throughput: 25.94x

**v2 diagnosis:** EU dynamic range is compressed. ID EU is over-predicted (head confuses hard ID samples for OOD), OOD EU is under-predicted (fake OOD ceiling ~0.13 constrains the head). Teacher ID-to-SVHN ratio = 5.5x; student = 2.5x.

---

### v3 (improvement plan, 2026-03-22)

**Goal:** Fix dynamic range compression via tier-aware losses and stronger Phase 1.

**Phase 1 changes:**
- Unfrozen blocks: 2 → **4** (last 4 ViT blocks)
- Epochs: 50 → **100**
- LR: 1e-4 → **2e-4** (base), LLRD factor 0.75
- Backbone LR floor: 0.01× (deep blocks stay at 2e-6)
- Same MixUp/CutMix/EMA/RandAugment augmentation pipeline

**Phase 2 changes:**
- EU head: 2-layer → **3-layer** with learnable scale parameter
  - Linear(584+200→512) → LeakyReLU(0.1) → Dropout(0.15) → Linear(512→128) → LeakyReLU(0.1) → `scale * Linear(128→1)`
- Loss: plain log1p-MSE → **combined**:
  - `tier_asymmetric_log1p_mse` (tier 0 clean: over-prediction penalized 3×; tier 2 OOD: under-prediction penalized 2×)
  - `pairwise_ranking_loss`
  - `id_suppression_loss` (suppress over-prediction on clean ID), weight=**2.0**
  - `tier_margin_loss` (push ordering: clean < corrupted < OOD), weight=0.5
- Tier labels: clean=0, corrupted=1, fake OOD=2
- Phase 2 total samples: ~101k (vs ~60k in v2)

**Results:**

| Metric | v2 | v3 | Change |
|--------|----|----|--------|
| Phase 1 val acc | 87.50% | **85.67%** | ❌ −1.83% |
| Phase 2 Spearman | 0.8407 | **0.7986** | ❌ −0.042 |
| Phase 2 Pearson | 0.7880 | **0.6518** | ❌ −0.136 |
| Student EU mean (clean) | 0.0841 | **0.0738** | ✅ closer to teacher 0.062 |
| Student EU mean (SVHN) | 0.2132 | **0.1157** | ❌ worse, further from teacher 0.3435 |
| SVHN AUROC | **0.8974** | 0.8439 | ❌ −0.054 (regressed) |
| CIFAR-10 AUROC | 0.8035 | **0.8722** | ✅ +0.069 |
| CIFAR-100 AUROC | 0.8262 | **0.8556** | ✅ +0.029 |
| STL10 AUROC | 0.7438 | **0.7609** | ✅ slight gain |
| DTD AUROC | 0.9074 | **0.9386** | ✅ beats teacher |
| FashionMNIST AUROC | 0.8874 | **0.9281** | ✅ improved |
| MNIST AUROC | 0.8796 | **0.8885** | ✅ improved |
| AURC (student EU) | — | 0.0346 | ❌ worse than teacher 0.026 |
| Distribution shift (4d) | — | **0.7182** | ✅ beats teacher 0.6326 |

---

## Problem 1: Student Accuracy (85.67% vs target ≥88%)

### What happened
Unfreezing 4 blocks with `base_lr=2e-4` was too aggressive. The training curve shows Phase 1 peaking at val_acc=0.8567 at epoch ~70 and plateauing — it never reaches 87.5%.

LLRD partially mitigates this (deep blocks at 2e-6), but the top 4 blocks at 2e-4→1.5e-4 are still being updated with higher learning rates than are ideal for a pretrained ViT.

### Comparison
- v2 (2 blocks unfrozen, lr=1e-4, 50 epochs): **87.50%**
- v3 (4 blocks unfrozen, lr=2e-4, 100 epochs): **85.67%**

Counterintuitively, unfreezing more blocks hurt accuracy. The pretrained ViT-S/16 has good general representations in layers 8–11; aggressively fine-tuning them on TinyImageNet (64×64, 200 classes) can drift them away from the ImageNet-pretrained features.

### Root cause
4-block unfreeze at `lr=2e-4` perturbs mid-level representations enough to reduce accuracy without enough training data to recover. v2's 2-block unfreeze was closer to optimal.

### Fix
Try `--p1_unfreeze_blocks 3` with `base_lr=1e-4` (or keep 2e-4 but reduce LLRD to 0.65 to lower block 3/4 LR more aggressively).

---

## Problem 2: SVHN and STL10 OOD AUROC

### Root cause A: Fake OOD EU ceiling

The EU head is trained exclusively on data with EU targets in the range [0.001, ~0.20]:

| Data split | Student EU mean | Teacher EU mean |
|------------|----------------|----------------|
| Clean ID | 0.0738 | 0.0623 |
| Corrupted | 0.09–0.11 | 0.08–0.12 |
| Fake OOD | ~0.13 | ~0.13 |
| **Real SVHN** | **0.1157** | **0.3435** |
| **Real STL10** | **0.1080** | **0.1508** |

The fake OODs (masked, shuffled, perturbed patches) sit at ~0.13 teacher EU — far below real SVHN (0.34). The head learns to cap its output around 0.15, because no training sample ever had a target higher than ~0.20. When it encounters SVHN at test time, it predicts 0.12, which barely separates from clean ID at 0.07.

This is why SVHN Spearman is **-0.026** — there is essentially zero correlation between what the teacher says about SVHN EU and what the student predicts. The student is extrapolating outside its training distribution.

### Root cause B: ID suppression loss collateral damage (v3-specific)

The `id_suppression_loss` with `weight=2.0` penalizes over-prediction on clean ID. This successfully reduced ID EU from 0.084 → 0.074. However, it had a side effect: the head learned to suppress its output globally (including for SVHN), since the architecture cannot easily distinguish "clean ID-like features" from "SVHN-like features" at the EU head level (the head only sees frozen backbone features, not raw pixels).

Evidence: v2 (no suppression loss) had SVHN EU mean = 0.213. v3 (suppression weight=2.0) has SVHN EU mean = 0.116. The suppression loss drove SVHN EU down by 45%.

**v2 vs v3 ID-to-OOD ratio:**

| Version | ID EU | SVHN EU | Ratio |
|---------|-------|---------|-------|
| v2 | 0.084 | 0.213 | 2.5x |
| v3 | 0.074 | 0.116 | **1.6x** (worse!) |
| Teacher | 0.062 | 0.344 | **5.5x** (target) |

### Why STL10 lags (but less severely)

STL10 (natural animal/vehicle images, similar domain to TinyImageNet) has a smaller teacher EU gap: teacher assigns 0.151 mean, student predicts 0.108. This is a 28% underestimate vs SVHN's 66% underestimate. STL10 is at the edge of the fake OOD training range, so the head partially generalizes. SVHN (street numbers, low-texture synthetic-looking images) is much further from TinyImageNet in feature space, making the gap larger.

---

## Summary: What Each Approach Tried and What Broke

| Approach | What it fixed | What it broke | Net |
|----------|--------------|---------------|-----|
| **v2 baseline** | Good Spearman (0.84), SVHN 0.90 | EU mean compressed (ID over, OOD under) | Acceptable baseline |
| **v3: 4 unfreeze blocks** | Aimed to improve accuracy | Accuracy dropped 1.8pp | ❌ wrong direction |
| **v3: 3-layer EU head** | More capacity for EU modeling | Harder to train stably | Neutral |
| **v3: tier-aware loss** | Better ordering clean<corrupt<OOD | Fake OOD ceiling unchanged (0.13) | Small improvement on non-SVHN |
| **v3: id_suppression (w=2.0)** | Reduced ID over-prediction | Collaterally suppressed SVHN/STL10 EU | ❌ SVHN AUROC −0.054 |
| **v3: tier_margin loss** | Push EU ordering | Limited effect without real OOD targets | Marginal |

---

## Recommended Next Steps

### Fix 1 (quick, ~2h): Reduce ID suppression weight
Re-run Phase 2 only from the Phase 1 v3 checkpoint:
```
--phase2_only --suppress_weight 1.0 --margin_weight 0.2
```
Expected: SVHN recovers toward 0.87–0.89 (close to v2), while ID EU stays lower than v2 (0.074 → 0.076–0.079). Low risk.

### Fix 2 (better, ~4h): Retrain Phase 1 with 3 blocks
```
--p1_unfreeze_blocks 3 --p1_lr 1e-4 --p1_epochs 80
```
Expected: val acc ~87.0–87.5%. Then re-run Phase 2 with Fix 1.

### Fix 3 (architectural): Unfreeze top 1–2 blocks in Phase 2
Allow the backbone to adapt slightly during EU head training. This would help SVHN since the features themselves would shift to be more separable.

Downside: risks classification accuracy degradation; needs careful LR (1e-5 for unfrozen blocks).

---

## Key Takeaway

The fundamental limitation is that **fake OOD EU targets (~0.13 mean) are far below real-world OOD EU (~0.34 for SVHN)**. No amount of loss engineering can fix this without either:
1. Raising the fake OOD targets (hard to do without real OOD reference), or
2. Including real OOD data in Phase 2 training.

The tier-aware losses and suppression are fighting over a range of [0.07, 0.13] when the real signal lives at [0.07, 0.35]. This is a data problem, not a model problem.
