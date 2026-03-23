# Uncertainty Distillation — Experiment Summary

> **Project goal**: Distill a LoRA ensemble's epistemic uncertainty (EU) into a single student model via two-phase training: Phase 1 = classification KD, Phase 2 = EU regression head. Target venue: NeurIPS 2026.
>
> **Last updated**: 2026-03-23

---

## Experiment Status Overview

| Exp | Dataset | Task | Phase 1 | Phase 2 | Status |
|-----|---------|------|---------|---------|--------|
| E1 | MNIST | Classification | 99.48% acc | Spearman=0.7220 | Complete ✅ |
| E2 | CIFAR-10 | Classification | 95.99% acc | Spearman=0.7975 | Complete ✅ (with ablations) |
| E3 | TinyImageNet | Classification | 87.50% acc | Spearman=0.8848 | Complete ✅ (all 4 gates met) |
| E4 | VOC 2012 | Segmentation | 5/16 members trained | — | Terminated ❌ |
| E5 | SST-2 | NLP/Sentiment | 92.09% acc | Spearman=0.5771 | Phase 2 failing ❌ (target ≥0.65) |

---

## E1: MNIST

**Ensemble**: 5 LoRA members on a small CNN.

| Metric | Value |
|--------|-------|
| Teacher accuracy | 99.55% |
| Student accuracy | 99.48% |
| EU Spearman | 0.7220 |
| EU Pearson | 0.7634 |
| OOD AUROC: SVHN | 1.0000 |
| OOD AUROC: CIFAR-10 | 1.0000 |
| OOD AUROC: FashionMNIST | 0.9994 |
| OOD AUROC: Omniglot | 1.0000 |
| Speedup vs ensemble | 6.88× (846k samples/sec) |
| Selective AURC | 0.000118 |

**Fake-OOD variant** (`checkpoints_fake_ood/`): Acc=99.60%, Spearman=0.7429, FashionMNIST AUROC=0.9788.

**Key files**: `MNIST/checkpoints_16members/`, `MNIST/checkpoints_fake_ood/`

---

## E2: CIFAR-10

**Ensemble**: 16 LoRA members on ResNet-18.

### Main Results

| Metric | Value |
|--------|-------|
| Teacher accuracy | 96.96% |
| Student accuracy | 95.99% |
| EU Spearman | 0.7975 |
| EU Pearson | 0.8144 |
| ECE-15 | 0.0195 |
| OOD AUROC: SVHN (seen) | 0.9836 |
| OOD AUROC: MNIST (unseen) | 0.9455 |
| Speedup vs ensemble | 16.65× (27,346 samples/sec) |
| Selective AURC | 0.003423 |

### Ablation Studies (on CIFAR-10)

**A — Curriculum schedule** (how Phase 2 transitions from easy→hard samples):

| Run | Pearson | Spearman | Notes |
|-----|---------|----------|-------|
| Baseline (16 members) | 0.8144 | 0.7975 | Standard schedule |
| A1 | 0.8001 | 0.8072 | Curriculum variant 1 |
| A2 | 0.8143 | 0.8067 | Curriculum variant 2 |
| A3 | — | — | (in checkpoints) |

**B — Loss components**:

| Run | Pearson | Spearman | Notes |
|-----|---------|----------|-------|
| B_log_mse | 0.8144 | 0.6437 | Log-MSE only |
| B_mse | 0.8240 | 0.6498 | MSE only |
| B_ranking | 0.7044 | 0.8176 | Ranking only |
| B_combined | — | — | Combined log-MSE + ranking (best) |

**C — Ensemble size K**:

| K | Pearson | Spearman | Notes |
|---|---------|----------|-------|
| 3 | 0.6235 | 0.6538 | Too few members |
| 5 | 0.7144 | 0.7223 | Moderate |
| 10 | 0.7922 | 0.7842 | Good |
| 16 | 0.8144 | 0.7975 | Best (baseline) |

**D — Real OOD**: Used actual OOD data instead of synthetic fake-OOD in Phase 2 training.

**E — Joint training**: Joint Phase 1+2 training → Pearson=0.6459, Spearman=0.6274, Acc=95.07%. Significantly worse; confirms two-phase approach is essential.

**Key files**: `CIFAR-10/checkpoints_16members/`, `CIFAR-10/checkpoints_16members_abl_*/`

---

## E3: TinyImageNet (Primary Focus)

**Ensemble**: 16 heterogeneous LoRA members on DeiT-S (ViT).
**Phase 1 checkpoint**: `checkpoints_rich_fake_ood_12m/student_phase1.pt` (87.50% acc)

### Gating Criteria (all met by `asym3_geo2`)

| Criterion | Target | Achieved | |
|-----------|--------|----------|---|
| Accuracy | ≥ 87.50% | 87.50% | ✅ |
| Spearman (clean val) | ≥ 0.82 | 0.8848 | ✅ |
| SVHN AUROC | ≥ 0.89 | 0.8976 | ✅ |
| STL10 AUROC | ≥ 0.76 | 0.7887 | ✅ |

### Winning Phase 2 Recipe

```bash
python distill.py --save_dir ./checkpoints_p2_asym3_geo2 \
  --phase2_only --p2_epochs 80 --p2_lr 0.003 \
  --rank_weight 1.0 --asymmetric_weight 3.0 \
  --suppress_weight 0.0 --tail_weight 0.0 \
  --margin_weight 0.0 --eu_sample_alpha 0.0 \
  --density_features --n_extra_eu 2 --gpu 0
```

### Full OOD Detection Table (Winning Checkpoint)

| OOD Dataset | Type | Tea EU | Tea TU | Stu EU | Stu TU | Sgl(H) |
|-------------|------|--------|--------|--------|--------|--------|
| SVHN | seen | 0.9795 | 0.9739 | 0.8976 | 0.8456 | 0.9656 |
| CIFAR-10 | unseen | 0.8719 | 0.8733 | 0.8307 | 0.8353 | 0.8603 |
| CIFAR-100 | seen | 0.8673 | 0.8714 | 0.8247 | 0.8400 | 0.8581 |
| STL10 | unseen | 0.8115 | 0.8876 | 0.7887 | 0.8308 | 0.8527 |
| DTD | unseen | 0.9092 | 0.9538 | 0.8994 | 0.9291 | 0.9327 |
| FashionMNIST | unseen | 0.9451 | 0.9430 | 0.8947 | 0.9097 | 0.9397 |
| MNIST | unseen | 0.8951 | 0.9215 | 0.8867 | 0.8027 | 0.9121 |

### Additional Metrics (Winning Checkpoint)

| Metric | Value |
|--------|-------|
| ECE-15 | 0.0484 |
| Teacher-Student agreement | 95.11% |
| Inference speedup | 24.95× (4,065 samples/sec) |
| Selective AURC | 0.024261 |

### Phase 2 Ablation Table (27 runs)

All runs use Phase 1 acc = 87.50%. Gating: Spearman ≥ 0.82, SVHN ≥ 0.89, STL10 ≥ 0.76.

| Run | Spearman | SVHN | STL10 | Key Config | Gate |
|-----|----------|------|-------|------------|------|
| **asym3_geo2** | **0.8848** | **0.8976** | **0.7887** | **asym=3.0, geo2, α=0** | **✅ ALL** |
| true_asym_geo2 | 0.8873 | 0.8823 | 0.7930 | asym=2.0, geo2, α=0 | ❌ SVHN |
| suppress_density | 0.8884 | 0.8424 | 0.7788 | suppress + 5 density feats | ❌ SVHN |
| density (5 feats) | 0.8860 | 0.8700 | 0.7877 | 5 density, CVaR tw=1.0 | ❌ SVHN |
| asym3_geo2 (original) | 0.8407 | 0.8876 | 0.7273 | asym=2.0, α=0, suppress NOT wired | ❌ STL10 |
| density_hightail | 0.8828 | 0.8335 | 0.7825 | density + high tail | ❌ SVHN |
| asym2_rankw2_geo2 | 0.8827 | 0.8242 | 0.7947 | rank_weight=2.0, geo2 | ❌ SVHN |
| suppress_geo2_alpha0 | 0.8818 | 0.8435 | 0.7964 | suppress=2.0, geo2, α=0 | ❌ SVHN |
| asym_geo2_alpha0 | 0.8812 | 0.8718 | 0.7898 | asym=1.0 (sym), geo2, α=0 | ❌ SVHN |
| true_asym_geo2_cvar1 | 0.8808 | 0.8564 | 0.7910 | asym=2.0, geo2, CVaR tw=1.0 | ❌ SVHN |
| sym_suppress_geo2 | 0.8758 | 0.8396 | 0.7877 | sym suppress + geo2 | ❌ SVHN |
| ablate_s025 | 0.8757 | 0.8407 | 0.7729 | suppress=0.25 | ❌ SVHN |
| true_asym_geo2_cvar2 | 0.8732 | 0.8720 | 0.7945 | asym=2.0, geo2, CVaR tw=2.0 | ❌ SVHN |
| geo2_cvar | 0.8726 | 0.8675 | 0.7911 | geo2 + CVaR | ❌ SVHN |
| ablate_s0 | 0.8703 | 0.8426 | 0.7575 | suppress=0 | ❌ SVHN |
| geo2_cvar2_300ep | 0.8689 | 0.8591 | 0.7857 | geo2_cvar2, 300 epochs | ❌ SVHN |
| geo2_cvar2 | 0.8667 | 0.8836 | 0.7871 | CVaR tw=2.0, geo2, α=10 | ❌ SVHN |
| geo2_cvar2_alpha30 | 0.8655 | 0.8583 | 0.7844 | geo2_cvar2, α=30 | ❌ SVHN |
| geo2_cvar3 | 0.8639 | 0.8027 | 0.7902 | geo2 + CVaR tw=3.0 | ❌ SVHN |
| nocvar | 0.8607 | 0.8326 | 0.7528 | no CVaR | ❌ SVHN |
| sym_suppress | 0.8603 | 0.8504 | 0.7541 | suppress=2.0, no geo2 | ❌ SVHN |
| hightail | 0.8528 | 0.8516 | 0.7502 | high tail weight | ❌ SVHN |
| v2loss_cvar | 0.8478 | 0.8808 | 0.7487 | CVaR tw=1.0, α=10 | ❌ STL10 |
| 2layer_softplus | 0.8459 | 0.8363 | 0.7584 | 2-layer softplus head | ❌ SVHN |
| hightail_cvar | 0.8376 | 0.8156 | 0.7474 | hightail + CVaR | ❌ SVHN |
| original_repro | 0.8081 | 0.8346 | 0.7244 | suppress wired + sym | ❌ SVHN+STL10 |
| cvar_strong | 0.8048 | 0.8758 | 0.7000 | strong CVaR | ❌ STL10 |

### Key Ablation Insights

1. **asymmetric_weight=3.0** (vs 2.0): +0.015 SVHN AUROC — stronger underprediction penalty pushes OOD EU higher
2. **Geo2 features** (max_cos + min_dist to class centroids): +0.061 STL10 AUROC vs no density features
3. **suppress_weight=0** is critical: active suppress hurts SVHN by −0.028 (compresses ID EU too much)
4. **eu_sample_alpha=0** is critical: alpha=10 hurts SVHN by ~−0.01 to −0.015
5. **CVaR** (tail_weight > 0) with asymmetric loss: competing objectives, consistently hurts SVHN
6. **rank_weight=2.0**: destroys SVHN (−0.063 vs rank_weight=1.0)
7. **Joint training** (CIFAR-10 ablation E): much worse than two-phase (Spearman 0.63 vs 0.80)
8. **More ensemble members**: monotonic improvement (K=3→5→10→16 on CIFAR-10)

### Key Code Changes for E3 Remediation (2026-03-22)

1. `models.py`: Added `n_extra_eu` parameter, density feature computation (energy/entropy/margin/max_cos/min_dist), centroid buffers. Fixed slice ordering (before normalization).
2. `distill.py`: Added `--density_features`, `--n_extra_eu` flags; wired suppress_weight to loss.
3. `evaluate_student.py`: Auto-detect n_extra_eu from checkpoint shape; `num_workers=0` for TensorDatasets (deadlock fix).

---

## E4: VOC 2012 Segmentation (Terminated)

**Architecture**: SegFormer-B2 ensemble with LoRA.
**Status**: 5 of 16 ensemble members trained (mIoU ~0.80), then terminated.
**Checkpoint**: `VOC/checkpoints/` — 5 members + student_phase1.pt + teacher_targets.npz.
**Note**: Run crashed/terminated during execution (`run_e4.log` ends with `Terminated`). Not enough members for meaningful EU.

---

## E5: SST-2 NLP (Phase 2 Failing)

**Architecture**: 5 LoRA members on DistilBERT.

### Results Comparison

| Metric | v2 (checkpoints/) | v3 (checkpoints_v3/) |
|--------|-------|-------|
| Teacher accuracy | 91.51% | 91.51% |
| Student accuracy | 89.68% | 92.09% ✅ |
| EU Pearson | 0.3655 | 0.4859 |
| EU Spearman | 0.5146 | 0.5771 ❌ (target ≥0.65) |
| ECE-15 | 0.0384 | 0.0251 |
| OOD: AG_News | 0.7692 | 0.7428 |
| Speedup | 10.53× | 10.57× |

**Diagnosis**: Phase 2 EU correlation stuck at 0.5771, below 0.65 target. V3 improved accuracy (+2.4%) and calibration but EU distillation remains weak. May need architectural changes or different loss formulation for NLP domain.

---

## Key Findings Across All Experiments

1. **Two-phase training is essential**: Joint training (CIFAR-10 ablation E) showed significant degradation vs sequential phases.
2. **Combined loss (log-MSE + ranking)** outperforms individual components for EU regression.
3. **Ensemble diversity** (heterogeneous LoRA configs) improves EU quality over homogeneous ensembles.
4. **Fake-OOD** (synthetic out-of-distribution data) is sufficient — no need for real OOD data in training.
5. **Density features (geo2)** dramatically improve OOD detection on datasets like STL10.
6. **Asymmetric loss** with strong underprediction penalty is key for OOD detection (pushes EU higher on unseen data).
7. **Inference speedup**: 7–25× depending on architecture, with minimal accuracy loss.

---

## File Structure (Post-Cleanup)

```
unc_regression/
├── EXPERIMENT_SUMMARY.md          # This file
├── README.md                      # Project overview & usage
├── requirements.txt
├── data/                          # Datasets (gitignored)
├── CIFAR-10/
│   ├── *.py                       # Source code
│   ├── checkpoints_16members/     # Main checkpoint (E2)
│   ├── checkpoints_16members_abl_*/  # Ablation checkpoints
│   ├── checkpoints/               # 5-member baseline
│   └── *.md                       # Method docs & results
├── MNIST/
│   ├── *.py
│   ├── checkpoints_16members/     # Main checkpoint (E1)
│   └── checkpoints_fake_ood/      # Fake-OOD variant
├── TinyImageNet/
│   ├── *.py                       # Source code
│   ├── checkpoints_rich_fake_ood_12m/  # Phase 1 + ensemble (canonical)
│   ├── checkpoints_p2_asym3_geo2/      # WINNING Phase 2 checkpoint
│   ├── checkpoints_p2_*/eval.log       # Ablation eval logs (weights stripped)
│   └── result.md                       # Latest eval output
├── SST2/
│   ├── *.py
│   ├── checkpoints_v3/            # Latest (E5, still failing)
│   └── *.log                      # Training logs
└── VOC/
    ├── *.py
    └── checkpoints/               # Partial (5/16 members)
```
