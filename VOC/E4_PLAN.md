# E4: VOC 2012 Segmentation — Completion Plan

## Current Status (as of 2026-03-23)

| Step | Status | Details |
|------|--------|---------|
| Ensemble training | ✅ Done | 5/5 members complete |
| Cache teacher targets | ✅ Done | `teacher_targets.npz` (256MB) |
| Phase 1 distillation | ⚠️ 57% | 22875/40000 steps, best mIoU=0.8056 saved |
| Phase 2 EU head | ❌ Not started | |
| Evaluation | ❌ Not started | |

### Ensemble Members

| Member | Seed | Augmentation | Label Smooth | Best mIoU |
|--------|------|-------------|--------------|-----------|
| 0 | 42 | scale | 0.0013 | 0.7999 |
| 1 | 43 | colorjitter | 0.0348 | 0.8043 |
| 2 | 44 | colorjitter | 0.0271 | 0.8042 |
| 3 | 45 | colorjitter | 0.0244 | 0.8081 |
| 4 | 46 | rotation | 0.0200 | 0.8076 |
| **Mean** | | | | **0.8048** |

### Phase 1 Training Curve (before termination)

```
Epoch  50 (step  9150): mIoU=0.7949
Epoch  85 (step 15555): mIoU=0.8042
Epoch 110 (step 20130): mIoU=0.8056  ← best (saved to student_phase1.pt)
Epoch 125 (step 22875): mIoU=0.7995  ← last logged, then killed (SIGTERM)
```

Student mIoU (0.8056) already exceeds teacher ensemble mean (0.8048). Curve had plateaued.

---

## Plan: Complete E4

### Step 1: Phase 2 EU Head Training

Use existing `student_phase1.pt` as-is (no need to finish Phase 1 — mIoU already at teacher level).

```bash
cd /home/maw6/maw6/unc_regression/VOC

python distill.py \
  --save_dir ./checkpoints \
  --phase2_only \
  --p2_iterations 20000 \
  --p2_lr 0.001 \
  --rank_weight 1.0 \
  --curriculum A3 \
  --gpu 0
```

**EU head architecture**: Conv(512→128, 1×1) → BN → ReLU → Conv(128→1, 1×1) → Softplus

**Phase 2 curriculum (A3)**:
- Tier 1: Clean train images
- Tier 2: Corrupted train (gaussian_noise, gaussian_blur, low_contrast)
- Tier 3: Synthetic OOD (mixup pairs + block-masked regions)

**Estimated time**: ~2-3 hours on single GPU.

### Step 2: Evaluation

```bash
python evaluate_student.py \
  --save_dir ./checkpoints \
  --data_dir ../data \
  --gpu 0
```

**Metrics computed**:
1. Segmentation: mIoU, pixel accuracy
2. Calibration: ECE-15, NLL
3. EU correlation: Pearson & Spearman vs teacher
4. Spatial analysis: boundary vs interior EU ratio
5. Selective segmentation: coverage-mIoU curve, AURC
6. OOD detection: image-level AUROC (VOC vs DTD far-OOD, VOC vs COCO near-OOD)
7. Throughput: images/sec speedup vs ensemble

**Estimated time**: ~30-60 minutes.

### Step 3: One-liner (both steps combined)

```bash
cd /home/maw6/maw6/unc_regression/VOC && \
python distill.py --save_dir ./checkpoints --phase2_only \
  --p2_iterations 20000 --p2_lr 0.001 --rank_weight 1.0 \
  --curriculum A3 --gpu 0 \
  > checkpoints/p2.log 2>&1 && \
python evaluate_student.py --save_dir ./checkpoints \
  --data_dir ../data --gpu 0 \
  > checkpoints/eval.log 2>&1
```

---

## Optional: Port E3 Improvements

The VOC `distill.py` currently uses `log1p_MSE + ranking` (same as CIFAR-10). The TinyImageNet E3 remediation found these improvements:

| Feature | E3 Impact | VOC Status |
|---------|-----------|------------|
| Asymmetric loss (asym=3.0) | +0.015 SVHN AUROC | Not ported |
| Geo2 density features | +0.061 STL10 AUROC | Not ported |
| suppress_weight=0 | Critical for OOD | N/A (VOC has no suppress) |
| eu_sample_alpha=0 | Critical for OOD | N/A (VOC has no alpha) |

If Phase 2 results are weak (especially OOD detection), consider porting asymmetric loss and density features from `TinyImageNet/distill.py` and `TinyImageNet/models.py` to the VOC pipeline.

---

## Notes

- The original run was killed by SIGTERM during Phase 1 (likely OOM or manual kill during model re-download)
- VOC uses K=5 members (not K=16 like CIFAR-10/TinyImageNet) — all 5 are trained
- Phase 2 loads all train images into memory (~10k images at 512×512). Monitor GPU memory.
- OOD datasets needed for evaluation: DTD (already in `../data/dtd/`), COCO (may need download)
