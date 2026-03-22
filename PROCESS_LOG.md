# Experiment Process Log
> Started: 2026-03-21
> Goal: Train students for E1-E3 with 16-member teachers, evaluate, run CIFAR-10 ablations

## Checkpoint Sources
- MNIST: `MNIST/checkpoints_16members/` (16 members, no student yet)
- CIFAR-10: `CIFAR-10/checkpoints_16members/` (16 members, no student yet)
- TinyImageNet: `TinyImageNet/checkpoints_rich_fake_ood_12m/` (16 members + student already trained)

## Key Constraints
- Only fake OOD (mixup, masking, shifted ID) — NO true OOD
- Student EU should perform similarly to teacher EU
- All work on cuda:0, conda activate maw6

## Progress

### Step 1: Understanding existing code [DONE]
- Read all E1-E3 code for cache/distill/evaluate
- Identified TinyImageNet Phase 2 issues:
  - Plain MSE instead of log1p_mse
  - rank_weight=0.1 (too low, CIFAR-10 uses 1.0)
  - Only 80 epochs
  - Student EU AUROC on SVHN: 0.7990 vs Teacher: 0.9795

### Step 2: Fix TinyImageNet Phase 2 [DONE]
- Changed `F.mse_loss` → `log1p_mse_loss` in TinyImageNet/distill.py
- Changed rank_weight default: 0.1 → 1.0
- Changed p2_epochs default: 80 → 150

### Step 3: Run E1 MNIST pipeline [DONE]
- Using checkpoints_16members (16 ensemble members)
- cache_ensemble_targets.py --p2_data_mode fake_ood [DONE]
- distill.py Phase 1: 99.52% test acc, Phase 2: Spearman 0.8596, Pearson 0.7225 [DONE]
- evaluate_student.py [DONE]
- Key results:
  - Teacher acc: 99.54%, Student acc: 99.52%
  - ECE: Teacher 0.0377, Student 0.0355
  - Clean EU Pearson: 0.8162, Spearman: 0.7493
  - OOD AUROC (Student EU): FashionMNIST 0.9987, EMNIST 0.9650, CIFAR-10 1.0000, SVHN 1.0000
  - Throughput: Student 523k samples/sec, Ensemble 36k (14.5x speedup)

### Step 4: Run E2 CIFAR-10 pipeline [DONE]
- Using checkpoints_16members (16 ensemble members)
- cache_ensemble_targets.py --p2_data_mode fake_ood [DONE]
- Phase 1: 95.85% best acc (200 epochs). Crash on load -> fixed weights_only=False
- Phase 2 only: Spearman 0.7988, Pearson 0.8148
- Key results:
  - Teacher acc: 96.96%, Student acc: 95.99%
  - ECE: Teacher 0.0294, Student 0.0195
  - Clean EU Pearson: 0.8148, Spearman: 0.7988
  - OOD AUROC (Student EU): SVHN 0.9836, CIFAR-100 0.9202, MNIST 0.9434, DTD 0.9479
  - Student EU BEATS teacher on SVHN (0.9836 vs 0.9806)!
  - Throughput: Student 25k, Ensemble 1.4k (18.5x speedup)

### Step 5: TinyImageNet Phase 2 Re-training [DONE]
- Used checkpoints_rich_fake_ood_12m
- Changed to log1p_mse, rank_weight=1.0, 150 epochs
- Phase 2 best Spearman: 0.8301 (vs 0.8339 before)
- OOD results: SVHN 0.7829 (vs 0.7990 before, slightly worse)
  DTD 0.9156 (vs 0.8931 before, improved and beats teacher 0.9092)
- Structural limitation: pre-extracted features approach limits OOD detection
- Student TU (entropy) is often better than student EU for OOD on TinyImageNet

### Step 6: E1-E3 Summary [DONE]
- Created E1_E3_RESULTS_SUMMARY.md with comprehensive tables

### Step 7: CIFAR-10 Ablation Studies [DONE]
- Created run_ablations.sh script
- All 12 ablation variants trained and evaluated:
  - A (Curriculum): A1 clean-only, A2 clean+corrupted, A3 full
  - B (Loss): MSE, log_MSE, ranking-only, combined
  - C (Ensemble K): K=3, K=5, K=10
  - D (OOD source): real OOD vs fake OOD
  - E (Training): joint vs two-phase
- Key findings:
  - Ranking loss critical for Spearman (0.82 vs 0.65 without)
  - Two-phase >> joint training (0.80 vs 0.62 Spearman)
  - Fake OOD closely matches real OOD (validates approach)
  - Even K=3 achieves 0.99 SVHN AUROC (OOD detection robust to K)
  - EU correlation improves monotonically with K
- Full results in E1_E3_RESULTS_SUMMARY.md

### Step 8: E5 SST-2 Experiment [DONE]
- Trained 5-member BERT-base LoRA ensemble (K=5)
  - Each member: rank=8, alpha=16, 5 epochs on SST-2 train
  - Dev accuracies saved in member_*.pt checkpoints
- Cached teacher targets for: clean train/dev, char-perturbed, word-perturbed, token-masked (3 rates)
- Phase 1 distillation: DistilBERT student, 10 epochs, best dev acc = 87.61%
- Phase 2 EU head: 50 epochs, best Spearman = 0.3833, Pearson = 0.3135
  - Low correlation expected: binary classification → very small EU values (mean ~0.01)
  - 872-sample dev set makes correlation estimation noisy
- Key results:
  - Teacher acc: 91.51%, Student acc: 87.61% (gap -3.90%, larger because DistilBERT ≠ BERT)
  - ECE: Teacher 0.0179, Student 0.0473
  - OOD AUROC (Student EU): IMDB 0.4689, Yelp 0.6430, Amazon 0.5664, AG_News 0.7785, 20NG 0.6709
  - Student EU beats teacher EU on all OOD datasets except IMDB (near-random for both)
  - Student entropy (0.83 on AG_News) is better than student EU (0.78) for OOD detection in NLP
  - AURC: Student EU (0.051) beats Teacher EU (0.065) for selective prediction
  - Throughput: Student 3,176 sent/sec, Ensemble 299 sent/sec (10.6x speedup)
- Results added to E1_E3_RESULTS_SUMMARY.md

