# Master Run Log — E1-E3 Evaluation + Ablations + E5 SST-2
Started: $(date)

## Overview
- E1 MNIST: checkpoints_16members (16 teacher members)
- E2 CIFAR-10: checkpoints_16members (16 teacher members)  
- E3 TinyImageNet: checkpoints_rich_fake_ood_12m (16 teacher members), Phase 2 RETRAINED with fixes:
  - log1p_mse instead of plain MSE
  - rank_weight: 0.1 → 1.0
  - p2_epochs: 80 → 150
- Ablation A/B/C/D/E on CIFAR-10
- E5 SST-2 full pipeline

## Status
See individual log files:
- MNIST/train_eval.log
- CIFAR-10/train_eval.log
- TinyImageNet/retrain_fixed.log
- CIFAR-10/ablation_A.log, ablation_B.log, ablation_C.log, ablation_D.log, ablation_E.log
- SST2/train_eval.log

## Key Files
- Final results summary: RESULTS_SUMMARY.md
