Loaded student (spearman=0.7596, pearson=0.6965)

============================================================
  1. Accuracy
============================================================
  Teacher (ensemble):  88.04%
  Student (distilled): 82.82%

============================================================
  1b. Calibration Metrics (ECE-15, NLL, Brier)
============================================================
  Model                              ECE-15        NLL      Brier
  ------------------------------------------------------------
  Teacher (ensemble)                 0.0587     0.4764     0.1769
  Student (distilled)                0.0286     0.6884     0.2423

============================================================
  2. Correctness Agreement
============================================================
  Both correct:            8087  (80.9%)
  Both wrong:              1001  (10.0%)
  Student correct only:     195  (1.9%)
  Teacher correct only:     717  (7.2%)
  Total agreement:       90.88%

============================================================
  3. EU Correlation (student EU vs teacher EU)
============================================================
  Dataset                           Pearson   Spearman   Stu mean   Tea mean
  ----------------------------------------------------------------------
  Clean TinyImageNet val             0.6998     0.7628     0.0365     0.0354
  Corrupted: gaussian_noise          0.7760     0.8004     0.0546     0.0537
  Corrupted: gaussian_blur           0.8006     0.8241     0.0398     0.0425
  Corrupted: low_contrast            0.7795     0.7770     0.0474     0.0446
  Corrupted: jpeg_compression        0.7822     0.8148     0.0837     0.0828
  Corrupted: brightness              0.8116     0.8338     0.0414     0.0410
  Corrupted: shot_noise              0.7416     0.7798     0.0925     0.0989
  OOD: SVHN                          0.1564     0.1545     0.1059     0.2009
  OOD: CIFAR-10                      0.4083     0.4417     0.0706     0.1268
  OOD: CIFAR-100                     0.3903     0.4138     0.0791     0.1284
  OOD: STL10                         0.3821     0.4317     0.0488     0.1104
  OOD: DTD                           0.3214     0.3446     0.1204     0.1877
  OOD: FashionMNIST                  0.0933     0.0852     0.0819     0.1464
  OOD: MNIST                         0.1642     0.1732     0.0768     0.1246

============================================================
  4a. OOD Detection — SEEN OOD (used in Phase 2 training)
============================================================

  Clean TinyImageNet (neg) vs SVHN (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.9681
  Teacher TU (ensemble entropy)              0.9536
  Student EU (learned)                       0.8713
  Student TU (entropy)                       0.9343
  1 - max softmax prob                       0.9145

  Clean TinyImageNet (neg) vs CIFAR-100 (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.8633
  Teacher TU (ensemble entropy)              0.8576
  Student EU (learned)                       0.7533
  Student TU (entropy)                       0.8002
  1 - max softmax prob                       0.7898

============================================================
  4b. OOD Detection — UNSEEN OOD (not in training)
============================================================

  Clean TinyImageNet (neg) vs CIFAR-10 (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.8678
  Teacher TU (ensemble entropy)              0.8562
  Student EU (learned)                       0.7265
  Student TU (entropy)                       0.7750
  1 - max softmax prob                       0.7662

  Clean TinyImageNet (neg) vs STL10 (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.8438
  Teacher TU (ensemble entropy)              0.8815
  Student EU (learned)                       0.6184
  Student TU (entropy)                       0.7495
  1 - max softmax prob                       0.7347

  Clean TinyImageNet (neg) vs DTD (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.9376
  Teacher TU (ensemble entropy)              0.9503
  Student EU (learned)                       0.8335
  Student TU (entropy)                       0.9132
  1 - max softmax prob                       0.8861

  Clean TinyImageNet (neg) vs FashionMNIST (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.9268
  Teacher TU (ensemble entropy)              0.9397
  Student EU (learned)                       0.7989
  Student TU (entropy)                       0.8478
  1 - max softmax prob                       0.8320

  Clean TinyImageNet (neg) vs MNIST (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.9058
  Teacher TU (ensemble entropy)              0.9381
  Student EU (learned)                       0.7675
  Student TU (entropy)                       0.8446
  1 - max softmax prob                       0.8297

============================================================
  4c. OOD Detection — Shifted TinyImageNet vs OOD
============================================================

  Shifted TinyImageNet (neg) vs SVHN (seen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.7536
  Student TU (entropy)                       0.9014

  Shifted TinyImageNet (neg) vs CIFAR-10 (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.6085
  Student TU (entropy)                       0.7123

  Shifted TinyImageNet (neg) vs CIFAR-100 (seen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.6373
  Student TU (entropy)                       0.7422

  Shifted TinyImageNet (neg) vs STL10 (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.5037
  Student TU (entropy)                       0.6860

  Shifted TinyImageNet (neg) vs DTD (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.7343
  Student TU (entropy)                       0.8777

  Shifted TinyImageNet (neg) vs FashionMNIST (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.6757
  Student TU (entropy)                       0.7960

  Shifted TinyImageNet (neg) vs MNIST (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.6466
  Student TU (entropy)                       0.7863

============================================================
  4d. Distribution Shift Detection
============================================================

  Clean TinyImageNet (neg) vs Shifted TinyImageNet (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble)                      0.6175
  Student EU (learned)                       0.6028
  Student TU (entropy)                       0.5751

============================================================
  5. Uncertainty Decomposition (TinyImageNet val)
============================================================
  Metric       Teacher mean   Student mean    Pearson   Spearman
  ----------------------------------------------------------
  TU               0.909620       0.997309     0.8169     0.7968
  AU               0.874266       0.960818     0.8094     0.7868
  EU               0.035354       0.036491     0.6998     0.7628

============================================================
  6. OOD Detection AUROC — Full Comparison
============================================================
  Dataset        Type     Tea EU  Tea TU |  Stu EU  Stu TU  Stu AU |  Sgl(H)
  ------------------------------------------------------------------------------
  SVHN           seen     0.9681  0.9536 |  0.8713  0.9343  0.9331 |  0.9409
  CIFAR-10       unseen   0.8678  0.8562 |  0.7265  0.7750  0.7723 |  0.8467
  CIFAR-100      seen     0.8633  0.8576 |  0.7533  0.8002  0.7980 |  0.8506
  STL10          unseen   0.8438  0.8815 |  0.6184  0.7495  0.7496 |  0.8862
  DTD            unseen   0.9376  0.9503 |  0.8335  0.9132  0.9112 |  0.9390
  FashionMNIST   unseen   0.9268  0.9397 |  0.7989  0.8478  0.8458 |  0.9326
  MNIST          unseen   0.9058  0.9381 |  0.7675  0.8446  0.8435 |  0.9202

  Tea EU = I[y;θ|x] (ensemble MI),  Tea TU = H[Ē[p]] (ensemble entropy)
  Stu EU = EU head,  Stu TU = H[softmax(logits)],  Stu AU = TU - EU
  Sgl(H) = entropy of one LoRA member (member_0)
  Fair comparison: Tea EU↔Stu EU (epistemic); Tea TU↔Stu TU↔Sgl(H) (entropy)

============================================================
  7. Baseline: Single Ensemble Member
============================================================
  Single member test accuracy: 87.62%

  Dataset          Type        Entropy  1-MaxProb
  ------------------------------------------------
  SVHN             seen         0.9409     0.9292
  CIFAR-10         unseen       0.8467     0.8396
  CIFAR-100        seen         0.8506     0.8412
  STL10            unseen       0.8862     0.8641
  DTD              unseen       0.9390     0.9199
  FashionMNIST     unseen       0.9326     0.9222
  MNIST            unseen       0.9202     0.8996

============================================================
  8. Selective Prediction (AURC)
============================================================
  Method                              AURC↓   OracleGap↓   @90%cov↑   @80%cov↑
  --------------------------------------------------------------------------
  Teacher EU                       0.033745     0.018062     0.8826     0.9316
  Student EU (ours)                0.048101     0.032417     0.8704     0.9046
  Student entropy                  0.037328     0.021644     0.8852     0.9279
  1 - MaxProb                      0.035668     0.019985     0.8858     0.9311
  Oracle                           0.015684     0.000000     0.9202     1.0000
  Random (baseline)                0.171800            —          —          —

============================================================
  9. Inference Throughput
============================================================
