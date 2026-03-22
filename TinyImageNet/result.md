Loaded student (spearman=0.8407, pearson=0.7880)

============================================================
  1. Accuracy
============================================================
  Teacher (ensemble):  88.13%
  Student (distilled): 87.50%

============================================================
  1b. Calibration Metrics (ECE-15, NLL, Brier)
============================================================
  Model                              ECE-15        NLL      Brier
  ------------------------------------------------------------
  Teacher (ensemble)                 0.0763     0.4862     0.1778
  Student (distilled)                0.0484     0.5139     0.1833

============================================================
  2. Correctness Agreement
============================================================
  Both correct:            8537  (85.4%)
  Both wrong:               974  (9.7%)
  Student correct only:     213  (2.1%)
  Teacher correct only:     276  (2.8%)
  Total agreement:       95.11%

============================================================
  3. EU Correlation (student EU vs teacher EU)
============================================================
  Dataset                           Pearson   Spearman   Stu mean   Tea mean
  ----------------------------------------------------------------------
  Clean TinyImageNet val             0.7942     0.8410     0.0841     0.0623
  Corrupted: gaussian_noise          0.8466     0.8739     0.1019     0.0835
  Corrupted: gaussian_blur           0.8586     0.8810     0.0890     0.0723
  Corrupted: low_contrast            0.8089     0.8138     0.1362     0.1183
  Corrupted: jpeg_compression        0.8566     0.8862     0.1370     0.1244
  Corrupted: brightness              0.8486     0.8802     0.0941     0.0771
  Corrupted: shot_noise              0.8157     0.8482     0.1389     0.1251
  OOD: SVHN                          0.0337     0.0135     0.2132     0.3435
  OOD: CIFAR-10                      0.5553     0.5576     0.1746     0.2042
  OOD: CIFAR-100                     0.5429     0.5302     0.1893     0.2080
  OOD: STL10                         0.5694     0.5973     0.1484     0.1508
  OOD: DTD                           0.3483     0.3150     0.2461     0.2301
  OOD: FashionMNIST                  0.3506     0.3392     0.2107     0.2682
  OOD: MNIST                         0.3586     0.3396     0.1962     0.1948

============================================================
  4a. OOD Detection — SEEN OOD (used in Phase 2 training)
============================================================

  Clean TinyImageNet (neg) vs SVHN (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.9795
  Teacher TU (ensemble entropy)              0.9739
  Student EU (learned)                       0.8974
  Student TU (entropy)                       0.8456
  1 - max softmax prob                       0.8050

  Clean TinyImageNet (neg) vs CIFAR-100 (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.8673
  Teacher TU (ensemble entropy)              0.8714
  Student EU (learned)                       0.8262
  Student TU (entropy)                       0.8400
  1 - max softmax prob                       0.8253

============================================================
  4b. OOD Detection — UNSEEN OOD (not in training)
============================================================

  Clean TinyImageNet (neg) vs CIFAR-10 (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.8719
  Teacher TU (ensemble entropy)              0.8733
  Student EU (learned)                       0.8035
  Student TU (entropy)                       0.8353
  1 - max softmax prob                       0.8217

  Clean TinyImageNet (neg) vs STL10 (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.8115
  Teacher TU (ensemble entropy)              0.8876
  Student EU (learned)                       0.7438
  Student TU (entropy)                       0.8308
  1 - max softmax prob                       0.8071

  Clean TinyImageNet (neg) vs DTD (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.9092
  Teacher TU (ensemble entropy)              0.9538
  Student EU (learned)                       0.9074
  Student TU (entropy)                       0.9291
  1 - max softmax prob                       0.9037

  Clean TinyImageNet (neg) vs FashionMNIST (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.9451
  Teacher TU (ensemble entropy)              0.9430
  Student EU (learned)                       0.8874
  Student TU (entropy)                       0.9097
  1 - max softmax prob                       0.8972

  Clean TinyImageNet (neg) vs MNIST (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.8951
  Teacher TU (ensemble entropy)              0.9215
  Student EU (learned)                       0.8796
  Student TU (entropy)                       0.8027
  1 - max softmax prob                       0.8110

============================================================
  4c. OOD Detection — Shifted TinyImageNet vs OOD
============================================================

  Shifted TinyImageNet (neg) vs SVHN (seen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.8257
  Student TU (entropy)                       0.7306

  Shifted TinyImageNet (neg) vs CIFAR-10 (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.7069
  Student TU (entropy)                       0.7328

  Shifted TinyImageNet (neg) vs CIFAR-100 (seen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.7386
  Student TU (entropy)                       0.7473

  Shifted TinyImageNet (neg) vs STL10 (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.6312
  Student TU (entropy)                       0.7186

  Shifted TinyImageNet (neg) vs DTD (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.8490
  Student TU (entropy)                       0.8711

  Shifted TinyImageNet (neg) vs FashionMNIST (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.8136
  Student TU (entropy)                       0.8429

  Shifted TinyImageNet (neg) vs MNIST (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.7990
  Student TU (entropy)                       0.6715

============================================================
  4d. Distribution Shift Detection
============================================================

  Clean TinyImageNet (neg) vs Shifted TinyImageNet (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble)                      0.6326
  Student EU (learned)                       0.6181
  Student TU (entropy)                       0.6434

============================================================
  5. Uncertainty Decomposition (TinyImageNet val)
============================================================
  Metric       Teacher mean   Student mean    Pearson   Spearman
  ----------------------------------------------------------
  TU               1.010657       0.957564     0.9058     0.8829
  AU               0.948397       0.873471     0.8942     0.8480
  EU               0.062259       0.084092     0.7942     0.8410

============================================================
  6. OOD Detection AUROC — Full Comparison
============================================================
  Dataset        Type     Tea EU  Tea TU |  Stu EU  Stu TU  Stu AU |  Sgl(H)
  ------------------------------------------------------------------------------
  SVHN           seen     0.9795  0.9739 |  0.8974  0.8456  0.8297 |  0.9656
  CIFAR-10       unseen   0.8719  0.8733 |  0.8035  0.8353  0.8259 |  0.8603
  CIFAR-100      seen     0.8673  0.8714 |  0.8262  0.8400  0.8290 |  0.8581
  STL10          unseen   0.8115  0.8876 |  0.7438  0.8308  0.8292 |  0.8527
  DTD            unseen   0.9092  0.9538 |  0.9074  0.9291  0.9233 |  0.9327
  FashionMNIST   unseen   0.9451  0.9430 |  0.8874  0.9097  0.9023 |  0.9397
  MNIST          unseen   0.8951  0.9215 |  0.8796  0.8027  0.7811 |  0.9121

  Tea EU = I[y;θ|x] (ensemble MI),  Tea TU = H[Ē[p]] (ensemble entropy)
  Stu EU = EU head,  Stu TU = H[softmax(logits)],  Stu AU = TU - EU
  Sgl(H) = entropy of one LoRA member (member_0)
  Fair comparison: Tea EU↔Stu EU (epistemic); Tea TU↔Stu TU↔Sgl(H) (entropy)

============================================================
  7. Baseline: Single Ensemble Member
============================================================
  Single member test accuracy: 87.01%

  Dataset          Type        Entropy  1-MaxProb
  ------------------------------------------------
  SVHN             seen         0.9656     0.9555
  CIFAR-10         unseen       0.8603     0.8507
  CIFAR-100        seen         0.8581     0.8470
  STL10            unseen       0.8527     0.8345
  DTD              unseen       0.9327     0.9120
  FashionMNIST     unseen       0.9397     0.9276
  MNIST            unseen       0.9121     0.8916

============================================================
  8. Selective Prediction (AURC)
============================================================
  Method                              AURC↓   OracleGap↓   @90%cov↑   @80%cov↑
  --------------------------------------------------------------------------
  Teacher EU                       0.023592     0.015432     0.9167     0.9530
  Student EU (ours)                0.028578     0.020418     0.9090     0.9425
  Student entropy                  0.023662     0.015502     0.9260     0.9589
  1 - MaxProb                      0.021845     0.013685     0.9271     0.9634
  Oracle                           0.008160     0.000000     0.9722     1.0000
  Random (baseline)                0.125000            —          —          —

============================================================
  9. Inference Throughput
============================================================
  Model                                     Samples/sec   Speedup vs ens
  ----------------------------------------------------------------------
  Ensemble (K=16, sequential)                       263            1.00x
  Single member                                   5,030           19.11x
  Student (single pass)                           6,828           25.94x

============================================================
Evaluation complete.
============================================================
