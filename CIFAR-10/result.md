Loaded student from ./checkpoints/student.pt
  Phase 1 acc=95.83%  EU Pearson=0.7337124943733215  Spearman=0.7378047108650208

============================================================
  1. Accuracy
============================================================
  Teacher (ensemble):  96.54%
  Student (distilled): 95.83%

============================================================
  2. Correctness Agreement
============================================================
  Both correct:           9494  (94.9%)
  Both wrong:              257  (2.6%)
  Student correct only:     89  (0.9%)
  Teacher correct only:    160  (1.6%)
  Total agreement:       97.51%
  Same predicted class:  97.21%

============================================================
  3. EU Correlation (student EU vs teacher EU)
============================================================
  Dataset                           Pearson   Spearman   Stu mean   Tea mean
  ----------------------------------------------------------------------
  Clean CIFAR-10 test                0.7337     0.7378     0.0713     0.0478
    └ misclassified only             0.3701     0.3378
  Corrupted: gaussian_noise          0.7466     0.8205     0.1460     0.1469
  Corrupted: gaussian_blur           0.5209     0.4492     0.4179     0.4678
  Corrupted: low_contrast            0.6954     0.7723     0.1352     0.1194
  OOD: SVHN                          0.4435     0.4170     0.4080     0.4623
  OOD: CIFAR-100                     0.5639     0.5527     0.2764     0.3036
  OOD: MNIST                         0.4189     0.4012     0.2910     0.2799
  OOD: FashionMNIST                  0.4236     0.3612     0.3302     0.3259
  OOD: STL10                         0.7514     0.8280     0.1413     0.1328
  OOD: DTD                           0.3793     0.3421     0.3102     0.3877

============================================================
  4a. OOD Detection — SEEN OOD (used in Phase 2 training)
============================================================

  Clean CIFAR-10 (neg) vs SVHN (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.9741
  Student EU (learned)                    0.9891
  Student entropy (softmax)               0.9617
  1 - max softmax prob                    0.9536
  ── EU stats ──
        Clean CIFAR-10  teacher=0.0478±0.1062  student=0.0713±0.0885
                  SVHN  teacher=0.4623±0.1537  student=0.4080±0.0675

  Clean CIFAR-10 (neg) vs CIFAR-100 (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.9113
  Student EU (learned)                    0.9159
  Student entropy (softmax)               0.8990
  1 - max softmax prob                    0.8954
  ── EU stats ──
        Clean CIFAR-10  teacher=0.0478±0.1062  student=0.0713±0.0885
             CIFAR-100  teacher=0.3036±0.1879  student=0.2764±0.1094

============================================================
  4b. OOD Detection — UNSEEN OOD (not in training)
============================================================

  Clean CIFAR-10 (neg) vs MNIST (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.9244
  Student EU (learned)                    0.9376
  Student entropy (softmax)               0.9261
  1 - max softmax prob                    0.9187
  ── EU stats ──
        Clean CIFAR-10  teacher=0.0478±0.1062  student=0.0713±0.0885
                 MNIST  teacher=0.2799±0.1442  student=0.2910±0.0871

  Clean CIFAR-10 (neg) vs FashionMNIST (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.9398
  Student EU (learned)                    0.9599
  Student entropy (softmax)               0.9521
  1 - max softmax prob                    0.9441
  ── EU stats ──
        Clean CIFAR-10  teacher=0.0478±0.1062  student=0.0713±0.0885
          FashionMNIST  teacher=0.3259±0.1523  student=0.3302±0.0854

  Clean CIFAR-10 (neg) vs STL10 (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.6661
  Student EU (learned)                    0.6621
  Student entropy (softmax)               0.6790
  1 - max softmax prob                    0.6787
  ── EU stats ──
        Clean CIFAR-10  teacher=0.0478±0.1062  student=0.0713±0.0885
                 STL10  teacher=0.1328±0.1740  student=0.1413±0.1291

  Clean CIFAR-10 (neg) vs DTD (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.9489
  Student EU (learned)                    0.9475
  Student entropy (softmax)               0.9307
  1 - max softmax prob                    0.9264
  ── EU stats ──
        Clean CIFAR-10  teacher=0.0478±0.1062  student=0.0713±0.0885
                   DTD  teacher=0.3877±0.1937  student=0.3102±0.0934

============================================================
  4c. OOD Detection — Shifted CIFAR-10 (ID) vs OOD
============================================================

  Shifted CIFAR-10 (neg) vs SVHN (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.7622
  Student EU (learned)                    0.7999
  Student entropy (softmax)               0.7425
  1 - max softmax prob                    0.7292
  ── EU stats ──
      Shifted CIFAR-10  teacher=0.2447±0.2351  student=0.2330±0.1675
                  SVHN  teacher=0.4623±0.1537  student=0.4080±0.0675

  Shifted CIFAR-10 (neg) vs CIFAR-100 (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.5983
  Student EU (learned)                    0.5611
  Student entropy (softmax)               0.6013
  1 - max softmax prob                    0.6077
  ── EU stats ──
      Shifted CIFAR-10  teacher=0.2447±0.2351  student=0.2330±0.1675
             CIFAR-100  teacher=0.3036±0.1879  student=0.2764±0.1094

  Shifted CIFAR-10 (neg) vs MNIST (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.5758
  Student EU (learned)                    0.5755
  Student entropy (softmax)               0.6546
  1 - max softmax prob                    0.6454
  ── EU stats ──
      Shifted CIFAR-10  teacher=0.2447±0.2351  student=0.2330±0.1675
                 MNIST  teacher=0.2799±0.1442  student=0.2910±0.0871

  Shifted CIFAR-10 (neg) vs FashionMNIST (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.6218
  Student EU (learned)                    0.6424
  Student entropy (softmax)               0.7282
  1 - max softmax prob                    0.7103
  ── EU stats ──
      Shifted CIFAR-10  teacher=0.2447±0.2351  student=0.2330±0.1675
          FashionMNIST  teacher=0.3259±0.1523  student=0.3302±0.0854

  Shifted CIFAR-10 (neg) vs STL10 (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.3686
  Student EU (learned)                    0.3323
  Student entropy (softmax)               0.3777
  1 - max softmax prob                    0.3853
  ── EU stats ──
      Shifted CIFAR-10  teacher=0.2447±0.2351  student=0.2330±0.1675
                 STL10  teacher=0.1328±0.1740  student=0.1413±0.1291

  Shifted CIFAR-10 (neg) vs DTD (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.6820
  Student EU (learned)                    0.6112
  Student entropy (softmax)               0.6537
  1 - max softmax prob                    0.6596
  ── EU stats ──
      Shifted CIFAR-10  teacher=0.2447±0.2351  student=0.2330±0.1675
                   DTD  teacher=0.3877±0.1937  student=0.3102±0.0934

============================================================
  4d. Distribution Shift Detection
============================================================

  Clean CIFAR-10 (neg) vs Shifted CIFAR-10 (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.7624
  Student EU (learned)                    0.7930
  Student entropy (softmax)               0.7668
  1 - max softmax prob                    0.7641
  ── EU stats ──
        Clean CIFAR-10  teacher=0.0478±0.1062  student=0.0713±0.0885
      Shifted CIFAR-10  teacher=0.2447±0.2351  student=0.2330±0.1675

============================================================
  5. Uncertainty Decomposition (Clean CIFAR-10 test)
============================================================
  Metric       Teacher mean   Student mean    Pearson   Spearman
  ----------------------------------------------------------
  TU                 0.2202         0.1724     0.7937     0.7008
  AU                 0.1723         0.1013     0.6757     0.3401
  EU                 0.0478         0.0713     0.7337     0.7378

============================================================
  6. OOD Detection: CIFAR-10 vs OOD — Decomposed Uncertainties
============================================================
  Dataset          Type      Tea TU   Tea EU   Tea AU |   Stu TU   Stu EU   Stu AU
  -----------------------------------------------------------------------------------
  SVHN             seen      0.9878   0.9741   0.9835 |   0.9617   0.9891   0.8253
  CIFAR-100        seen      0.9285   0.9113   0.9279 |   0.8990   0.9159   0.7443
  MNIST            unseen    0.9625   0.9244   0.9682 |   0.9261   0.9376   0.8113
  FashionMNIST     unseen    0.9735   0.9398   0.9763 |   0.9521   0.9599   0.8632
  STL10            unseen    0.6875   0.6661   0.6876 |   0.6790   0.6621   0.6150
  DTD              unseen    0.9633   0.9489   0.9594 |   0.9307   0.9475   0.7973

  Student TU = H[softmax(logits)], EU = EU head, AU = TU - EU
  Expectation: EU >> AU on OOD (epistemic dominates)
               AU ≈ stable on ID vs OOD (aleatoric is data-intrinsic)
  Loaded single member (member_0, acc=94.31%)

============================================================
  7. Baseline: Single Ensemble Member — OOD Detection
============================================================
  Single member test accuracy: 94.31%

  Dataset          Type       Entropy  1-MaxProb
  ------------------------------------------------
  SVHN             seen        0.9321     0.9281
  CIFAR-100        seen        0.8365     0.8365
  MNIST            unseen      0.8933     0.8898
  FashionMNIST     unseen      0.9240     0.9204
  STL10            unseen      0.6256     0.6267
  DTD              unseen      0.8847     0.8844
  Shifted CIFAR-10 shift       0.7053     0.7043


    Dataset          Type    Tea EU |  Stu EU  | SingleModel
  -----------------------------------------------------------------------------------
  SVHN             seen      0.9741 |   0.9891 | 0.9321
  CIFAR-100        seen      0.9113 |   0.9159 | 0.8365
  MNIST            unseen    0.9244 |   0.9376 | 0.8933
  FashionMNIST     unseen    0.9398 |   0.9599 | 0.9240 
  STL10            unseen    0.6661 |   0.6621 | 0.6256
  DTD              unseen    0.9489 |   0.9475 | 0.8847
