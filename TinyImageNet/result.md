Loaded student (spearman=0.7986, pearson=0.6454)

============================================================
  1. Accuracy
============================================================
  Teacher (ensemble):  88.13%
  Student (distilled): 85.67%

============================================================
  1b. Calibration Metrics (ECE-15, NLL, Brier)
============================================================
  Model                              ECE-15        NLL      Brier
  ------------------------------------------------------------
  Teacher (ensemble)                 0.0763     0.4862     0.1778
  Student (distilled)                0.0385     0.5902     0.2085

============================================================
  2. Correctness Agreement
============================================================
  Both correct:            8386  (83.9%)
  Both wrong:              1006  (10.1%)
  Student correct only:     181  (1.8%)
  Teacher correct only:     427  (4.3%)
  Total agreement:       93.92%

============================================================
  3. EU Correlation (student EU vs teacher EU)
============================================================
  Dataset                           Pearson   Spearman   Stu mean   Tea mean
  ----------------------------------------------------------------------
  Clean TinyImageNet val             0.6518     0.7986     0.0738     0.0623
  Corrupted: gaussian_noise          0.6203     0.8237     0.0981     0.0835
  Corrupted: gaussian_blur           0.6145     0.8085     0.0903     0.0723
  Corrupted: low_contrast            0.6015     0.7223     0.0957     0.1183
  Corrupted: jpeg_compression        0.6728     0.8515     0.1050     0.1244
  Corrupted: brightness              0.6590     0.8313     0.0827     0.0771
  Corrupted: shot_noise              0.6521     0.7812     0.1116     0.1251
  OOD: SVHN                          0.0643    -0.0261     0.1157     0.3435
  OOD: CIFAR-10                      0.4833     0.4972     0.1205     0.2042
  OOD: CIFAR-100                     0.5116     0.5143     0.1184     0.2080
  OOD: STL10                         0.5329     0.5724     0.1080     0.1508
  OOD: DTD                           0.2911     0.3069     0.1373     0.2301
  OOD: FashionMNIST                  0.3489     0.3052     0.1258     0.2682
  OOD: MNIST                         0.0846    -0.0227     0.1194     0.1948

============================================================
  4a. OOD Detection — SEEN OOD (used in Phase 2 training)
============================================================

  Clean TinyImageNet (neg) vs SVHN (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.9795
  Teacher TU (ensemble entropy)              0.9739
  Student EU (learned)                       0.8439
  Student EU (calibrated)                    0.8437
  Student TU (entropy)                       0.7460
  1 - max softmax prob                       0.7207

  Clean TinyImageNet (neg) vs CIFAR-100 (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.8673
  Teacher TU (ensemble entropy)              0.8714
  Student EU (learned)                       0.8556
  Student EU (calibrated)                    0.8552
  Student TU (entropy)                       0.7877
  1 - max softmax prob                       0.7790

============================================================
  4b. OOD Detection — UNSEEN OOD (not in training)
============================================================

  Clean TinyImageNet (neg) vs CIFAR-10 (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.8719
  Teacher TU (ensemble entropy)              0.8733
  Student EU (learned)                       0.8722
  Student EU (calibrated)                    0.8716
  Student TU (entropy)                       0.7756
  1 - max softmax prob                       0.7703

  Clean TinyImageNet (neg) vs STL10 (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.8115
  Teacher TU (ensemble entropy)              0.8876
  Student EU (learned)                       0.7609
  Student EU (calibrated)                    0.7604
  Student TU (entropy)                       0.6910
  1 - max softmax prob                       0.6885

  Clean TinyImageNet (neg) vs DTD (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.9092
  Teacher TU (ensemble entropy)              0.9538
  Student EU (learned)                       0.9386
  Student EU (calibrated)                    0.9374
  Student TU (entropy)                       0.8874
  1 - max softmax prob                       0.8596

  Clean TinyImageNet (neg) vs FashionMNIST (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.9451
  Teacher TU (ensemble entropy)              0.9430
  Student EU (learned)                       0.9281
  Student EU (calibrated)                    0.9273
  Student TU (entropy)                       0.9015
  1 - max softmax prob                       0.8856

  Clean TinyImageNet (neg) vs MNIST (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.8951
  Teacher TU (ensemble entropy)              0.9215
  Student EU (learned)                       0.8885
  Student EU (calibrated)                    0.8882
  Student TU (entropy)                       0.8095
  1 - max softmax prob                       0.7953

============================================================
  4c. OOD Detection — Shifted TinyImageNet vs OOD
============================================================

  Shifted TinyImageNet (neg) vs SVHN (seen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.7035
  Student TU (entropy)                       0.6989

  Shifted TinyImageNet (neg) vs CIFAR-10 (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.7600
  Student TU (entropy)                       0.7325

  Shifted TinyImageNet (neg) vs CIFAR-100 (seen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.7370
  Student TU (entropy)                       0.7482

  Shifted TinyImageNet (neg) vs STL10 (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.5859
  Student TU (entropy)                       0.6368

  Shifted TinyImageNet (neg) vs DTD (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.8873
  Student TU (entropy)                       0.8569

  Shifted TinyImageNet (neg) vs FashionMNIST (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.8597
  Student TU (entropy)                       0.8733

  Shifted TinyImageNet (neg) vs MNIST (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.7789
  Student TU (entropy)                       0.7615

============================================================
  4d. Distribution Shift Detection
============================================================

  Clean TinyImageNet (neg) vs Shifted TinyImageNet (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble)                      0.6326
  Student EU (learned)                       0.7182
  Student TU (entropy)                       0.5591

============================================================
  5. Uncertainty Decomposition (TinyImageNet val)
============================================================
  Metric       Teacher mean   Student mean    Pearson   Spearman
  ----------------------------------------------------------
  TU               1.010657       0.979346     0.8792     0.8601
  AU               0.948397       0.905574     0.8735     0.8450
  EU               0.062259       0.073772     0.6518     0.7986

============================================================
  6. OOD Detection AUROC — Full Comparison
============================================================
  Dataset        Type     Tea EU  Tea TU |  Stu EU  Stu TU  Stu AU |  Sgl(H)
  ------------------------------------------------------------------------------
  SVHN           seen     0.9795  0.9739 |  0.8439  0.7460  0.7249 |  0.9656
  CIFAR-10       unseen   0.8719  0.8733 |  0.8722  0.7756  0.7601 |  0.8603
  CIFAR-100      seen     0.8673  0.8714 |  0.8556  0.7877  0.7754 |  0.8581
  STL10          unseen   0.8115  0.8876 |  0.7609  0.6910  0.6766 |  0.8527
  DTD            unseen   0.9092  0.9538 |  0.9386  0.8874  0.8817 |  0.9327
  FashionMNIST   unseen   0.9451  0.9430 |  0.9281  0.9015  0.8976 |  0.9397
  MNIST          unseen   0.8951  0.9215 |  0.8885  0.8095  0.8006 |  0.9121

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
  Teacher EU                       0.026167     0.015371     0.9073     0.9493
  Student EU (ours)                0.034612     0.023815     0.8996     0.9314
  Student entropy                  0.028660     0.017864     0.9096     0.9456
  1 - MaxProb                      0.026763     0.015966     0.9121     0.9490
  Oracle                           0.010796     0.000000     0.9519     1.0000
  Random (baseline)                0.143300            —          —          —

============================================================
  9. Inference Throughput
============================================================
  Model                                     Samples/sec   Speedup vs ens
  ----------------------------------------------------------------------
  Ensemble (K=16, sequential)                       265            1.00x
  Single member                                   5,066           19.10x
  Student (single pass)                           6,879           25.94x

============================================================
Evaluation complete.
============================================================
