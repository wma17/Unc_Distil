Loaded student (spearman=0.8059, pearson=0.7930)

============================================================
  1. Accuracy
============================================================
  Teacher (ensemble):  88.01%
  Student (distilled): 87.51%

============================================================
  2. Correctness Agreement
============================================================
  Both correct:            8512  (85.1%)
  Both wrong:               960  (9.6%)
  Student correct only:     239  (2.4%)
  Teacher correct only:     289  (2.9%)
  Total agreement:       94.72%

============================================================
  3. EU Correlation (student EU vs teacher EU)
============================================================
  Dataset                           Pearson   Spearman   Stu mean   Tea mean
  ----------------------------------------------------------------------
  Clean TinyImageNet val             0.7948     0.8060     0.0778     0.0674
  Corrupted: gaussian_noise          0.8287     0.8436     0.0877     0.0827
  Corrupted: gaussian_blur           0.8518     0.8479     0.0787     0.0749
  Corrupted: low_contrast            0.8468     0.8257     0.0788     0.0792
  Corrupted: jpeg_compression        0.8542     0.8792     0.1226     0.1256
  Corrupted: brightness              0.8616     0.8632     0.0828     0.0806
  Corrupted: shot_noise              0.7681     0.8197     0.1180     0.1210
  OOD: SVHN                          0.8285     0.8219     0.3082     0.3271
  OOD: CIFAR-10                      0.9541     0.9512     0.2062     0.2096
  OOD: CIFAR-100                     0.9584     0.9540     0.2046     0.2107
  OOD: STL10                         0.4948     0.5293     0.1504     0.1232
  OOD: DTD                           0.3459     0.3241     0.2516     0.2226
  OOD: FashionMNIST                  0.3383     0.3302     0.2739     0.2691
  OOD: MNIST                         0.4700     0.4603     0.2689     0.2313

============================================================
  4a. OOD Detection — SEEN OOD (used in Phase 2 training)
============================================================

  Clean TinyImageNet (neg) vs SVHN (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.9660
  Teacher TU (ensemble entropy)              0.9796
  Student EU (learned)                       0.9544
  Student TU (entropy)                       0.8959
  1 - max softmax prob                       0.8656

  Clean TinyImageNet (neg) vs CIFAR-100 (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.8544
  Teacher TU (ensemble entropy)              0.8780
  Student EU (learned)                       0.8285
  Student TU (entropy)                       0.8338
  1 - max softmax prob                       0.8191

============================================================
  4b. OOD Detection — UNSEEN OOD (not in training)
============================================================

  Clean TinyImageNet (neg) vs CIFAR-10 (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.8596
  Teacher TU (ensemble entropy)              0.8815
  Student EU (learned)                       0.8402
  Student TU (entropy)                       0.8255
  1 - max softmax prob                       0.8125

  Clean TinyImageNet (neg) vs STL10 (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.7416
  Teacher TU (ensemble entropy)              0.8574
  Student EU (learned)                       0.7463
  Student TU (entropy)                       0.8031
  1 - max softmax prob                       0.7817

  Clean TinyImageNet (neg) vs DTD (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.8869
  Teacher TU (ensemble entropy)              0.9493
  Student EU (learned)                       0.8882
  Student TU (entropy)                       0.9117
  1 - max softmax prob                       0.8862

  Clean TinyImageNet (neg) vs FashionMNIST (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.9262
  Teacher TU (ensemble entropy)              0.9401
  Student EU (learned)                       0.9255
  Student TU (entropy)                       0.9204
  1 - max softmax prob                       0.9046

  Clean TinyImageNet (neg) vs MNIST (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble MI)                   0.9050
  Teacher TU (ensemble entropy)              0.9401
  Student EU (learned)                       0.9362
  Student TU (entropy)                       0.8187
  1 - max softmax prob                       0.8077

============================================================
  4c. OOD Detection — Shifted TinyImageNet vs OOD
============================================================

  Shifted TinyImageNet (neg) vs SVHN (seen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.9299
  Student TU (entropy)                       0.8620

  Shifted TinyImageNet (neg) vs CIFAR-10 (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.7957
  Student TU (entropy)                       0.7824

  Shifted TinyImageNet (neg) vs CIFAR-100 (seen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.7847
  Student TU (entropy)                       0.7947

  Shifted TinyImageNet (neg) vs STL10 (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.6947
  Student TU (entropy)                       0.7536

  Shifted TinyImageNet (neg) vs DTD (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.8516
  Student TU (entropy)                       0.8833

  Shifted TinyImageNet (neg) vs FashionMNIST (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.8927
  Student TU (entropy)                       0.8935

  Shifted TinyImageNet (neg) vs MNIST (unseen) (pos)
  Method                                      AUROC
  ------------------------------------------------
  Student EU (learned)                       0.9030
  Student TU (entropy)                       0.7701

============================================================
  4d. Distribution Shift Detection
============================================================

  Clean TinyImageNet (neg) vs Shifted TinyImageNet (pos)
  Method                                      AUROC
  ------------------------------------------------
  Teacher EU (ensemble)                      0.5836
  Student EU (learned)                       0.5484
  Student TU (entropy)                       0.5649

============================================================
  5. Uncertainty Decomposition (TinyImageNet val)
============================================================
  Metric       Teacher mean   Student mean    Pearson   Spearman
  ----------------------------------------------------------
  TU               0.970446       0.945582     0.8829     0.8656
  AU               0.903068       0.867801     0.8696     0.8416
  EU               0.067379       0.077781     0.7948     0.8060

============================================================
  6. OOD Detection AUROC — Full Comparison
============================================================
  Dataset        Type     Tea EU  Tea TU |  Stu EU  Stu TU  Stu AU |  Sgl(H)
  ------------------------------------------------------------------------------
  SVHN           seen     0.9660  0.9796 |  0.9544  0.8959  0.8584 |  0.9578
  CIFAR-10       unseen   0.8596  0.8815 |  0.8402  0.8255  0.8101 |  0.8464
  CIFAR-100      seen     0.8544  0.8780 |  0.8285  0.8338  0.8213 |  0.8519
  STL10          unseen   0.7416  0.8574 |  0.7463  0.8031  0.7963 |  0.8272
  DTD            unseen   0.8869  0.9493 |  0.8882  0.9117  0.9036 |  0.9222
  FashionMNIST   unseen   0.9262  0.9401 |  0.9255  0.9204  0.9097 |  0.9051
  MNIST          unseen   0.9050  0.9401 |  0.9362  0.8187  0.7720 |  0.8490

  Tea EU = I[y;θ|x] (ensemble MI),  Tea TU = H[Ē[p]] (ensemble entropy)
  Stu EU = EU head,  Stu TU = H[softmax(logits)],  Stu AU = TU - EU
  Sgl(H) = entropy of one LoRA member (member_0)
  Fair comparison: Tea EU↔Stu EU (epistemic); Tea TU↔Stu TU↔Sgl(H) (entropy)

============================================================
  7. Baseline: Single Ensemble Member
============================================================
  Single member test accuracy: 86.75%

  Dataset          Type        Entropy  1-MaxProb
  ------------------------------------------------
  SVHN             seen         0.9578     0.9463
  CIFAR-10         unseen       0.8464     0.8348
  CIFAR-100        seen         0.8519     0.8401
  STL10            unseen       0.8272     0.8103
  DTD              unseen       0.9222     0.9024
  FashionMNIST     unseen       0.9051     0.8886
  MNIST            unseen       0.8490     0.8316

============================================================
Evaluation complete.
============================================================
