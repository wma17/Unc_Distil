(maw6) maw6@islserver6:~/maw6/unc_regression/MNIST$ python evaluate_student.py --save_dir ./checkpoints --gpu 0
Loaded student from ./checkpoints/student.pt
  Phase 1 acc=99.48%  EU Pearson=0.7158544063568115  Spearman=0.7842811346054077

============================================================
  1. Accuracy
============================================================
  Teacher (ensemble):  99.55%
  Student (distilled): 99.48%

============================================================
  2. Correctness Agreement
============================================================
  Both correct:           9939  (99.4%)
  Both wrong:               36  (0.4%)
  Student correct only:      9  (0.1%)
  Teacher correct only:     16  (0.2%)
  Total agreement:       99.75%
  Same predicted class:  99.73%

============================================================
  3. EU Correlation (student EU vs teacher EU)
============================================================
  Dataset                           Pearson   Spearman   Stu mean   Tea mean
  ----------------------------------------------------------------------
  Clean MNIST test                   0.7634     0.7220     0.0174     0.0116
    └ misclassified only             0.4857     0.4434
  Corrupted: gaussian_noise          0.6995     0.6950     0.0400     0.0256
  Corrupted: gaussian_blur           0.7457     0.6943     0.3850     0.3880
  Corrupted: low_contrast            0.8628     0.8000     0.6814     0.6853
(maw6) maw6@islserver6:~/maw6/unc_regression/MNIST$ python plot_eu.py --save_dir ./checkpoints --gpu 0 --out_dir ./figuresA
Loading model and data...
Loaded student from ./checkpoints/student.pt
  Phase 1 acc=99.48%  EU Pearson=0.7158544063568115  Spearman=0.7842811346054077
Collecting EU for all datasets...
  Datasets: ['Clean MNIST', 'Shifted MNIST', 'FashionMNIST', 'Omniglot', 'EMNIST-Letters', 'CIFAR-10', 'SV
  Datasets: ['Clean MNIST', 'Shifted MNIST', 'FashionMNIST', 'Omniglot', 'EMNIST-Letters', 'CIFAR-10', 'SVHN']

Generating figures...
  Saved ./figuresA/eu_correlation.png
  Saved ./figuresA/eu_dist_teacher.png
  Saved ./figuresA/eu_dist_student.png
  Saved ./figuresA/eu_violin_comparison.png
  Saved ./figuresA/uncertainty_decomposition.png
  Saved ./figuresA/student_tu_eu_au_dist.png
  Saved ./figuresA/teacher_tu_eu_au_dist.png
  Saved ./figuresA/decomposition_teacher_vs_student.png

Done! Figures saved to: ./figuresA  Corrupted: inversion               0.8196     0.7772     0.5903     0.5904
  Corrupted: colored_background      0.5793     0.5396     0.5573     0.5616
  Corrupted: colored_digits          0.7575     0.7897     0.1905     0.1951
  Corrupted: salt_pepper             0.7893     0.8173     0.2098     0.2113
  Corrupted: pixelate                0.6779     0.6480     0.2695     0.2715
  OOD: FashionMNIST                  0.8286     0.8197     0.4235     0.4281
  OOD: Omniglot                      0.8033     0.7701     0.4647     0.4698
  OOD: EMNIST-Letters                0.5333     0.5569     0.1431     0.1519
  OOD: CIFAR-10                      0.3148     0.3164     0.6098     0.6994
  OOD: SVHN                          0.4525     0.4113     0.6535     0.8307

============================================================
  4a. OOD Detection — SEEN OOD
============================================================

  Clean MNIST (neg) vs FashionMNIST (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.9966
  Student EU (learned)                    0.9994
  Student entropy (softmax)               0.4689
  1 - max softmax prob                    0.4752

  Clean MNIST (neg) vs Omniglot (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.9997
  Student EU (learned)                    1.0000
  Student entropy (softmax)               0.7621
  1 - max softmax prob                    0.8103

============================================================
  4b. OOD Detection — UNSEEN OOD
============================================================

  Clean MNIST (neg) vs EMNIST-Letters (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.9509
  Student EU (learned)                    0.9638
  Student entropy (softmax)               0.9388
  1 - max softmax prob                    0.9385

  Clean MNIST (neg) vs CIFAR-10 (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   1.0000
  Student EU (learned)                    1.0000
  Student entropy (softmax)               0.0047
  1 - max softmax prob                    0.0050

  Clean MNIST (neg) vs SVHN (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   1.0000
  Student EU (learned)                    1.0000
  Student entropy (softmax)               0.0138
  1 - max softmax prob                    0.0150

============================================================
  4c. OOD Detection — Shifted MNIST vs OOD
============================================================

  Shifted MNIST (neg) vs FashionMNIST (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.5649
  Student EU (learned)                    0.5736
  Student entropy (softmax)               0.4686
  1 - max softmax prob                    0.4585

  Shifted MNIST (neg) vs Omniglot (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.6113
  Student EU (learned)                    0.6283
  Student entropy (softmax)               0.5306
  1 - max softmax prob                    0.5857

  Shifted MNIST (neg) vs EMNIST-Letters (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.2635
  Student EU (learned)                    0.2247
  Student entropy (softmax)               0.8160
  1 - max softmax prob                    0.7940

  Shifted MNIST (neg) vs CIFAR-10 (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.8733
  Student EU (learned)                    0.8101
  Student entropy (softmax)               0.0949
  1 - max softmax prob                    0.0436

  Shifted MNIST (neg) vs SVHN (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.9557
  Student EU (learned)                    0.8589
  Student entropy (softmax)               0.1175
  1 - max softmax prob                    0.0970

============================================================
  4d. Distribution Shift Detection
============================================================

  Clean MNIST (neg) vs Shifted MNIST (pos)
  Method                                   AUROC
  ---------------------------------------------
  Teacher EU (ensemble)                   0.9386
  Student EU (learned)                    0.9430
  Student entropy (softmax)               0.6179
  1 - max softmax prob                    0.6250

============================================================
  5. Uncertainty Decomposition (Clean MNIST test)
============================================================
  Metric       Teacher mean   Student mean    Pearson   Spearman
  ----------------------------------------------------------
  TU                 0.1463         0.1399     0.9672     0.9352
  AU                 0.1347         0.1225     0.9619     0.9169
  EU                 0.0116         0.0174     0.7634     0.7220

============================================================
  6. OOD Detection: MNIST vs OOD — Decomposed Uncertainties
============================================================
  Dataset            Type      Tea TU   Tea EU   Tea AU |   Stu TU   Stu EU   Stu AU
  -------------------------------------------------------------------------------------
  FashionMNIST       seen      0.9744   0.9966   0.9184 |   0.4689   0.9994   0.3226
  Omniglot           seen      0.9802   0.9997   0.9402 |   0.7621   1.0000   0.2301
  EMNIST-Letters     unseen    0.9497   0.9509   0.9479 |   0.9388   0.9638   0.9175
  CIFAR-10           unseen    0.9838   1.0000   0.8910 |   0.0047   1.0000   0.0013
  SVHN               unseen    0.9911   1.0000   0.9119 |   0.0138   1.0000   0.0025

  Student TU = H[softmax(logits)], EU = EU head, AU = TU - EU
  Loaded single member (member_0, acc=99.16%)

============================================================
  7. Baseline: Single Ensemble Member — OOD Detection
============================================================
  Single member test accuracy: 99.16%

  Dataset            Type       Entropy  1-MaxProb
  --------------------------------------------------
  FashionMNIST       seen        0.5716     0.5817
  Omniglot           seen        0.4668     0.4903
  EMNIST-Letters     unseen      0.9086     0.9107
  CIFAR-10           unseen      0.0317     0.0337
  SVHN               unseen      0.1094     0.1152
  Shifted MNIST      shift       0.7094     0.7141

  ============================================================
  OOD Detection: MNIST vs OOD — Epistemic Uncertainty
============================================================

Dataset            Type        Tea EU    Stu EU    SingleModel
---------------------------------------------------------------
FashionMNIST       seen        0.9966    0.9994    0.5716
Omniglot           seen        0.9997    1.0000    0.4668
EMNIST-Letters     unseen      0.9509    0.9638    0.9086
CIFAR-10           unseen      1.0000    1.0000    0.0317
SVHN               unseen      1.0000    1.0000    0.1094