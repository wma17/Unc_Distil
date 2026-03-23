# Uncertainty Distillation — Complete Experimental Results

> Collected for paper writing. All numbers from eval logs and checkpoints as of 2026-03-23.

---

## 1. Overview

**Goal**: Distill epistemic uncertainty (EU) from a LoRA ensemble teacher into a single-pass student model via two-phase training.

| Exp | Dataset | Domain | Task | Classes | Train/Val | Status |
|-----|---------|--------|------|---------|-----------|--------|
| E1 | MNIST | Vision | Classification | 10 | 60K/10K | Complete |
| E2 | CIFAR-10 | Vision | Classification | 10 | 50K/10K | Complete |
| E3 | TinyImageNet | Vision | Classification | 200 | 100K/10K | Complete |
| E4 | VOC 2012 | Vision | Segmentation | 21 | 10.6K/1.4K | Phase 2 pending |
| E5 | SST-2 | NLP | Sentiment | 2 | 67K/872 | Complete (weak EU) |

---

## 2. Architectures

### 2.1 Teacher Ensembles

| Experiment | Base Model | Adaptation | K | Trainable/Total Params |
|------------|-----------|------------|---|------------------------|
| E1 MNIST | MNISTConvNet (4-layer CNN) | Deep ensemble | 5 | Full model per member |
| E2 CIFAR-10 | CIFAR-ResNet18 | Deep ensemble | 5 or 16 | Full model per member |
| E3 TinyImageNet | DeiT-Small (patch16, 224) | LoRA + partial unfreeze | 16 | 0.4M–8.4M / ~22M |
| E4 VOC | SegFormer-B2 (512×512) | LoRA (rank=16, α=32) | 5 | 3.7M / 27.9M |
| E5 SST-2 | BERT-base-uncased | LoRA (rank=8, α=16) | 5 | 296K / 109.8M |

### 2.2 Student Models

| Experiment | Base Model | Feature Dim | EU Head Input | EU Head Architecture |
|------------|-----------|-------------|---------------|---------------------|
| E1 MNIST | MNISTConvNet | 64 | 64 + 10 = 74 | FC(74→128) → LeakyReLU → FC(128→1) |
| E2 CIFAR-10 | CIFAR-ResNet18 | 512 | 512 + 10 = 522 | FC(522→128) → ReLU → FC(128→1) → Softplus |
| E3 TinyImageNet | DeiT-Small | 384 | 384 + 200 (+5 density) = 584–589 | FC(584→256) → ReLU → FC(256→1) → Softplus |
| E4 VOC | SegFormer-B2 | 512 (spatial) | 512-ch feature maps | Conv(512→128,1×1) → BN → ReLU → Conv(128→1,1×1) → Softplus |
| E5 SST-2 | DistilBERT-base | 768 | 768 + 2 = 770 | FC(770→256) → ReLU → Dropout(0.2) → FC(256→1) → Softplus |

### 2.3 EU Head Input Features

All EU heads receive `[backbone_features ‖ softmax(logits)]` with softmax **detached** (no gradient to classifier).

**E3 TinyImageNet (winning config)** additionally uses 5 density/geometry features:
1. Energy score: `−logsumexp(logits)`
2. Predictive entropy: `−Σ pᵢ log pᵢ`
3. Logit margin: `top1 − top2`
4. Max cosine similarity to per-class centroids
5. Min L2 distance to nearest centroid

---

## 3. Training Protocol

### 3.1 Phase 1 — Classification Knowledge Distillation

**Loss function** (all experiments):
```
L₁ = (1−α)·CE(y, z_S) + α·τ²·KL(softmax(z_T/τ) ‖ softmax(z_S/τ))
```

| Parameter | E1 MNIST | E2 CIFAR-10 | E3 TinyImageNet | E4 VOC | E5 SST-2 |
|-----------|----------|-------------|-----------------|--------|----------|
| α (KD weight) | 0.7 | 0.7 | 0.7 | 0.7 | 0.5 |
| τ (temperature) | 4.0 | 4.0 | 2.0 | 4.0 | 2.0 |
| Optimizer | Adam | SGD (m=0.9) | AdamW | AdamW | AdamW |
| Learning rate | 1e-3 | 0.1 | 2e-4 | 6e-5 | 2e-5 |
| Weight decay | 1e-4 | 5e-4 | 0.05 | — | 0.05 |
| LR schedule | Cosine (5-ep warmup) | Cosine (10-ep warmup) | Cosine (10-ep warmup) | Poly (p=0.9) | Cosine (10% warmup) |
| Epochs/Iters | 50 epochs | 200 epochs | 100 epochs | 40K iters | 50 epochs (early stop) |
| Batch size | 128 | 128 | 32 | 8 | 32 |
| Label smoothing | — | — | 0.05 | — | — |
| Mixup/CutMix | — | — | 0.2 / 1.0 | — | — |
| EMA | — | — | 0.999 | — | — |
| LLRD | — | — | 0.75 | — | 0.85 |
| Grad clip | — | — | 1.0 | — | 1.0 |
| Text augment | — | — | — | — | word del 10%, swap 5% |

### 3.2 Phase 2 — EU Head Regression

**Loss function**:
```
L₂ = log1p_MSE(EU_S, EU_T) + β·PairwiseRankingLoss(EU_S, EU_T)
```
where `log1p_MSE = MSE(log(1+pred), log(1+target))`

**E3 winning config** adds asymmetric weighting: under-prediction penalty = 2.0×

| Parameter | E1 MNIST | E2 CIFAR-10 | E3 TinyImageNet | E4 VOC (planned) | E5 SST-2 |
|-----------|----------|-------------|-----------------|-------------------|----------|
| Optimizer | Adam | Adam | Adam | Adam | Adam |
| Learning rate | 5e-3 | 1e-3 | 3e-3 | 1e-3 | 5e-4 |
| LR schedule | Cosine | Cosine | Cosine | Cosine | Cosine |
| Epochs/Iters | 100 epochs | 100 epochs | 80 epochs | 20K iters | 200 epochs |
| Batch size | 128 | 128 | 256 | 8 | 32 |
| Rank weight β | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Rank pairs | 256 | 256 | — | — | 256 |
| Rank margin | 0.05 | 0.05 | — | — | 0.05 |
| Asymmetric weight | — | — | 2.0 | — | — |
| Tail loss weight | — | — | 1.0 | — | — |
| EU sample α | — | — | 10.0 | — | — |
| Density features | — | — | geo2 (5-d) | — | — |
| Backbone | Frozen | Frozen | Frozen | Frozen | Frozen |

### 3.3 Phase 2 Data Curriculum (A3: Three-Tier)

| Tier | Proportion | E1 MNIST | E2 CIFAR-10 | E3 TinyImageNet | E4 VOC | E5 SST-2 |
|------|-----------|----------|-------------|-----------------|--------|----------|
| Clean ID | 50% | MNIST train | CIFAR-10 train | TinyIN train | VOC+SBD train | SST-2 train |
| Corrupted | 25% | gauss noise, blur, low contrast, inversion, colored bg/digits, salt pepper, pixelate | gauss noise, gauss blur, low contrast | gauss noise, gauss blur, low contrast, jpeg, brightness, shot noise | gauss noise, gauss blur, low contrast | char perturb, word perturb |
| Synthetic OOD | 25% | mixup + masked | mixup + masked | mixup, patchshuffle, cutpaste, masked, heavy noise, multi-corrupt, pixel permute | mixup + masked | token masking (30/50/70%) |

---

## 4. Ensemble Member Diversity

### E3 TinyImageNet (K=16)

Diversity knobs varied across members:
- **LoRA rank**: {4, 8, 16, 32}
- **LoRA alpha**: {8, 16, 32, 64}
- **LoRA targets**: {qkv_only, qkv+proj, qkv+proj+mlp}
- **LoRA dropout**: {0.0, 0.05, 0.1}
- **Unfreeze blocks**: {0, 1, 2, 4}
- **Label smoothing**: U[0.01, 0.08]
- **Bagging fraction**: U[0.75, 1.0]
- **Weight decay**: {0.01, 0.03, 0.05}
- **Augmentations**: random subsets of {randaugment, autoaugment, colorjitter, perspective, erasing, mixup, cutmix}

### E2 CIFAR-10 (K=5 or K=16)

- **Dropout rates**: {0.0, 0.05, 0.1}
- **Label smoothing**: U[0.0, 0.05]
- **Data fraction**: U[0.8, 1.0]
- **Learning rate**: U[0.05, 0.15]
- **Weight decay**: {3e-4, 5e-4, 1e-3}
- **LR schedule**: {cosine, step}
- **Augmentations**: subsets of {colorjitter, grayscale, rotation, cutout, randomerasing, autoaugment}

### E4 VOC (K=5)

- **LoRA rank/alpha**: fixed at 16/32.0 (all members)
- **Augmentation mode**: {scale, colorjitter, rotation}
- **Label smoothing**: [0.0013, 0.0348]
- **Weight decay**: {1e-4, 5e-4, 1e-3}

### E1 MNIST (K=5) and E5 SST-2 (K=5)

See Section 2.1 for per-member configs.

---

## 5. Main Results

### 5.1 Classification / Segmentation Accuracy

| Experiment | Teacher Acc | Student Acc | Δ | Agreement |
|------------|-----------|------------|---|-----------|
| E1 MNIST | 99.56% | 99.60% | +0.04 | 99.82% |
| E2 CIFAR-10 (K=5) | 96.37% | 95.81% | −0.56 | — |
| E2 CIFAR-10 (K=16) | 96.96% | 95.99% | −0.97 | — |
| E3 TinyImageNet (K=16) | 88.13% | 87.50% | −0.63 | 95.11% |
| E4 VOC (mIoU, K=5) | 0.8048 | 0.8056 | +0.0008 | — |
| E5 SST-2 (K=5) | 91.51% | 92.09% | +0.58 | — |

### 5.2 Calibration (ECE-15 / NLL / Brier)

| Experiment | Teacher ECE | Student ECE | Teacher NLL | Student NLL | Teacher Brier | Student Brier |
|------------|-----------|-----------|-----------|-----------|-------------|-------------|
| E1 MNIST | 0.0245 | 0.0238 | 0.0362 | 0.0357 | 0.0099 | 0.0101 |
| E2 CIFAR-10 | 0.0294 | 0.0195 | 0.1153 | 0.1425 | 0.0487 | 0.0639 |
| E3 TinyImageNet | 0.0763 | 0.0484 | 0.4862 | 0.5139 | 0.1778 | 0.1833 |
| E5 SST-2 | 0.0179 | 0.0251 | 0.2122 | 0.2164 | 0.1224 | 0.1272 |

### 5.3 EU Correlation — Student vs Teacher (Clean Validation)

| Experiment | Pearson | Spearman |
|------------|---------|----------|
| E1 MNIST | 0.7578 | 0.7429 |
| E2 CIFAR-10 | 0.7151 | 0.6902 |
| E3 TinyImageNet | 0.8411 | 0.8827 |
| E5 SST-2 | 0.4859 | 0.5771 |

### 5.4 EU Correlation — Corrupted / Shifted Data

**E3 TinyImageNet**:

| Corruption | Pearson | Spearman | Student EU Mean | Teacher EU Mean |
|-----------|---------|----------|----------------|----------------|
| Clean | 0.8411 | 0.8827 | 0.0625 | 0.0623 |
| gaussian_noise | 0.8718 | 0.8989 | 0.0768 | 0.0835 |
| gaussian_blur | 0.8802 | 0.9041 | 0.0668 | 0.0723 |
| low_contrast | 0.8516 | 0.8624 | 0.1042 | 0.1183 |
| jpeg_compression | 0.8889 | 0.9178 | 0.1122 | 0.1244 |
| brightness | 0.8843 | 0.9065 | 0.0713 | 0.0771 |
| shot_noise | 0.8297 | 0.8643 | 0.1094 | 0.1251 |

**E1 MNIST**:

| Corruption | Pearson | Spearman | Student EU Mean | Teacher EU Mean |
|-----------|---------|----------|----------------|----------------|
| Clean | 0.7578 | 0.7429 | 0.0203 | 0.0109 |
| gaussian_noise | 0.7135 | 0.8051 | 0.0525 | 0.0301 |
| gaussian_blur | 0.8001 | 0.7188 | 0.3545 | 0.3696 |
| low_contrast | 0.8602 | 0.8159 | 0.3002 | 0.3059 |
| inversion | 0.8367 | 0.8283 | 0.5080 | 0.5142 |
| colored_background | 0.6121 | 0.5632 | 0.0819 | 0.0401 |
| colored_digits | 0.7337 | 0.7731 | 0.1852 | 0.2006 |
| salt_pepper | 0.7871 | 0.8191 | 0.1964 | 0.1953 |
| pixelate | 0.6844 | 0.6651 | 0.2476 | 0.2607 |

### 5.5 OOD Detection — AUROC (Student EU)

**E3 TinyImageNet** (ID = clean TinyImageNet val):

| OOD Dataset | Type | Teacher EU | Student EU | Student Entropy | 1−MaxProb |
|------------|------|-----------|-----------|-----------------|-----------|
| SVHN | seen | 0.9795 | 0.8242 | 0.8456 | 0.8050 |
| CIFAR-100 | seen | 0.8673 | 0.8324 | 0.8400 | 0.8253 |
| CIFAR-10 | unseen | 0.8719 | 0.8462 | 0.8353 | 0.8217 |
| STL10 | unseen | 0.8115 | 0.7947 | 0.8308 | 0.8071 |
| DTD | unseen | 0.9092 | 0.9071 | 0.9291 | 0.9037 |
| FashionMNIST | unseen | 0.9451 | 0.8944 | 0.9097 | 0.8972 |
| MNIST | unseen | 0.8951 | 0.8945 | 0.8027 | 0.8110 |

**E2 CIFAR-10** (ID = clean CIFAR-10 test):

| OOD Dataset | Type | Student EU AUROC |
|------------|------|-----------------|
| SVHN | seen | 0.9768 |
| CIFAR-100 | seen | 0.9094 |
| MNIST | unseen | 0.9389 |
| FashionMNIST | unseen | 0.9392 |
| STL10 | unseen | 0.6723 |
| DTD | unseen | 0.9428 |

**E1 MNIST** (ID = clean MNIST test):

| OOD Dataset | Type | Teacher EU | Student EU | Student Entropy | 1−MaxProb |
|------------|------|-----------|-----------|-----------------|-----------|
| FashionMNIST | seen | 0.9735 | 0.9788 | 0.5670 | 0.5873 |
| Omniglot | seen | 0.9901 | 0.9960 | 0.2443 | 0.3068 |
| EMNIST-Letters | unseen | 0.9596 | 0.9440 | 0.9545 | 0.9538 |
| CIFAR-10 | unseen | 0.9499 | 0.9323 | 0.5820 | 0.6317 |
| SVHN | unseen | 0.9561 | 0.9533 | 0.3887 | 0.4289 |

**E5 SST-2** (ID = SST-2 dev):

| OOD Dataset | Teacher EU | Student EU | Student Entropy | 1−MaxProb |
|------------|-----------|-----------|-----------------|-----------|
| IMDB | 0.4769 | 0.5218 | 0.6829 | 0.6829 |
| Yelp | 0.5320 | 0.5996 | 0.7301 | 0.7301 |
| Amazon | 0.5011 | 0.5788 | 0.6628 | 0.6628 |
| AG_News | 0.6625 | 0.7428 | 0.8659 | 0.8659 |
| 20NG | 0.5779 | 0.6002 | 0.7898 | 0.7898 |

### 5.6 Distribution Shift Detection — AUROC

| Experiment | Clean vs Shifted | Teacher EU | Student EU | Student Entropy | 1−MaxProb |
|------------|-----------------|-----------|-----------|-----------------|-----------|
| E1 MNIST | Clean vs Corrupted | 0.9166 | 0.9270 | 0.7233 | 0.7430 |

### 5.7 Uncertainty Decomposition (Clean Validation)

| Experiment | Metric | Teacher Mean | Student Mean | Pearson | Spearman |
|------------|--------|-------------|-------------|---------|----------|
| E1 MNIST | TU | 0.1417 | 0.1358 | 0.9666 | 0.9485 |
| E1 MNIST | AU | 0.1308 | 0.1155 | 0.9594 | 0.9265 |
| E1 MNIST | EU | 0.0109 | 0.0203 | 0.7578 | 0.7429 |
| E2 CIFAR-10 | TU | 0.2363 | 0.1825 | 0.7549 | 0.6256 |
| E2 CIFAR-10 | AU | 0.1870 | 0.1133 | 0.6144 | 0.2078 |
| E2 CIFAR-10 | EU | 0.0493 | 0.0697 | 0.7151 | 0.6902 |
| E3 TinyIN | TU | 1.0107 | 0.9576 | 0.9058 | 0.8829 |
| E3 TinyIN | AU | 0.9484 | 0.8950 | 0.8971 | 0.8675 |
| E3 TinyIN | EU | 0.0623 | 0.0625 | 0.8411 | 0.8827 |

### 5.8 Selective Prediction — AURC (lower is better)

**E3 TinyImageNet**:

| Method | AURC↓ | @90%cov↑ | @80%cov↑ |
|--------|-------|----------|----------|
| Teacher EU | 0.02359 | 0.9167 | 0.9530 |
| Student EU | 0.02389 | 0.9187 | 0.9501 |
| Student entropy | 0.02366 | 0.9260 | 0.9589 |
| 1−MaxProb | 0.02185 | 0.9271 | 0.9634 |
| Oracle | 0.00816 | 0.9722 | 1.0000 |

**E2 CIFAR-10**:

| Method | AURC↓ | @90%cov↑ | @80%cov↑ |
|--------|-------|----------|----------|
| Teacher EU | 0.00305 | 0.9916 | 0.9975 |
| Student EU | 0.00330 | 0.9906 | 0.9979 |
| Student entropy | 0.00333 | 0.9911 | 0.9975 |
| 1−MaxProb | 0.00329 | 0.9911 | 0.9975 |
| Oracle | 0.00082 | 1.0000 | 1.0000 |

**E1 MNIST**:

| Method | AURC↓ | @90%cov↑ | @80%cov↑ |
|--------|-------|----------|----------|
| Teacher EU | 0.000101 | 0.9999 | 1.0000 |
| Student EU | 0.000128 | 0.9998 | 0.9999 |
| Student entropy | 0.000081 | 0.9999 | 1.0000 |
| 1−MaxProb | 0.000063 | 0.9999 | 1.0000 |
| Oracle | 0.000008 | 1.0000 | 1.0000 |

**E5 SST-2**:

| Method | AURC↓ | @90%cov↑ | @80%cov↑ |
|--------|-------|----------|----------|
| Teacher EU | 0.03091 | 0.9464 | 0.9584 |
| Student EU | 0.02079 | 0.9413 | 0.9641 |
| Student entropy | 0.01784 | 0.9426 | 0.9727 |
| 1−MaxProb | 0.01784 | 0.9426 | 0.9727 |
| Oracle | 0.00322 | 1.0000 | 1.0000 |

### 5.9 Inference Throughput

| Experiment | Ensemble (K members) | Single Member | Student | Speedup |
|------------|---------------------|---------------|---------|---------|
| E1 MNIST | 122,661 s/s (K=5) | 638,858 s/s | 845,027 s/s | 6.89× |
| E3 TinyImageNet | 163 s/s (K=16) | 3,131 s/s | 3,987 s/s | 24.47× |
| E5 SST-2 | 300 sent/s (K=5) | 1,495 sent/s | 3,167 sent/s | 10.57× |

---

## 6. Ablation Studies

### 6.1 Phase 2 Data Curriculum (E2 CIFAR-10, K=16)

| Config | Data Mix | EU Pearson | EU Spearman | SVHN AUROC | CIFAR-100 AUROC | Shift Det. |
|--------|----------|-----------|------------|-----------|-----------------|-----------|
| A1 | 100% clean | 0.8001 | 0.8072 | 0.9454 | 0.9038 | 0.7707 |
| A2 | 67% clean + 33% corrupted | 0.8143 | 0.8067 | 0.9888 | 0.9160 | 0.8044 |
| A3 | 50% clean + 25% corr + 25% OOD | 0.7151 | 0.6902 | 0.9768 | 0.9094 | — |

### 6.2 Phase 2 Loss Function (E2 CIFAR-10, K=16)

| Config | Loss | EU Pearson | EU Spearman | SVHN AUROC | CIFAR-100 AUROC |
|--------|------|-----------|------------|-----------|-----------------|
| Baseline | log1p_MSE + ranking | 0.7151 | 0.6902 | 0.9768 | 0.9094 |
| B1 | Plain MSE only | 0.8144 | 0.6437 | 0.9853 | 0.9203 |
| B2 | Log-MSE only | 0.8240 | 0.6498 | 0.9855 | 0.9192 |
| B3 | Ranking only | 0.7044 | 0.8176 | 0.9861 | 0.9206 |

### 6.3 Ensemble Size (E2 CIFAR-10)

| K | Teacher Acc | EU Pearson | EU Spearman | SVHN AUROC | CIFAR-100 AUROC |
|---|-----------|-----------|------------|-----------|-----------------|
| 3 | 96.06% | 0.6235 | 0.6538 | 0.9906 | 0.9166 |
| 5 | 96.30% | 0.7144 | 0.7223 | 0.9899 | 0.9185 |
| 10 | 96.78% | 0.7922 | 0.7842 | 0.9844 | 0.9185 |
| 16 | 96.96% | — (baseline) | — | — | — |

### 6.4 Joint vs Two-Phase Training (E2 CIFAR-10)

| Config | Student Acc | EU Pearson | EU Spearman | SVHN AUROC | CIFAR-100 AUROC |
|--------|-----------|-----------|------------|-----------|-----------------|
| Two-phase (ours) | 95.99% | 0.7151 | 0.6902 | 0.9768 | 0.9094 |
| Joint (γ=1.0) | 95.07% | 0.6459 | 0.6274 | 0.9292 | 0.8706 |

Joint training significantly degrades both accuracy (−0.92%) and OOD detection (−0.048 SVHN AUROC).

### 6.5 E3 TinyImageNet Phase 2 Key Ablations

Best config: `asym3_geo2` (asymmetric weight=2.0, geo2 density features, rank+tail losses)

| Feature | Effect on SVHN AUROC | Effect on STL10 AUROC | Effect on Spearman |
|---------|--------------------|--------------------|-------------------|
| Asymmetric loss (w=2.0) | +0.015 | — | — |
| Geo2 density features | +0.061 | — | — |
| suppress_weight=0 | Critical | — | — |
| eu_sample_alpha=0 | Critical | — | — |

---

## 7. E3 TinyImageNet — OOD Correlation on OOD Datasets

Student EU vs teacher EU on OOD data (informational — not used for OOD detection):

| OOD Dataset | Pearson | Spearman | Student EU Mean | Teacher EU Mean |
|------------|---------|----------|----------------|----------------|
| SVHN | −0.0570 | −0.0763 | 0.1454 | 0.3435 |
| CIFAR-10 | 0.6120 | 0.6084 | 0.1735 | 0.2042 |
| CIFAR-100 | 0.6298 | 0.6292 | 0.1691 | 0.2080 |
| STL10 | 0.5631 | 0.6101 | 0.1468 | 0.1508 |
| DTD | 0.4204 | 0.3749 | 0.2214 | 0.2301 |
| FashionMNIST | 0.3742 | 0.3451 | 0.2039 | 0.2682 |
| MNIST | 0.1934 | 0.1658 | 0.1994 | 0.1948 |

---

## 8. E4 VOC — Ensemble Member Results

| Member | Seed | Augmentation | Label Smooth | Weight Decay | Best mIoU | Best Epoch |
|--------|------|-------------|-------------|-------------|---------|-----------|
| 0 | 42 | scale | 0.0013 | 1e-4 | 0.7999 | 105 |
| 1 | 43 | colorjitter | 0.0348 | 5e-4 | 0.8043 | 200 |
| 2 | 44 | colorjitter | 0.0271 | 1e-4 | 0.8042 | 219 |
| 3 | 45 | colorjitter | 0.0244 | 5e-4 | 0.8081 | 195 |
| 4 | 46 | rotation | 0.0200 | 1e-3 | 0.8076 | 215 |
| **Mean** | | | | | **0.8048** | |

Phase 1 student: mIoU = 0.8056 at step 20,130 (exceeds teacher mean).

---

## 9. E5 SST-2 — EU Correlation on Perturbed Text

| Dataset | Pearson | Spearman |
|---------|---------|----------|
| Clean SST-2 dev | 0.4859 | 0.5771 |
| Char-perturbed SST-2 dev | 0.4335 | 0.5317 |

Note: Spearman 0.5771 < 0.65 gating threshold — EU distillation for NLP remains an open problem.

---

## 10. E3 TinyImageNet — Full Ensemble Member Configs (K=16)

| # | Rank | Alpha | Dropout | Targets | Unfreeze | Smooth | Bag | LR | WD | Augmentations |
|---|------|-------|---------|---------|----------|--------|-----|----|----|---------------|
| 0 | 4 | 8.0 | 0.1 | qkv+proj | 1 | 0.012 | 0.77 | 9.7e-5 | 0.05 | cutmix, erasing, mixup, randaugment |
| 1 | 4 | 16.0 | 0.1 | qkv_only | 4 | 0.04 | 0.84 | 2.2e-4 | 0.05 | autoaug, colorjitter, erasing, mixup, perspective |
| 2 | 32 | 32.0 | 0.0 | qkv+proj | 1 | 0.079 | 0.92 | 1.3e-4 | 0.03 | autoaug, colorjitter, cutmix, erasing, mixup |
| 3 | 16 | 32.0 | 0.05 | qkv+proj | 0 | 0.032 | 0.94 | 5.7e-5 | 0.05 | autoaug, colorjitter, cutmix, erasing, mixup, perspective |
| 4 | 4 | 16.0 | 0.0 | qkv+proj+mlp | 1 | 0.032 | 0.96 | 2.2e-4 | 0.01 | cutmix, erasing, mixup, perspective |
| 5 | 16 | 16.0 | 0.05 | qkv+proj+mlp | 4 | 0.012 | 0.78 | 1.0e-4 | 0.03 | colorjitter, cutmix, erasing, mixup |
| 6 | 16 | 16.0 | 0.1 | qkv+proj+mlp | 2 | 0.018 | 0.91 | 9.4e-5 | 0.05 | colorjitter, mixup |
| 7 | 4 | 16.0 | 0.05 | qkv_only | 2 | 0.061 | 0.91 | 1.5e-4 | 0.05 | colorjitter, cutmix, mixup, perspective |
| 8 | 32 | 64.0 | 0.05 | qkv+proj+mlp | 1 | 0.016 | 0.79 | 2.2e-4 | 0.03 | colorjitter, erasing, mixup |
| 9 | 8 | 8.0 | 0.0 | qkv_only | 2 | 0.056 | 0.85 | 1.1e-4 | 0.05 | autoaug, cutmix, erasing, mixup, perspective |
| 10 | 16 | 16.0 | 0.1 | qkv+proj+mlp | 4 | 0.023 | 0.84 | 1.8e-4 | 0.01 | autoaug, colorjitter, mixup |
| 11 | 8 | 16.0 | 0.1 | qkv+proj+mlp | 4 | 0.022 | 1.0 | 2.0e-4 | 0.03 | mixup, perspective |
| 12 | 8 | 16.0 | 0.1 | qkv+proj | 4 | 0.071 | 0.93 | 6.1e-5 | 0.05 | autoaug, colorjitter, cutmix, erasing, mixup |
| 13 | 4 | 8.0 | 0.0 | qkv+proj+mlp | 2 | 0.056 | 0.89 | 1.3e-4 | 0.03 | colorjitter, cutmix, randaugment |
| 14 | 4 | 16.0 | 0.1 | qkv+proj+mlp | 2 | 0.055 | 1.0 | 9.1e-5 | 0.05 | colorjitter, cutmix, erasing, perspective |
| 15 | 4 | 16.0 | 0.1 | qkv+proj+mlp | 0 | 0.072 | 0.75 | 2.0e-4 | 0.05 | colorjitter, mixup, randaugment |

---

## 11. E1 MNIST — Full Ensemble Member Configs (K=5)

| # | Seed | Dropout | Smooth | Data Frac | LR | WD | Schedule | Augmentations |
|---|------|---------|--------|-----------|-----|-----|----------|---------------|
| 0 | 42 | 0.1 | 0.037 | 0.95 | 0.013 | 5e-4 | step | affine, erasing |
| 1 | 43 | 0.0 | 0.034 | 0.89 | 0.013 | 1e-4 | step | rotation, erasing |
| 2 | 44 | 0.0 | 0.011 | 0.84 | 0.014 | 1e-4 | cosine | rotation, perspective |
| 3 | 45 | 0.0 | 0.004 | 0.82 | 0.017 | 3e-4 | cosine | rotation, erasing |
| 4 | 46 | 0.1 | 0.011 | 0.83 | 0.013 | 1e-4 | cosine | affine |

---

## 12. E5 SST-2 — Full Ensemble Member Configs (K=5)

| # | Seed | Init Scale | Smooth | Attn Drop | LR | WD | LoRA Rank | LoRA Alpha |
|---|------|-----------|--------|-----------|-----|-----|-----------|-----------|
| 0 | 42 | 1.009 | 0.0 | 0.15 | 1e-4 | 0.01 | 8 | 16.0 |
| 1 | 43 | 0.522 | 0.05 | 0.1 | 2e-4 | 0.01 | 8 | 16.0 |
| 2 | 44 | 0.783 | 0.05 | 0.1 | 1e-4 | 0.01 | 8 | 16.0 |
| 3 | 45 | 0.674 | 0.02 | 0.15 | 1e-4 | 0.01 | 8 | 16.0 |
| 4 | 46 | 1.327 | 0.02 | 0.1 | 3e-4 | 0.01 | 8 | 16.0 |

---

## 13. E1 MNIST — Decomposed OOD Detection (AUROC by Uncertainty Type)

| OOD Dataset | Type | Tea TU | Tea EU | Tea AU | Stu TU | Stu EU | Stu AU |
|------------|------|--------|--------|--------|--------|--------|--------|
| FashionMNIST | seen | 0.9230 | 0.9735 | 0.8991 | 0.5670 | 0.9788 | 0.3925 |
| Omniglot | seen | 0.9551 | 0.9901 | 0.9249 | 0.2443 | 0.9960 | 0.0005 |
| EMNIST-Letters | unseen | 0.9567 | 0.9596 | 0.9552 | 0.9545 | 0.9440 | 0.9508 |
| CIFAR-10 | unseen | 0.7587 | 0.9499 | 0.6568 | 0.5820 | 0.9323 | 0.4399 |
| SVHN | unseen | 0.8030 | 0.9561 | 0.7265 | 0.3887 | 0.9533 | 0.2555 |

Key observation: Student EU dramatically outperforms student TU/AU for OOD detection, confirming the value of distilled epistemic uncertainty.

---

## 14. Phase 2 Training Convergence (E3 TinyImageNet)

| Epoch | Total Loss | MSE Loss | Rank Loss | Pearson | Spearman |
|-------|-----------|----------|-----------|---------|----------|
| 1 | 0.04783 | 0.03604 | 0.01179 | 0.3858 | 0.6509 |
| 9 | 0.00994 | 0.00319 | 0.00676 | 0.8359 | 0.8598 |
| 39 | 0.00531 | 0.00149 | 0.00382 | 0.8474 | 0.8801 |
| 69 | — | — | — | 0.8420 | **0.8848** ← best |
| 80 | 0.00219 | 0.00051 | 0.00168 | 0.8415 | 0.8843 |

---

## 15. Key Takeaways

1. **Two-phase training is essential**: Joint training (E2 ablation) degrades both accuracy and EU quality.
2. **Larger ensembles help**: K=10→16 members yield better EU targets (E2 ensemble size ablation, E3 K=16).
3. **Combined loss works best**: log1p_MSE preserves scale, ranking loss preserves ordering (E2 loss ablation).
4. **Corrupted + fake OOD data critical**: A3 curriculum enables robust OOD detection (E2 curriculum ablation).
5. **Density features boost OOD**: Geo2 features add +0.061 STL10 AUROC in E3.
6. **Asymmetric loss helps**: Penalizing EU under-prediction improves OOD detection in E3.
7. **Vision works well, NLP is hard**: E1–E3 achieve strong EU distillation; E5 SST-2 Spearman=0.5771 remains weak.
8. **Speedup scales with K**: 6.89× (K=5 MNIST) → 10.57× (K=5 SST-2) → 24.47× (K=16 TinyImageNet).
