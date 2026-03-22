# Uncertainty Distillation — Improvement Plan

**Date:** 2026-03-21
**Goal:** Fix E3 (TinyImageNet) OOD detection degradation and E5 (SST-2) weak training.
**Executor:** Sonnet model — follow each step precisely, verify outputs, and report results.

---

## Table of Contents

1. [Diagnosis](#diagnosis)
2. [Part A: E5 (SST-2) Fixes](#part-a-e5-sst-2-fixes) — do FIRST (faster, ~30 min)
3. [Part B: E3 (TinyImageNet) Fixes](#part-b-e3-tinyimagenet-fixes) — do SECOND (~2-3 hours)
4. [Monitoring Checklist](#monitoring-checklist)

---

## Diagnosis

### E3 (TinyImageNet): Student EU dynamic range is compressed

The student over-predicts EU on in-distribution (ID) data and under-predicts on OOD:

| Data | Teacher EU mean | Student EU mean | Problem |
|---|---|---|---|
| Clean ID | 0.0623 | 0.0841 | Student 35% too high |
| SVHN (OOD) | 0.3435 | 0.2132 | Student 38% too low |

Teacher ID-to-OOD ratio: 5.5x. Student: 2.5x. This compressed range kills AUROC.

**Root causes:**
- **Phase 1 (backbone):** Only 2 transformer blocks unfrozen in 50 epochs. The backbone hasn't learned discriminative enough features, so many ID samples get "confused" CLS representations that the EU head interprets as high uncertainty. The user suspects this is the primary cause.
- **Phase 2 (EU head):** The loss treats over-prediction and under-prediction symmetrically. The `asymmetric_weight=2.0` actually penalizes *under*-prediction more, which is exactly wrong for clean ID samples where *over*-prediction is the problem. Additionally, fake OOD EU targets max at ~0.13 mean, far below real OOD (0.34 mean), so the head never learns the full EU range.

### E5 (SST-2): Catastrophic Phase 1 overfitting

Training log shows:
```
Epoch  1: Train 88.88% | Dev 89.68%   <-- BEST (barely trained!)
Epoch  2: Train 93.13% | Dev 87.16%   <-- already dropped 2.5pp
Epoch  6: Train 95.04% | Dev 83.26%   <-- 6.4pp below best
Epoch 20: Train 98.11% | Dev 84.63%   <-- massive overfit
```

Best checkpoint = epoch 1 = backbone barely moved from pretrained DistilBERT.

**Root causes:**
- `lr=2e-4` is 10x too high for DistilBERT fine-tuning (standard: 2e-5)
- `tau=6.0` with 2 classes flattens teacher distribution to near-uniform, destroying KL signal
- No LLRD, insufficient weight decay, no data augmentation
- Phase 2 then stalls at Spearman=0.5146 because frozen features are from a barely-trained backbone

---

## Part A: E5 (SST-2) Fixes

**Run this first — it's faster and validates the approach.**

### A1. Modify `SST2/models.py` — Wider EU head + dropout

**File:** `/home/maw6/maw6/unc_regression/SST2/models.py`

In the `DistilBERTStudent` class, make these changes:

1. Change `__init__` signature default: `eu_hidden: int = 256` (was 128)

2. Add dropout layer. Change the `__init__` body from:
```python
        eu_in = self.HIDDEN_DIM + num_classes
        self.eu_fc1 = nn.Linear(eu_in, eu_hidden)
        self.eu_fc2 = nn.Linear(eu_hidden, 1)
        self.eu_act = nn.Softplus()
```
to:
```python
        eu_in = self.HIDDEN_DIM + num_classes
        self.eu_fc1 = nn.Linear(eu_in, eu_hidden)
        self.eu_drop = nn.Dropout(0.2)
        self.eu_fc2 = nn.Linear(eu_hidden, 1)
        self.eu_act = nn.Softplus()
```

3. Change `forward` EU computation from:
```python
        eu = self.eu_act(self.eu_fc2(F.relu(self.eu_fc1(eu_in)))).squeeze(-1)
```
to:
```python
        eu = F.relu(self.eu_fc1(eu_in))
        eu = self.eu_drop(eu)
        eu = self.eu_act(self.eu_fc2(eu)).squeeze(-1)
```

4. Change `create_student` default: `eu_hidden=256` (was 128)

### A2. Modify `SST2/distill.py` — Fix Phase 1 + improve Phase 2

**File:** `/home/maw6/maw6/unc_regression/SST2/distill.py`

#### A2a. Add LLRD function

Add this function right after the `spearman_corr` function (before the `# Phase 1` section):

```python
def _build_llrd_param_groups_distilbert(model, base_lr, wd, llrd_factor):
    """Layer-wise learning rate decay for DistilBERT (6 transformer layers)."""
    param_groups = []
    seen = set()

    # Classifier: full lr
    for p in model.classifier.parameters():
        if p.requires_grad:
            param_groups.append({"params": [p], "lr": base_lr, "weight_decay": wd})
            seen.add(id(p))

    # Transformer layers: LLRD from top (layer 5) to bottom (layer 0)
    n_layers = len(model.distilbert.transformer.layer)
    for i in range(n_layers - 1, -1, -1):
        depth = (n_layers - 1) - i
        layer_lr = base_lr * (llrd_factor ** depth)
        layer_params = [p for p in model.distilbert.transformer.layer[i].parameters()
                        if p.requires_grad and id(p) not in seen]
        if layer_params:
            param_groups.append({
                "params": layer_params, "lr": layer_lr, "weight_decay": wd
            })
            for p in layer_params:
                seen.add(id(p))

    # Embeddings: deepest lr
    embed_lr = base_lr * (llrd_factor ** n_layers)
    embed_params = [p for p in model.distilbert.embeddings.parameters()
                    if p.requires_grad and id(p) not in seen]
    if embed_params:
        param_groups.append({
            "params": embed_params, "lr": embed_lr, "weight_decay": wd
        })

    return param_groups
```

#### A2b. Add text augmentation to DistillTextDataset

Change the `DistillTextDataset` class. Add an `augment` parameter and augmentation method:

```python
class DistillTextDataset:
    """SST-2 paired with teacher soft labels and EU."""

    def __init__(self, texts, labels, teacher_probs, teacher_eu, tokenizer,
                 max_len=MAX_SEQ_LEN, augment=False):
        self.texts = texts
        self.labels = labels
        self.teacher_probs = torch.from_numpy(teacher_probs).float()
        self.teacher_eu = torch.from_numpy(teacher_eu).float()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

    def _augment_text(self, text):
        """Simple augmentation: random word deletion (10%) + adjacent swap (5%)."""
        import random
        words = text.split()
        if len(words) < 4:
            return text
        # Random deletion (keep each word with 90% probability)
        words = [w for w in words if random.random() > 0.1]
        if not words:
            return text
        # Random adjacent swap
        for i in range(len(words) - 1):
            if random.random() < 0.05:
                words[i], words[i + 1] = words[i + 1], words[i]
        return ' '.join(words)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.augment:
            text = self._augment_text(text)
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "teacher_probs": self.teacher_probs[idx],
            "teacher_eu": self.teacher_eu[idx],
        }
```

#### A2c. Modify argument defaults

Change these argument defaults in `main()`:

| Argument | Old default | New default |
|---|---|---|
| `--p1_epochs` | 20 | **50** |
| `--p1_lr` | 2e-4 | **2e-5** |
| `--tau` | 6.0 | **2.0** |
| `--p2_epochs` | 100 | **200** |
| `--p2_lr` | 0.001 | **5e-4** |

Add these NEW arguments to `main()`:

```python
    parser.add_argument("--p1_wd", type=float, default=0.05,
                        help="Weight decay for Phase 1 optimizer")
    parser.add_argument("--llrd_factor", type=float, default=0.85,
                        help="Layer-wise learning rate decay factor for DistilBERT")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience for Phase 1")
```

#### A2d. Modify Phase 1 training setup in `main()`

1. When creating the training dataset, pass `augment=True`:

Change:
```python
        p1_train_ds = DistillTextDataset(
            train_texts, raw_ds["train"]["label"],
            data["train_probs"], data["train_eu"], tokenizer)
```
to:
```python
        p1_train_ds = DistillTextDataset(
            train_texts, raw_ds["train"]["label"],
            data["train_probs"], data["train_eu"], tokenizer, augment=True)
```

2. Replace the Phase 1 optimizer setup. Change:
```python
        optimizer = optim.AdamW(model.parameters(), lr=args.p1_lr, weight_decay=0.01)
```
to:
```python
        param_groups = _build_llrd_param_groups_distilbert(
            model, args.p1_lr, args.p1_wd, args.llrd_factor)
        optimizer = optim.AdamW(param_groups)
```

3. Add early stopping to the Phase 1 training loop. Change the loop from:
```python
        best_acc = 0.0

        for epoch in range(1, args.p1_epochs + 1):
            t0 = time.time()
            tr_ce, tr_kl, tr_acc = train_phase1_epoch(
                model, p1_train_loader, optimizer, scheduler, args.alpha, args.tau, device)
            te_ce, te_kl, te_acc = eval_phase1(
                model, p1_dev_loader, args.alpha, args.tau, device)
            elapsed = time.time() - t0

            print(f"  P1 Epoch {epoch:2d}/{args.p1_epochs} | "
                  f"Train {tr_acc:.2f}% (ce={tr_ce:.4f} kl={tr_kl:.4f}) | "
                  f"Dev {te_acc:.2f}% (ce={te_ce:.4f} kl={te_kl:.4f}) | {elapsed:.1f}s")

            if te_acc > best_acc:
                best_acc = te_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch, "dev_acc": te_acc, "phase": 1,
                }, p1_path)

        print(f"\n  Phase 1 best accuracy: {best_acc:.2f}%  ->  {p1_path}")
```
to:
```python
        best_acc = 0.0
        no_improve = 0

        for epoch in range(1, args.p1_epochs + 1):
            t0 = time.time()
            tr_ce, tr_kl, tr_acc = train_phase1_epoch(
                model, p1_train_loader, optimizer, scheduler, args.alpha, args.tau, device)
            te_ce, te_kl, te_acc = eval_phase1(
                model, p1_dev_loader, args.alpha, args.tau, device)
            elapsed = time.time() - t0

            print(f"  P1 Epoch {epoch:2d}/{args.p1_epochs} | "
                  f"Train {tr_acc:.2f}% (ce={tr_ce:.4f} kl={tr_kl:.4f}) | "
                  f"Dev {te_acc:.2f}% (ce={te_ce:.4f} kl={te_kl:.4f}) | {elapsed:.1f}s")

            if te_acc > best_acc:
                best_acc = te_acc
                no_improve = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch, "dev_acc": te_acc, "phase": 1,
                }, p1_path)
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                    break

        print(f"\n  Phase 1 best accuracy: {best_acc:.2f}%  ->  {p1_path}")
```

### A3. Run E5 training

```bash
cd ~/maw6/unc_regression/SST2
mkdir -p ./checkpoints_v3

# Copy teacher targets (do NOT retrain teacher)
cp ./checkpoints/teacher_targets.npz ./checkpoints_v3/
cp ./checkpoints/member_*.pt ./checkpoints_v3/
cp ./checkpoints/ensemble_configs.json ./checkpoints_v3/

# Run full two-phase training
python distill.py --save_dir ./checkpoints_v3 \
  --p1_lr 2e-5 --tau 2.0 --p1_epochs 50 --p1_wd 0.05 \
  --llrd_factor 0.85 --patience 10 \
  --alpha 0.5 \
  --p2_lr 5e-4 --p2_epochs 200 --rank_weight 1.0 \
  --curriculum A3 --loss_mode combined \
  --gpu 0 2>&1 | tee retrain_v3.log
```

### A4. Monitor E5 Phase 1

**While training runs**, watch the log for these signals:

**GOOD signs (Phase 1 is working):**
- Dev accuracy steadily improves over epochs (not just epoch 1)
- Dev accuracy reaches >= 90% and stays there
- Train accuracy grows slower than before (no instant memorization)
- Dev CE and KL losses don't diverge from train losses

**BAD signs (still overfitting — adjust and rerun):**
- Dev accuracy peaks early (epoch 1-3) then drops → lr still too high, try `1e-5`
- Dev accuracy plateaus below 89% → lr too low or tau too low, try `3e-5` or `tau=3.0`
- Train accuracy reaches 98%+ while dev is below 88% → add more regularization

**Expected timeline:** Phase 1 ~50 epochs × 87s/epoch ≈ 73 min. Phase 2 ~200 epochs × 42s/epoch ≈ 140 min. Total ~3.5 hours.

### A5. Monitor E5 Phase 2

**GOOD signs:**
- Spearman improves past 0.55 and continues climbing
- Training MSE stays flat or decreases (not increasing like before)
- Best Spearman found after epoch 30+ (not peaking at epoch 20)

**BAD signs:**
- Spearman plateaus below 0.55 → backbone features still not good enough
- Training MSE increases over time → lr too high, try `2e-4`

### A6. Evaluate E5

```bash
cd ~/maw6/unc_regression/SST2
python evaluate_student.py --save_dir ./checkpoints_v3 --gpu 0 2>&1 | tee retrain_v3_eval.log
```

### A7. Success criteria for E5

| Metric | Old (v2) | Target (v3) | Notes |
|---|---|---|---|
| Dev accuracy | 89.68% | >= 90.5% | Close to teacher's 91.51% |
| EU Spearman (clean) | 0.5146 | >= 0.65 | Was capped by bad backbone |
| EU Pearson (clean) | 0.3655 | >= 0.50 | |
| AG_News AUROC (Stu EU) | 0.7692 | >= 0.75 | Maintain or improve |
| IMDB AUROC (Stu EU) | 0.4372 | >= 0.50 | Hardest near-OOD |
| AURC (Student EU) | 0.032 | <= 0.030 | Selective prediction |

---

## Part B: E3 (TinyImageNet) Fixes

### B1. Modify `TinyImageNet/models.py` — Wider EU head + dropout + learnable scale

**File:** `/home/maw6/maw6/unc_regression/TinyImageNet/models.py`

In the `DeiTStudent` class, make these changes:

1. Change `__init__`. Replace the EU head construction from:
```python
        eu_in = self.FEAT_DIM + num_classes
        self.eu_fc1 = nn.Linear(eu_in, eu_hidden)
        self.eu_fc2 = nn.Linear(eu_hidden, 1)
```
to:
```python
        eu_in = self.FEAT_DIM + num_classes
        self.eu_fc1 = nn.Linear(eu_in, 512)
        self.eu_drop = nn.Dropout(0.15)
        self.eu_fc2 = nn.Linear(512, 128)
        self.eu_fc3 = nn.Linear(128, 1)
        self.eu_scale = nn.Parameter(torch.ones(1))
```

Note: the `eu_hidden` constructor parameter is no longer used for sizing (sizes are fixed at 512/128). You can keep it in the signature for backward compat but it won't affect the layers.

2. Change `forward`. Replace the EU computation from:
```python
        eu_in = torch.cat([feat.detach(), probs], dim=-1)
        eu = self.eu_fc2(F.leaky_relu(self.eu_fc1(eu_in), 0.1)).squeeze(-1)
        return logits, eu
```
to:
```python
        eu_in = torch.cat([feat.detach(), probs], dim=-1)
        eu = F.leaky_relu(self.eu_fc1(eu_in), 0.1)
        eu = self.eu_drop(eu)
        eu = F.leaky_relu(self.eu_fc2(eu), 0.1)
        eu = (self.eu_scale * self.eu_fc3(eu)).squeeze(-1)
        return logits, eu
```

3. Change `eu_head_parameters` property to yield all EU params:
```python
    @property
    def eu_head_parameters(self):
        yield from self.eu_fc1.parameters()
        yield from self.eu_fc2.parameters()
        yield from self.eu_fc3.parameters()
        yield self.eu_scale
```

4. Change `reinit_eu_head` to reinit all layers:
```python
    def reinit_eu_head(self):
        nn.init.kaiming_normal_(self.eu_fc1.weight)
        nn.init.zeros_(self.eu_fc1.bias)
        nn.init.kaiming_normal_(self.eu_fc2.weight)
        nn.init.zeros_(self.eu_fc2.bias)
        nn.init.kaiming_normal_(self.eu_fc3.weight)
        nn.init.zeros_(self.eu_fc3.bias)
        self.eu_scale.data.fill_(1.0)
```

### B2. Modify `TinyImageNet/distill.py` — Phase 1 + Phase 2 improvements

**File:** `/home/maw6/maw6/unc_regression/TinyImageNet/distill.py`

#### B2a. Add new loss functions

Add these functions after the existing `pairwise_ranking_loss` function (around line 73):

```python
def id_suppression_loss(pred, target):
    """Penalize over-prediction of EU on clean ID samples.

    Only activates when pred > target, preventing the student from
    inflating EU on in-distribution data (the main E3 failure mode).
    """
    overshoot = F.relu(pred - target)
    return (overshoot ** 2).mean()


def tier_asymmetric_log1p_mse(pred, target, tier_labels):
    """Asymmetric MSE in log(1+x) space with tier-dependent weighting.

    tier_labels: 0=clean, 1=corrupted, 2=fake_ood
    - Clean: penalize over-prediction 3x (prevent ID EU inflation)
    - Fake OOD: penalize under-prediction 2x (preserve high EU)
    - Corrupted: symmetric (1x)
    """
    log_pred = torch.log1p(pred.clamp(min=0))
    log_tgt = torch.log1p(target)
    sq_err = (log_pred - log_tgt) ** 2
    over = pred > target
    weights = torch.ones_like(pred)
    weights[(tier_labels == 0) & over] = 3.0    # clean + over-predict
    weights[(tier_labels == 2) & (~over)] = 2.0  # OOD + under-predict
    return (weights * sq_err).mean()


def tier_margin_loss(pred, tier_labels, margin1=0.01, margin2=0.02):
    """Push EU ordering: clean < corrupted < fake_ood.

    Samples random pairs across tiers and enforces margins.
    """
    clean_eu = pred[tier_labels == 0]
    corrupt_eu = pred[tier_labels == 1]
    ood_eu = pred[tier_labels == 2]

    loss = pred.new_tensor(0.0)
    n = min(len(clean_eu), len(corrupt_eu))
    if n > 0:
        idx = torch.randperm(n, device=pred.device)
        loss = loss + F.relu(clean_eu[:n][idx] - corrupt_eu[:n][idx] + margin1).mean()
    n2 = min(len(corrupt_eu), len(ood_eu))
    if n2 > 0:
        idx2 = torch.randperm(n2, device=pred.device)
        loss = loss + F.relu(corrupt_eu[:n2][idx2] - ood_eu[:n2][idx2] + margin2).mean()
    return loss
```

#### B2b. Modify `_build_p2_loaders` to return tier labels

The function currently returns a list of `(name, loader, eu_array)` tuples. We need to also track which tier each source belongs to.

Change the function signature and add tier tracking. At the top of the function, after `loaders = []`:

```python
    tier_map = {}  # name -> tier_id (0=clean, 1=corrupted, 2=fake_ood)
```

After the clean data loader is appended (the line `loaders.append(("clean", ...))`), add:
```python
    tier_map["clean"] = 0
```

In the corruptions loop, after each `loaders.append((cname, ...))`, add:
```python
        tier_map[cname] = 1
```

In the fake OOD loop, after each `loaders.append((label, ...))`, add:
```python
            tier_map[label] = 2
```

In the real OOD fallback section, after each `loaders.append((f"OOD {name}", ...))`, add:
```python
            tier_map[f"OOD {name}"] = 2
```

Change the return statement from:
```python
    return loaders
```
to:
```python
    return loaders, tier_map
```

#### B2c. Modify `run_phase2` to use tier-aware losses

1. Update the call to `_build_p2_loaders`. Change:
```python
    p2_sources = _build_p2_loaders(targets, train_ds, val_ds, args)
```
to:
```python
    p2_sources, tier_map = _build_p2_loaders(targets, train_ds, val_ds, args)
```

2. Track tier IDs when building the feature dataset. In the loop that extracts features, change from:
```python
    all_eu_in, all_eu_tgt = [], []
    for src_name, loader, eu_arr in p2_sources:
        feats = _extract_eu_inputs(model, loader, device)
        all_eu_in.append(feats)
        all_eu_tgt.append(torch.tensor(eu_arr[:len(feats)], dtype=torch.float32))
    all_eu_in = torch.cat(all_eu_in)
    all_eu_tgt = torch.cat(all_eu_tgt)
```
to:
```python
    all_eu_in, all_eu_tgt, all_tiers = [], [], []
    for src_name, loader, eu_arr in p2_sources:
        feats = _extract_eu_inputs(model, loader, device)
        all_eu_in.append(feats)
        all_eu_tgt.append(torch.tensor(eu_arr[:len(feats)], dtype=torch.float32))
        tier_id = tier_map.get(src_name, 1)  # default to corrupted if unknown
        all_tiers.append(torch.full((len(feats),), tier_id, dtype=torch.long))
    all_eu_in = torch.cat(all_eu_in)
    all_eu_tgt = torch.cat(all_eu_tgt)
    all_tiers = torch.cat(all_tiers)
```

3. Change the TensorDataset to include tiers. Change:
```python
    feat_ds = TensorDataset(all_eu_in, all_eu_tgt)
```
to:
```python
    feat_ds = TensorDataset(all_eu_in, all_eu_tgt, all_tiers)
```

Also update the WeightedRandomSampler section — no changes needed there since it only uses `all_eu_tgt` for weights.

4. Update the training loop to unpack tiers and use new losses. Change the batch unpacking from:
```python
        for eu_in_batch, eu_tgt_batch in feat_loader:
            eu_in_batch = eu_in_batch.to(device, non_blocking=True)
            eu_tgt_batch = eu_tgt_batch.to(device, non_blocking=True)
```
to:
```python
        for eu_in_batch, eu_tgt_batch, tier_batch in feat_loader:
            eu_in_batch = eu_in_batch.to(device, non_blocking=True)
            eu_tgt_batch = eu_tgt_batch.to(device, non_blocking=True)
            tier_batch = tier_batch.to(device, non_blocking=True)
```

5. Update the EU head forward pass to use the new 3-layer architecture. Change:
```python
            eu_pred = model.eu_fc2(
                F.leaky_relu(model.eu_fc1(eu_in_batch), 0.1)).squeeze(-1)
```
to:
```python
            h = F.leaky_relu(model.eu_fc1(eu_in_batch), 0.1)
            h = model.eu_drop(h)
            h = F.leaky_relu(model.eu_fc2(h), 0.1)
            eu_pred = (model.eu_scale * model.eu_fc3(h)).squeeze(-1)
```

6. Replace the loss computation. Change:
```python
            if args.asymmetric_weight > 1.0:
                mse = asymmetric_log1p_mse_loss(eu_pred, eu_tgt_batch,
                                                 under_weight=args.asymmetric_weight)
            else:
                mse = log1p_mse_loss(eu_pred, eu_tgt_batch)
            rank = pairwise_ranking_loss(eu_pred, eu_tgt_batch)
            loss = mse + args.rank_weight * rank
```
to:
```python
            mse = tier_asymmetric_log1p_mse(eu_pred, eu_tgt_batch, tier_batch)
            rank = pairwise_ranking_loss(eu_pred, eu_tgt_batch)

            # ID suppression: penalize over-predicting EU on clean data
            clean_mask = (tier_batch == 0)
            if clean_mask.any():
                suppress = id_suppression_loss(
                    eu_pred[clean_mask], eu_tgt_batch[clean_mask])
            else:
                suppress = eu_pred.new_tensor(0.0)

            # Tier margin: push clean < corrupted < OOD
            margin = tier_margin_loss(eu_pred, tier_batch)

            loss = mse + args.rank_weight * rank \
                   + args.suppress_weight * suppress \
                   + args.margin_weight * margin
```

7. Update the loss logging. Change the print line to also show suppress and margin:
```python
        print(f"  P2 [{epoch+1:02d}/{args.p2_epochs}] lr={lr:.6f}  "
              f"loss={total_loss/count:.6f}  mse={total_mse/count:.6f}  "
              f"rank={total_rank/count:.6f}  "
              f"pearson={pearson:.4f}  spearman={spearman:.4f}")
```
(This can stay as-is for now — the important thing is the overall loss. If you want, add suppress/margin tracking but it's optional.)

#### B2d. Update argument defaults

Change these defaults:
| Argument | Old | New |
|---|---|---|
| `--p1_epochs` | 50 | **100** |
| `--p1_unfreeze_blocks` | 2 | **4** |
| `--p1_lr` | 1e-4 | **2e-4** |
| `--warmup` | 5 | **10** |
| `--p2_epochs` | 150 | **200** |

Add these NEW arguments:
```python
    parser.add_argument("--suppress_weight", type=float, default=2.0,
                        help="Weight for ID EU suppression loss")
    parser.add_argument("--margin_weight", type=float, default=0.5,
                        help="Weight for tier margin loss")
```

### B3. Run E3 Phase 1

```bash
cd ~/maw6/unc_regression/TinyImageNet
mkdir -p ./checkpoints_v3

# Copy teacher targets and ensemble members (do NOT retrain teacher)
cp ./checkpoints_rich_fake_ood_12m/teacher_targets.npz ./checkpoints_v3/
cp ./checkpoints_rich_fake_ood_12m/ensemble_configs.json ./checkpoints_v3/
cp ./checkpoints_rich_fake_ood_12m/member_*.pt ./checkpoints_v3/

# Run Phase 1 only (Phase 2 will be run separately after Phase 1 completes)
python distill.py --save_dir ./checkpoints_v3 \
  --p1_epochs 100 --p1_unfreeze_blocks 4 --p1_lr 2e-4 --warmup 10 \
  --p1_backbone_lr_factor 0.01 --llrd 0.75 \
  --alpha 0.7 --tau 2.0 \
  --gpu 0 2>&1 | tee retrain_v3_p1.log
```

**IMPORTANT:** This command runs BOTH phases. When Phase 1 finishes, it automatically continues to Phase 2. If you want to inspect Phase 1 results before Phase 2, use the `--phase2_only` approach:

Alternative (split approach):
```bash
# Phase 1 only — stop the script after Phase 1 by adding a check, OR just let it run
# and re-run Phase 2 with --phase2_only if Phase 1 results look good.
```

### B4. Monitor E3 Phase 1

**GOOD signs:**
- Val accuracy steadily improves past 87.5% (the old best was 87.51%)
- Val accuracy reaches >= 88.0% (matching teacher's 88.13%)
- Training loss decreases smoothly with no divergence
- EMA val accuracy tracks close to raw val accuracy

**BAD signs:**
- Val accuracy stalls below 87.5% → LLRD too aggressive, try `llrd=0.85` or `p1_lr=3e-4`
- Val accuracy oscillates wildly → gradient clipping too loose or lr too high
- Train accuracy >> val accuracy by 5%+ → overfitting, reduce `p1_unfreeze_blocks` to 3

**Expected time:** ~100 epochs. Time per epoch depends on GPU but expect 2-5 min/epoch. Total ~3-8 hours.

### B5. Run E3 Phase 2 (if using split approach)

If you ran Phase 1 separately and want to rerun Phase 2 with the new losses:

```bash
cd ~/maw6/unc_regression/TinyImageNet
python distill.py --save_dir ./checkpoints_v3 \
  --phase2_only \
  --p2_epochs 200 --p2_lr 0.003 \
  --rank_weight 1.0 --suppress_weight 2.0 --margin_weight 0.5 \
  --eu_sample_alpha 10.0 \
  --gpu 0 2>&1 | tee retrain_v3_p2.log
```

### B6. Monitor E3 Phase 2

**GOOD signs:**
- Spearman starts above 0.70 and climbs past 0.84
- Spearman continues improving past epoch 100 (not plateauing early)
- Loss decreases smoothly

**BAD signs:**
- Spearman lower than v2 baseline (0.84) → new losses may be too aggressive, try `suppress_weight=1.0` and `margin_weight=0.2`
- Loss oscillates or diverges → learning rate too high, try `p2_lr=0.001`
- Spearman plateaus at v2 level (~0.84) → new losses aren't hurting but also not helping on this metric (check AUROC in eval instead, since the real goal is OOD separation, not Spearman)

### B7. Add post-hoc EU calibration to evaluation

**File:** `/home/maw6/maw6/unc_regression/TinyImageNet/evaluate_student.py`

Add an import at the top:
```python
from sklearn.isotonic import IsotonicRegression
```

After the "EU Correlation" section (section 3) computes `student_eu_clean` and `teacher_eu_clean` for the clean validation set, add a calibration step:

```python
    # ── Post-hoc EU calibration via isotonic regression ──
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(student_eu_clean, teacher_eu_clean)
    print(f"\n  Isotonic calibration fitted on {len(student_eu_clean)} clean val samples")
    print(f"  Student EU mean: {student_eu_clean.mean():.4f} -> Calibrated: {iso_reg.predict(student_eu_clean).mean():.4f}")
    print(f"  Teacher EU mean: {teacher_eu_clean.mean():.4f}")
```

Then in the OOD detection sections (4a and 4b), after computing AUROC for "Student EU (learned)", also compute AUROC for the calibrated version:

```python
    # For each OOD dataset, after computing student_eu_ood:
    student_eu_ood_calibrated = iso_reg.predict(student_eu_ood)
    auroc_calibrated = compute_auroc(student_eu_clean_calibrated, student_eu_ood_calibrated)
    print(f"  Student EU (calibrated)                  {auroc_calibrated:.4f}")
```

Where `student_eu_clean_calibrated = iso_reg.predict(student_eu_clean)`.

**Note:** This is a lightweight addition. If sklearn is not installed, install it: `pip install scikit-learn`. It should already be available.

### B8. Evaluate E3

```bash
cd ~/maw6/unc_regression/TinyImageNet
python evaluate_student.py --save_dir ./checkpoints_v3 --gpu 0 2>&1 | tee retrain_v3_eval.log
```

### B9. Success criteria for E3

| Metric | Old (v2) | Target (v3) | Notes |
|---|---|---|---|
| Student accuracy | 87.50% | >= 88.0% | Phase 1 improvement |
| Student EU mean (clean ID) | 0.0841 | <= 0.065 | Must drop to teacher level (0.062) |
| EU Spearman (clean) | 0.8407 | >= 0.84 | Maintain (not the bottleneck) |
| SVHN AUROC (Stu EU) | 0.8974 | >= 0.93 | Main target |
| SVHN AUROC (calibrated) | N/A | >= 0.95 | Post-hoc calibration bonus |
| CIFAR-100 AUROC (Stu EU) | 0.8262 | >= 0.85 | |
| CIFAR-10 AUROC (Stu EU) | 0.8035 | >= 0.84 | |
| FashionMNIST AUROC (Stu EU) | 0.8874 | >= 0.92 | |

**The key diagnostic number is Student EU mean on clean ID.** If it drops from 0.084 to ~0.062 (matching teacher), all AUROC numbers should improve significantly even without other changes.

---

## Monitoring Checklist

Use this checklist to track progress:

### E5 (SST-2)
- [ ] A1: models.py modified (wider EU head + dropout)
- [ ] A2: distill.py modified (LLRD, lower lr, lower tau, early stopping, text aug)
- [ ] A3: Training started
- [ ] A4: Phase 1 monitored — best dev acc = ___% at epoch ___
- [ ] A5: Phase 2 monitored — best Spearman = ___ at epoch ___
- [ ] A6: Evaluation complete
- [ ] A7: Success criteria checked

### E3 (TinyImageNet)
- [ ] B1: models.py modified (wider EU head + dropout + learnable scale)
- [ ] B2: distill.py modified (new losses + tier tracking + args)
- [ ] B3: Phase 1 training started
- [ ] B4: Phase 1 monitored — best val_acc = ___% at epoch ___
- [ ] B5: Phase 2 training started (if split approach)
- [ ] B6: Phase 2 monitored — best Spearman = ___ at epoch ___
- [ ] B7: Post-hoc calibration added to evaluate_student.py
- [ ] B8: Evaluation complete
- [ ] B9: Success criteria checked — Student EU mean (clean ID) = ___

---

## Troubleshooting

### If E5 Phase 1 still overfits (dev acc drops after early epochs)
1. Reduce lr further: try `1e-5`
2. Increase weight decay: try `0.1`
3. Increase LLRD: try `0.8` (more aggressive decay for lower layers)
4. Reduce alpha: try `0.3` (more weight on hard labels)

### If E3 Phase 2 Spearman drops below v2 baseline
1. Reduce `suppress_weight` from 2.0 to 1.0 or 0.5
2. Reduce `margin_weight` from 0.5 to 0.1
3. Check if the tier labels are correctly assigned by printing counts: `tier 0: N, tier 1: N, tier 2: N`

### If E3 Student EU mean on clean ID doesn't decrease
1. Increase `suppress_weight` to 5.0
2. Check that tier 0 (clean) samples are present in each batch — if using WeightedRandomSampler with high alpha, clean samples may be under-sampled
3. Try reducing `eu_sample_alpha` from 10.0 to 5.0 to let more clean samples through

### If E3 Phase 1 val_acc doesn't improve beyond 87.5%
1. The 4-block unfreezing may be too aggressive. Try `--p1_unfreeze_blocks 3`
2. Increase epochs: try 150
3. Try `--p1_lr 3e-4` with `--llrd 0.8`
