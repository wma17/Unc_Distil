#!/usr/bin/env bash
# =============================================================================
# CIFAR-10 — Full Experiment Pipeline (Evaluation + All Ablation Studies)
#
# Sections:
#   0. Evaluate main model (existing checkpoint)
#   A. Ablation A — Phase 2 curriculum  (A1/A2/A3)
#   B. Ablation B — Phase 2 loss mode   (mse/log_mse/ranking/combined)
#   C. Ablation C — Ensemble size K     (K=3 and K=5 from existing members)
#   D. Ablation D — OOD source          (real_ood vs fake_ood)
#   E. Ablation E — Joint training      (joint vs two-phase)
#
# All ablation Phase-2 runs reuse the existing Phase 1 checkpoint to isolate
# the Phase 2 variable being tested (faster and cleaner comparison).
# Exception: Ablation E trains everything jointly from scratch.
#
# Usage:
#   bash run_experiments.sh               # run everything
#   SECTIONS="0 A B" bash run_experiments.sh   # run only selected sections
#   GPU=1 bash run_experiments.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/home/maw6/maw6/unc_regression/data}"
SAVE_DIR="${SAVE_DIR:-$ROOT/checkpoints}"   # main trained checkpoint
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WORKERS="${WORKERS:-4}"

# Which sections to run (space-separated). Default = all.
SECTIONS="${SECTIONS:-0 A B C D E}"

# Phase 2 hyperparameters (shared across all ablations)
P2_EPOCHS="${P2_EPOCHS:-100}"
P2_LR="${P2_LR:-0.001}"
RANK_WEIGHT="${RANK_WEIGHT:-1.0}"

# Phase 1 hyperparameters (used only by Ablation E joint training)
P1_EPOCHS="${P1_EPOCHS:-200}"
P1_LR="${P1_LR:-0.1}"
WARMUP="${WARMUP:-10}"
ALPHA="${ALPHA:-0.7}"
TAU="${TAU:-4.0}"
JOINT_GAMMA="${JOINT_GAMMA:-1.0}"

cd "$ROOT"
export PYTHONUNBUFFERED=1

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
hdr()  { echo; echo "================================================================"; echo "  $*"; echo "================================================================"; }
skip() { log "SKIP $* — student.pt not found"; }

# Check whether a section is enabled
has_section() { [[ " $SECTIONS " == *" $1 "* ]]; }

# ---------------------------------------------------------------------------
# Helper: copy shared files into an ablation directory
#   - ensemble member checkpoints (for throughput benchmark in evaluate_student)
#   - ensemble_configs.json
#   - student_phase1.pt (so --phase2_only works)
# Does NOT copy teacher_targets.npz; caller provides or generates it.
# ---------------------------------------------------------------------------
setup_abl_dir() {
    local abl_dir="$1"
    mkdir -p "$abl_dir"
    cp "$SAVE_DIR"/member_*.pt          "$abl_dir/"
    cp "$SAVE_DIR/student_phase1.pt"    "$abl_dir/"
    [ -f "$SAVE_DIR/ensemble_configs.json" ] && \
        cp "$SAVE_DIR/ensemble_configs.json" "$abl_dir/" || true
}

# ---------------------------------------------------------------------------
# Helper: run Phase 2 only + evaluate in a given directory
# ---------------------------------------------------------------------------
run_phase2_and_eval() {
    local abl_dir="$1"; shift   # remaining args passed to distill.py
    log "Phase 2 training in: $abl_dir"
    python distill.py \
        --save_dir   "$abl_dir" \
        --data_dir   "$DATA_DIR" \
        --gpu        "$GPU" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$WORKERS" \
        --p2_epochs  "$P2_EPOCHS" \
        --p2_lr      "$P2_LR" \
        --rank_weight "$RANK_WEIGHT" \
        --phase2_only \
        "$@" \
        2>&1 | tee "$abl_dir/distill.log"

    log "Evaluating: $abl_dir"
    python evaluate_student.py \
        --save_dir   "$abl_dir" \
        --data_dir   "$DATA_DIR" \
        --gpu        "$GPU" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$WORKERS" \
        2>&1 | tee "$abl_dir/eval.log"
}

# ===========================================================================
# Section 0: Evaluate main model
# ===========================================================================
if has_section "0"; then
    hdr "Section 0: Main model evaluation"
    if [ -f "$SAVE_DIR/student.pt" ]; then
        log "Evaluating main checkpoint: $SAVE_DIR"
        python evaluate_student.py \
            --save_dir   "$SAVE_DIR" \
            --data_dir   "$DATA_DIR" \
            --gpu        "$GPU" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$WORKERS" \
            2>&1 | tee "$SAVE_DIR/eval.log"
        log "Done. Log -> $SAVE_DIR/eval.log"
    else
        skip "Section 0 ($SAVE_DIR)"
    fi
fi

# ===========================================================================
# Section A: Ablation A — Phase 2 curriculum (A1 / A2 / A3)
# ===========================================================================
if has_section "A"; then
    hdr "Section A: Ablation — Phase 2 curriculum"
    # A3 = default (ours). Run all three so results are directly comparable.
    for CURR in A1 A2 A3; do
        ABL_DIR="$ROOT/checkpoints_abl_A_${CURR}"
        log "--- Curriculum ${CURR} ---"
        setup_abl_dir "$ABL_DIR"
        cp "$SAVE_DIR/teacher_targets.npz" "$ABL_DIR/"
        run_phase2_and_eval "$ABL_DIR" --curriculum "$CURR"
        log "Finished Ablation A/${CURR} -> $ABL_DIR/eval.log"
    done
fi

# ===========================================================================
# Section B: Ablation B — Phase 2 loss mode (mse / log_mse / ranking / combined)
# ===========================================================================
if has_section "B"; then
    hdr "Section B: Ablation — Phase 2 loss components"
    # combined = default (ours). Run all four.
    for MODE in mse log_mse ranking combined; do
        ABL_DIR="$ROOT/checkpoints_abl_B_${MODE}"
        log "--- Loss mode: ${MODE} ---"
        setup_abl_dir "$ABL_DIR"
        cp "$SAVE_DIR/teacher_targets.npz" "$ABL_DIR/"
        run_phase2_and_eval "$ABL_DIR" --loss_mode "$MODE"
        log "Finished Ablation B/${MODE} -> $ABL_DIR/eval.log"
    done
fi

# ===========================================================================
# Section C: Ablation C — Ensemble size K
# Reuses existing Phase 1 checkpoint (backbone trained with full K=5 teacher),
# but re-caches teacher EU targets with only K members so Phase 2 learns from
# a weaker teacher — isolating the effect of teacher ensemble size on EU quality.
# Note: For K > 5 you must first train more members with train_ensemble.py.
# ===========================================================================
if has_section "C"; then
    hdr "Section C: Ablation — Ensemble size K"
    N_MEMBERS=$(ls "$SAVE_DIR"/member_*.pt 2>/dev/null | wc -l)
    log "Found $N_MEMBERS ensemble members in $SAVE_DIR"

    for K in 3 5; do
        if [ "$K" -gt "$N_MEMBERS" ]; then
            log "SKIP K=$K — only $N_MEMBERS members available"
            continue
        fi
        ABL_DIR="$ROOT/checkpoints_abl_C_K${K}"
        log "--- Ensemble size K=${K} ---"
        mkdir -p "$ABL_DIR"
        # Copy only first K members (for re-caching and throughput benchmark)
        for i in $(seq 0 $((K - 1))); do
            cp "$SAVE_DIR/member_${i}.pt" "$ABL_DIR/"
        done
        cp "$SAVE_DIR/student_phase1.pt" "$ABL_DIR/"
        [ -f "$SAVE_DIR/ensemble_configs.json" ] && \
            cp "$SAVE_DIR/ensemble_configs.json" "$ABL_DIR/" || true

        # Re-cache teacher targets using only K members
        log "Caching teacher targets with K=${K} members..."
        python cache_ensemble_targets.py \
            --save_dir   "$ABL_DIR" \
            --data_dir   "$DATA_DIR" \
            --gpu        "$GPU" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$WORKERS" \
            --subset_size "$K" \
            2>&1 | tee "$ABL_DIR/cache.log"

        run_phase2_and_eval "$ABL_DIR"
        log "Finished Ablation C/K=${K} -> $ABL_DIR/eval.log"
    done
fi

# ===========================================================================
# Section D: Ablation D — OOD source (real OOD vs synthetic/fake OOD)
# Both use the same Phase 1 checkpoint; only the Phase 2 data tier 3 differs.
# D1 (fake_ood) is the self-contained approach (ours).
# D2 (real_ood) uses SVHN + CIFAR-100 as external OOD.
# Evaluate unseen OOD AUROC only (to avoid leakage advantage for real_ood).
# ===========================================================================
if has_section "D"; then
    hdr "Section D: Ablation — Phase 2 OOD source"

    for MODE in fake_ood ood; do
        LABEL="${MODE/ood/real_ood}"   # rename "ood" -> "real_ood" for clarity
        LABEL="${LABEL/fake_/}"        # "fake_ood" -> "fake_real_ood"... fix:
        [ "$MODE" = "ood" ] && LABEL="real_ood" || LABEL="fake_ood"
        ABL_DIR="$ROOT/checkpoints_abl_D_${LABEL}"
        log "--- OOD source: ${LABEL} (p2_data_mode=${MODE}) ---"
        setup_abl_dir "$ABL_DIR"

        # Re-cache with the requested OOD mode
        log "Caching teacher targets with p2_data_mode=${MODE}..."
        python cache_ensemble_targets.py \
            --save_dir         "$ABL_DIR" \
            --data_dir         "$DATA_DIR" \
            --gpu              "$GPU" \
            --batch_size       "$BATCH_SIZE" \
            --num_workers      "$WORKERS" \
            --p2_data_mode     "$MODE" \
            2>&1 | tee "$ABL_DIR/cache.log"

        run_phase2_and_eval "$ABL_DIR"
        log "Finished Ablation D/${LABEL} -> $ABL_DIR/eval.log"
    done
fi

# ===========================================================================
# Section E: Ablation E — Two-phase vs Joint training
# Joint training trains backbone + classifier + EU head simultaneously.
# Two-phase (ours) = the main model already trained. We train joint from
# scratch in a separate directory for a fair comparison.
# ===========================================================================
if has_section "E"; then
    hdr "Section E: Ablation — Joint vs Two-phase training"
    ABL_DIR="$ROOT/checkpoints_abl_E_joint"
    log "--- Joint training ---"
    mkdir -p "$ABL_DIR"
    cp "$SAVE_DIR"/member_*.pt          "$ABL_DIR/"
    cp "$SAVE_DIR/teacher_targets.npz"  "$ABL_DIR/"
    [ -f "$SAVE_DIR/ensemble_configs.json" ] && \
        cp "$SAVE_DIR/ensemble_configs.json" "$ABL_DIR/" || true

    log "Joint distillation training..."
    python distill.py \
        --save_dir    "$ABL_DIR" \
        --data_dir    "$DATA_DIR" \
        --gpu         "$GPU" \
        --batch_size  "$BATCH_SIZE" \
        --num_workers "$WORKERS" \
        --p1_epochs   "$P1_EPOCHS" \
        --p1_lr       "$P1_LR" \
        --warmup_epochs "$WARMUP" \
        --alpha       "$ALPHA" \
        --tau         "$TAU" \
        --joint \
        --joint_gamma "$JOINT_GAMMA" \
        2>&1 | tee "$ABL_DIR/distill.log"

    log "Evaluating joint model..."
    python evaluate_student.py \
        --save_dir   "$ABL_DIR" \
        --data_dir   "$DATA_DIR" \
        --gpu        "$GPU" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$WORKERS" \
        2>&1 | tee "$ABL_DIR/eval.log"

    log "Finished Ablation E/joint -> $ABL_DIR/eval.log"
fi

# ===========================================================================
# Summary
# ===========================================================================
hdr "All requested sections complete"
echo "  Sections run: $SECTIONS"
echo ""
echo "  Results (eval.log) in:"
for d in \
    "$SAVE_DIR" \
    "$ROOT"/checkpoints_abl_A_* \
    "$ROOT"/checkpoints_abl_B_* \
    "$ROOT"/checkpoints_abl_C_* \
    "$ROOT"/checkpoints_abl_D_* \
    "$ROOT"/checkpoints_abl_E_*; do
    [ -f "$d/eval.log" ] && echo "    $d/eval.log"
done
echo ""
