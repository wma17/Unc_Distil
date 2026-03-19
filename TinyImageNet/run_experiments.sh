#!/usr/bin/env bash
# =============================================================================
# TinyImageNet — Full Evaluation Pipeline
# Runs evaluate_student.py on all trained checkpoint directories.
#
# Usage:
#   bash run_experiments.sh
#   GPU=1 bash run_experiments.sh
#   SAVE_DIR=./checkpoints bash run_experiments.sh   # single dir only
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/home/maw6/maw6/unc_regression/data}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"

cd "$ROOT"
export PYTHONUNBUFFERED=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Helper: evaluate one checkpoint directory
# ---------------------------------------------------------------------------
evaluate() {
    local save_dir="$1"
    local label="$2"
    if [ ! -f "$save_dir/student.pt" ]; then
        log "SKIP $label — student.pt not found in $save_dir"
        return
    fi
    log ">>> Evaluating: $label"
    python evaluate_student.py \
        --save_dir  "$save_dir" \
        --data_dir  "$DATA_DIR" \
        --batch_size "$BATCH_SIZE" \
        --gpu "$GPU" \
        2>&1 | tee "$save_dir/eval.log"
    log "<<< Done: $label  (results -> $save_dir/../result.md, log -> $save_dir/eval.log)"
}

# ---------------------------------------------------------------------------
# If SAVE_DIR is set explicitly, evaluate that directory only
# ---------------------------------------------------------------------------
if [ -n "${SAVE_DIR:-}" ]; then
    log "===== TinyImageNet Evaluation (single dir: $SAVE_DIR) ====="
    evaluate "$SAVE_DIR" "$(basename "$SAVE_DIR")"
    log "===== Done ====="
    exit 0
fi

# ---------------------------------------------------------------------------
# Otherwise, evaluate all known checkpoint directories in priority order
# ---------------------------------------------------------------------------
log "===== TinyImageNet Evaluation Pipeline ====="

# Best model first (most recent / highest-performing variant)
evaluate "$ROOT/checkpoints_fake_ood_ft_wd005_blr1e3"  "fake_ood wd005 blr1e3 (best)"
evaluate "$ROOT/checkpoints_fake_ood_ft_moderate"       "fake_ood ft_moderate"
evaluate "$ROOT/checkpoints_fake_ood_ft_lowblr"         "fake_ood ft_lowblr"
evaluate "$ROOT/checkpoints_fake_ood_ft"                "fake_ood ft (baseline)"
evaluate "$ROOT/checkpoints"                            "main checkpoints"

log "===== All TinyImageNet evaluations complete ====="
