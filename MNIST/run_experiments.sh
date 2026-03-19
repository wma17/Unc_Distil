#!/usr/bin/env bash
# =============================================================================
# MNIST — Full Evaluation Pipeline
# Runs evaluate_student.py on all available checkpoint directories.
#
# Usage:
#   bash run_experiments.sh
#   GPU=1 bash run_experiments.sh
#   SAVE_DIR=./checkpoints bash run_experiments.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/home/maw6/maw6/unc_regression/data}"
GPU="${GPU:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WORKERS="${WORKERS:-4}"

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
        --num_workers "$WORKERS" \
        --gpu "$GPU" \
        2>&1 | tee "$save_dir/eval.log"
    log "<<< Done: $label  (log -> $save_dir/eval.log)"
}

# ---------------------------------------------------------------------------
# 1. Main checkpoint (real OOD)
# ---------------------------------------------------------------------------
log "===== MNIST Evaluation Pipeline ====="
evaluate "$ROOT/checkpoints"          "main (real OOD)"

# ---------------------------------------------------------------------------
# 2. Fake-OOD checkpoint (if it exists)
# ---------------------------------------------------------------------------
evaluate "$ROOT/checkpoints_fake_ood" "fake OOD"

log "===== All MNIST evaluations complete ====="
