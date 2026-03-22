#!/bin/bash
# VOC 2012 (E4) — Full experiment pipeline
# Usage: bash run_experiments.sh
#        GPU=0 bash run_experiments.sh

set -e

GPU=${GPU:-0}
SAVE_DIR=${SAVE_DIR:-./checkpoints}
DATA_DIR=${DATA_DIR:-../data}

echo "============================================================"
echo "  Pascal VOC 2012 (E4) Experiment Pipeline"
echo "  GPU=$GPU  SAVE_DIR=$SAVE_DIR"
echo "============================================================"

# Step 1: Train ensemble
if [ ! -f "$SAVE_DIR/member_0.pt" ]; then
    echo ""
    echo "=== Step 1: Training ensemble ==="
    python train_ensemble.py --num_members 5 --iterations 40000 \
        --batch_size 8 --save_dir "$SAVE_DIR" --data_dir "$DATA_DIR" --gpu $GPU
fi

# Step 2: Cache teacher targets
if [ ! -f "$SAVE_DIR/teacher_targets.npz" ]; then
    echo ""
    echo "=== Step 2: Caching teacher targets ==="
    python cache_ensemble_targets.py --save_dir "$SAVE_DIR" \
        --data_dir "$DATA_DIR" --gpu $GPU
fi

# Step 3: Distill student
if [ ! -f "$SAVE_DIR/student.pt" ]; then
    echo ""
    echo "=== Step 3: Distilling student ==="
    python distill.py --save_dir "$SAVE_DIR" --data_dir "$DATA_DIR" --gpu $GPU
fi

# Step 4: Evaluate
echo ""
echo "=== Step 4: Evaluating student ==="
python evaluate_student.py --save_dir "$SAVE_DIR" --data_dir "$DATA_DIR" --gpu $GPU

echo ""
echo "============================================================"
echo "  VOC 2012 (E4) pipeline complete!"
echo "============================================================"
