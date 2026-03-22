#!/bin/bash
# CIFAR-10 Ablation Studies
# All ablations use the same Phase 1 checkpoint from checkpoints_16members
set -e

GPU=${GPU:-0}
BASE_DIR=./checkpoints_16members
DATA_DIR=../data
P2_EPOCHS=100
RANK_WEIGHT=1.0

echo "============================================================"
echo "  CIFAR-10 Ablation Studies"
echo "  GPU=$GPU  BASE_DIR=$BASE_DIR"
echo "============================================================"

# Helper: create ablation dir with symlinks to Phase 1 checkpoint + targets
setup_abl() {
    local abl_dir=$1
    mkdir -p "$abl_dir"
    for f in student_phase1.pt teacher_targets.npz ensemble_configs.json; do
        [ -f "$BASE_DIR/$f" ] && ln -sf "$(realpath $BASE_DIR/$f)" "$abl_dir/$f" 2>/dev/null || true
    done
    # Also link member checkpoints for evaluation
    for f in "$BASE_DIR"/member_*.pt; do
        [ -f "$f" ] && ln -sf "$(realpath $f)" "$abl_dir/$(basename $f)" 2>/dev/null || true
    done
}

# ============================================================
# Ablation A: Phase 2 Training Curriculum
# ============================================================
echo ""
echo "=== Ablation A: Curriculum ==="

for curriculum in A1 A2 A3; do
    abl_dir="${BASE_DIR}_abl_A_${curriculum}"
    setup_abl "$abl_dir"
    echo ""
    echo "--- Curriculum=$curriculum -> $abl_dir ---"
    python distill.py --save_dir "$abl_dir" --data_dir "$DATA_DIR" --gpu $GPU \
        --phase2_only --p2_epochs $P2_EPOCHS --rank_weight $RANK_WEIGHT \
        --curriculum $curriculum
    python evaluate_student.py --save_dir "$abl_dir" --data_dir "$DATA_DIR" --gpu $GPU
done

# ============================================================
# Ablation B: Phase 2 Loss Components
# ============================================================
echo ""
echo "=== Ablation B: Loss Components ==="

for loss_mode in mse log_mse ranking combined; do
    abl_dir="${BASE_DIR}_abl_B_${loss_mode}"
    setup_abl "$abl_dir"
    echo ""
    echo "--- Loss=$loss_mode -> $abl_dir ---"
    python distill.py --save_dir "$abl_dir" --data_dir "$DATA_DIR" --gpu $GPU \
        --phase2_only --p2_epochs $P2_EPOCHS --rank_weight $RANK_WEIGHT \
        --loss_mode $loss_mode
    python evaluate_student.py --save_dir "$abl_dir" --data_dir "$DATA_DIR" --gpu $GPU
done

# ============================================================
# Ablation C: Ensemble Size K
# ============================================================
echo ""
echo "=== Ablation C: Ensemble Size K ==="

for K in 3 5 10; do
    abl_dir="${BASE_DIR}_abl_C_K${K}"
    mkdir -p "$abl_dir"
    # Link member checkpoints
    for f in "$BASE_DIR"/member_*.pt "$BASE_DIR"/ensemble_configs.json; do
        [ -f "$f" ] && ln -sf "$(realpath $f)" "$abl_dir/$(basename $f)" 2>/dev/null || true
    done

    # Re-cache targets with subset K
    if [ ! -f "$abl_dir/teacher_targets.npz" ]; then
        echo ""
        echo "--- Caching targets for K=$K ---"
        python cache_ensemble_targets.py --save_dir "$abl_dir" --data_dir "$DATA_DIR" \
            --gpu $GPU --p2_data_mode fake_ood --subset_size $K
    fi

    # Link Phase 1 checkpoint
    ln -sf "$(realpath $BASE_DIR/student_phase1.pt)" "$abl_dir/student_phase1.pt" 2>/dev/null || true

    echo ""
    echo "--- K=$K -> $abl_dir ---"
    python distill.py --save_dir "$abl_dir" --data_dir "$DATA_DIR" --gpu $GPU \
        --phase2_only --p2_epochs $P2_EPOCHS --rank_weight $RANK_WEIGHT
    python evaluate_student.py --save_dir "$abl_dir" --data_dir "$DATA_DIR" --gpu $GPU
done

# ============================================================
# Ablation D: Real OOD vs Synthetic OOD
# D1=fake_ood (already in A3/combined), D2=real_ood
# ============================================================
echo ""
echo "=== Ablation D: Real OOD vs Synthetic OOD ==="

abl_dir="${BASE_DIR}_abl_D_real_ood"
mkdir -p "$abl_dir"
for f in "$BASE_DIR"/member_*.pt "$BASE_DIR"/ensemble_configs.json "$BASE_DIR"/student_phase1.pt; do
    [ -f "$f" ] && ln -sf "$(realpath $f)" "$abl_dir/$(basename $f)" 2>/dev/null || true
done

if [ ! -f "$abl_dir/teacher_targets.npz" ]; then
    echo "--- Caching targets with real OOD ---"
    python cache_ensemble_targets.py --save_dir "$abl_dir" --data_dir "$DATA_DIR" \
        --gpu $GPU --p2_data_mode ood
fi

echo ""
echo "--- D2: Real OOD -> $abl_dir ---"
python distill.py --save_dir "$abl_dir" --data_dir "$DATA_DIR" --gpu $GPU \
    --phase2_only --p2_epochs $P2_EPOCHS --rank_weight $RANK_WEIGHT
python evaluate_student.py --save_dir "$abl_dir" --data_dir "$DATA_DIR" --gpu $GPU

# ============================================================
# Ablation E: Joint Training
# ============================================================
echo ""
echo "=== Ablation E: Joint Training ==="

abl_dir="${BASE_DIR}_abl_E_joint"
setup_abl "$abl_dir"
echo ""
echo "--- Joint training -> $abl_dir ---"
python distill.py --save_dir "$abl_dir" --data_dir "$DATA_DIR" --gpu $GPU \
    --joint --joint_gamma 1.0 --p1_epochs 200
python evaluate_student.py --save_dir "$abl_dir" --data_dir "$DATA_DIR" --gpu $GPU

echo ""
echo "============================================================"
echo "  All CIFAR-10 ablations complete!"
echo "============================================================"
