#!/bin/bash
# SST-2 (E5) — Full experiment pipeline
# Usage: bash run_experiments.sh
#        GPU=0 bash run_experiments.sh
#        SECTIONS="0 A" bash run_experiments.sh

set -e

GPU=${GPU:-0}
SAVE_DIR=${SAVE_DIR:-./checkpoints}
DATA_DIR=${DATA_DIR:-../data}
SECTIONS=${SECTIONS:-"0"}

echo "============================================================"
echo "  SST-2 (E5) Experiment Pipeline"
echo "  GPU=$GPU  SAVE_DIR=$SAVE_DIR"
echo "  Sections: $SECTIONS"
echo "============================================================"

run_section() {
    local section=$1
    case $section in
        0)
            echo ""
            echo "=== Section 0: Main Model ==="

            if [ ! -f "$SAVE_DIR/member_0.pt" ]; then
                echo "Training ensemble..."
                python train_ensemble.py --num_members 5 --epochs 5 \
                    --save_dir "$SAVE_DIR" --gpu $GPU
            fi

            if [ ! -f "$SAVE_DIR/teacher_targets.npz" ]; then
                echo "Caching teacher targets..."
                python cache_ensemble_targets.py --save_dir "$SAVE_DIR" --gpu $GPU
            fi

            if [ ! -f "$SAVE_DIR/student.pt" ]; then
                echo "Distilling student..."
                python distill.py --save_dir "$SAVE_DIR" --gpu $GPU
            fi

            echo "Evaluating student..."
            python evaluate_student.py --save_dir "$SAVE_DIR" --gpu $GPU
            ;;

        A)
            echo ""
            echo "=== Section A: Curriculum Ablation ==="
            for curriculum in A1 A2 A3; do
                abl_dir="${SAVE_DIR}_abl_A_${curriculum}"
                mkdir -p "$abl_dir"
                # Link ensemble + targets
                for f in member_*.pt teacher_targets.npz ensemble_configs.json student_phase1.pt; do
                    [ -f "$SAVE_DIR/$f" ] && ln -sf "$(realpath $SAVE_DIR/$f)" "$abl_dir/$f" 2>/dev/null || true
                done
                echo "  Curriculum=$curriculum -> $abl_dir"
                python distill.py --save_dir "$abl_dir" --gpu $GPU \
                    --phase2_only --curriculum $curriculum
                python evaluate_student.py --save_dir "$abl_dir" --gpu $GPU
            done
            ;;

        *)
            echo "Unknown section: $section"
            ;;
    esac
}

for section in $SECTIONS; do
    run_section "$section"
done

echo ""
echo "============================================================"
echo "  All requested sections complete!"
echo "============================================================"
