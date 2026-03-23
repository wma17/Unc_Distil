#!/bin/bash
# Wait for the training chain (PID 1105268) to finish
WAIT_PID=1105268
echo "$(date): Waiting for CVaR training chain (PID $WAIT_PID)..."
while kill -0 $WAIT_PID 2>/dev/null; do
    sleep 30
done
echo "$(date): Training done. Starting evaluations."

cd /home/maw6/maw6/unc_regression/TinyImageNet
PYTHON=/home/maw6/miniconda3/envs/maw6/bin/python
DATA=/home/maw6/maw6/unc_regression/data

for DIR in checkpoints_p2_true_asym_geo2_cvar1 checkpoints_p2_true_asym_geo2_cvar2; do
    if [ ! -f "$DIR/student.pt" ]; then
        echo "$(date): SKIP $DIR (no student.pt)"
        continue
    fi
    echo "$(date): Evaluating $DIR ..."
    $PYTHON -u evaluate_student.py \
      --save_dir ./$DIR \
      --data_dir $DATA \
      --gpu 0 > $DIR/eval.log 2>&1
    echo "$(date): Done $DIR"
    SVHN=$(grep -E "SVHN.*seen.*\|" $DIR/eval.log | awk -F'|' '{print $2}' | awk '{print $1}')
    STL10=$(grep -E "STL10.*unseen.*\|" $DIR/eval.log | awk -F'|' '{print $2}' | awk '{print $1}')
    SPEAR=$(grep "Clean TinyImageNet val" $DIR/eval.log | awk '{print $3}')
    echo "  SVHN=$SVHN  STL10=$STL10  Spearman=$SPEAR"
done

echo "$(date): CVaR evaluations complete."
