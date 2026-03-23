#!/bin/bash
# Wait for training chain to complete (PID 856895 = last wrapper in chain)
WAIT_PID=856895
echo "$(date): Waiting for training chain (PID $WAIT_PID) to finish..."
while kill -0 $WAIT_PID 2>/dev/null; do
    sleep 30
done
echo "$(date): Training chain done. Starting evaluations."

cd /home/maw6/maw6/unc_regression/TinyImageNet

PYTHON=/home/maw6/miniconda3/envs/maw6/bin/python
DATA=/home/maw6/maw6/unc_regression/data

for DIR in checkpoints_p2_original_repro checkpoints_p2_suppress_geo2_alpha0 checkpoints_p2_asym_geo2_alpha0 checkpoints_p2_true_asym_geo2; do
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
    # Print key metrics
    SVHN=$(grep -A5 "vs SVHN" $DIR/eval.log | grep "Student EU (learned)" | head -1 | awk '{print $NF}')
    STL10=$(grep -A5 "vs STL10" $DIR/eval.log | grep "Student EU (learned)" | tail -1 | awk '{print $NF}')
    SPEAR=$(grep "Clean TinyImageNet val" $DIR/eval.log | awk '{print $3}')
    echo "  SVHN=$SVHN  STL10=$STL10  Spearman=$SPEAR"
done

echo "$(date): All evaluations complete."
