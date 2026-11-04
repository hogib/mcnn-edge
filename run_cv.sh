#!/usr/bin/env bash
# 5-fold cross-validation of one configuration: train + quantize each fold, then pool.
#   ./run_cv.sh NAME CONFIG SCALE EPOCHS AUG [SEED] [extra train args...]
#   e.g.  ./run_cv.sh small_cv small 2 60 basic 0      ./run_cv.sh seg_cv small 4/3 150 strong 0 --seg --seg-bg-w 0.1
# Fold k is the test set, fold (k+1) mod 5 the validation set. Output: runs/NAME_f{0..4}, runs/NAME_cv.json
set -euo pipefail
export PYTHONUNBUFFERED=1
name=$1; config=$2; scale=$3; epochs=$4; aug=${5:-basic}; seed=${6:-0}; shift $(( $# < 6 ? $# : 6 )); extra=("$@")
for k in 0 1 2 3 4; do
  v=$(( (k + 1) % 5 ))
  out=runs/${name}_f$k
  if [ -f "$out/results.json" ]; then echo "skip $out (done)"; else
    uv run sonar-edge train --config "$config" --scale "$scale" --epochs "$epochs" --aug "$aug" \
      --test-fold "$k" --val-fold "$v" --seed "$seed" --out "$out" "${extra[@]}" 2>&1 | grep --line-buffered -v Warning | tee "$out.log"
  fi
  if [ ! -f "$out/pred_test_int8.json" ]; then
    uv run sonar-edge quantize "$out/best.pt" --test-fold "$k" --val-fold "$v" 2>&1 | grep --line-buffered -v Warning | tee -a "$out.log"
  fi
done
uv run sonar-edge cv "runs/$name" 2>&1 | tee "runs/${name}_cv.log"
