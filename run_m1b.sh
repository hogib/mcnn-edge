#!/usr/bin/env bash
# M1 iteration: longer schedule, strong augmentation, width. Dev split (fold 0 test, fold 1 val).
set -euo pipefail
export PYTHONUNBUFFERED=1
run() { name=$1; shift; uv run sonar-edge train "$@" --out runs/$name 2>&1 | grep --line-buffered -v Warning | tee runs/$name.log; }
run base_e150        --config base  --scale 1 --epochs 150 --aug basic
run base_e150_aug    --config base  --scale 1 --epochs 150 --aug strong
run small_e150_aug   --config small --scale 2 --epochs 150 --aug strong
run wide_e150_aug    --config wide  --scale 1 --epochs 150 --aug strong
