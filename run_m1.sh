#!/usr/bin/env bash
# M1 dev-split runs (fold 0 test, fold 1 val). Logs: runs/<name>/train.log
set -euo pipefail
export PYTHONUNBUFFERED=1
uv run sonar-edge train --config base  --scale 1 --epochs 60 --out runs/base_s1_f0  2>&1 | grep --line-buffered -v Warning | tee runs/base_s1_f0.log
uv run sonar-edge train --config small --scale 2 --epochs 60 --out runs/small_s2_f0 2>&1 | grep --line-buffered -v Warning | tee runs/small_s2_f0.log
