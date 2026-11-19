#!/usr/bin/env bash
# Addendum 5 of docs/FORWARD_CHECK.md: v1 / p3 / p4 (all with outage imputation) on the tables built
# from GH Archive by the official preprocess.py, 12 development weeks + the fresh weeks.
# Usage: ./backtest_new.sh <gpu_v1> <gpu_p3> <gpu_p4> [fresh weeks, default 2026-10-14,2026-10-21,2026-10-28]
# PREFIX=a6 names the results a6_* (default new_*).  Check nvidia-smi first.  Needs artifacts/dataset_imp.parquet (build_dataset.py --impute).
set -euo pipefail
cd "$(dirname "$0")/src"; PY=../.venv/bin/python
FRESH=${4:-2026-10-14,2026-10-21,2026-10-28}
WEEKS=2026-05-06,2026-05-27,2026-06-17,2026-07-08,2026-07-29,2026-08-19,2026-09-02,2026-09-09,2026-09-16,2026-09-23,2026-09-30,2026-10-07,$FRESH
GA=${1:?gpu}; GB=${2:?gpu}; GC=${3:?gpu}
PIDS=()
for j in "$GA v1 v1_baseline_imp" "$GB p3 p3_imp" "$GC p4 p4_imp"; do
  set -- $j
  CUDA_VISIBLE_DEVICES=$1 $PY pipeline.py --backtest --name ${PREFIX:-new}_$2_imp --config configs/final_config_$3.json --val "$WEEKS" > ../logs/${PREFIX:-new}_$2_imp.log 2>&1 &
  PIDS+=("$!:$2")
done
fail=0
for p in "${PIDS[@]}"; do   # wait for each job and report every failure (a bare `wait` returns 0)
  wait "${p%%:*}" || { echo "ERROR: backtest ${PREFIX:-new}_${p#*:}_imp failed (see logs/${PREFIX:-new}_${p#*:}_imp.log)"; fail=1; }
done
exit $fail
