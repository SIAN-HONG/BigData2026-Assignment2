#!/usr/bin/env bash
# Q2 / Q3 ablations repeated on the GH Archive tables (same commands as the original ledger rows,
# names prefixed new_).  Validation: Wednesdays 09-02 .. 10-07 (09-02 reported only), every 3rd cutoff.
# Usage: ./ablation_new.sh <gpu>   (check nvidia-smi first)
set -euo pipefail
cd "$(dirname "$0")/src"; PY=../.venv/bin/python; export CUDA_VISIBLE_DEVICES=${1:?gpu}
run() { n=$1; shift; $PY exp.py --name new_$n --every 3 "$@" > ../logs/new_$n.log 2>&1; echo "new_$n done $(date +%H:%M)"; }
run abl_T0 --groups T0
run abl_T0T1 --groups T0,T1
run abl_T0T1R1 --groups T0,T1,R1
run abl_full --groups T0,T1,R1,R2
run abl_T0R1R2 --groups T0,R1,R2
run abl_full_s1 --groups T0,T1,R1,R2 --seed 1
run abl_full_s2 --groups T0,T1,R1,R2 --seed 2
run full_hl90 --groups T0,T1,R1,R2 --params '{"halflife_days": 90}'
run full_start0601 --groups T0,T1,R1,R2 --start 2026-06-01
run full_xgb_d10 --groups T0,T1,R1,R2 --params '{"max_depth": 10, "min_child_weight": 50}'
for v in gnn mlp; do
  $PY gnn.py --variant $v --name new_${v}_T0T1R1 --every 3 --epochs 8 > ../logs/new_${v}_T0T1R1.log 2>&1; echo "new_${v} done $(date +%H:%M)"
done
