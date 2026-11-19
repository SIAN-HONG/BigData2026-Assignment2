#!/usr/bin/env bash
# Pre-registered out-of-sample check on the weeks after 10-07 (docs/FORWARD_CHECK.md, addenda 3 and 4):
# v1 vs p3 vs p4, all with outage imputation (dataset_imp).
# Usage: ./forward_check.sh <gpu_a> [gpu_b] [gpu_c]   (check nvidia-smi first; one job per idle GPU)
# Works in fwd/ (own data clone and artifacts); artifacts/ and the development results are untouched.
set -euo pipefail
GA=${1:?gpu}; GB=${2:-$1}; GC=${3:-$GB}
cd "$(dirname "$0")"
ROOT=$PWD; FWD=$ROOT/fwd; PY=$ROOT/.venv/bin/python
mkdir -p "$FWD/logs" "$FWD/configs"
rm -rf "$FWD/assignment2"
git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git "$FWD/assignment2"
MSG=$(git -c safe.directory="$FWD/assignment2" -C "$FWD/assignment2" log -1 --format=%s)   # "data up to 2026-10-20T12:00:00Z"
END=$(echo "$MSG" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}' | tr T ' ')
echo "release: $MSG"
rm -rf "$FWD/src" && cp -r "$ROOT/src" "$FWD/src" && rm -rf "$FWD/src/__pycache__"
cp "$ROOT"/configs/final_config_{v1_baseline,p3,p4}_imp.json "$FWD/configs/"
cd "$FWD/src"
WEEKS=$($PY forward_decide.py --list-weeks --data-end "$END")
if [ -z "$WEEKS" ]; then
  echo "no fresh week has a finished label window yet (data end $END)"
else
  $PY build_dataset.py --jobs 16 > ../logs/build_dataset.log            # events.parquet (GNN graphs, label censoring)
  $PY build_dataset.py --impute --jobs 16 > ../logs/build_dataset_imp.log
  prev=""
  for job in "$GA v1_imp v1_baseline_imp" "$GB p3_imp p3_imp" "$GC p4_imp p4_imp"; do
    set -- $job
    [ "$1" = "$prev" ] && wait             # same GPU as the previous job: run one after the other
    CUDA_VISIBLE_DEVICES=$1 $PY pipeline.py --backtest --name fwd_$2 --config configs/final_config_$3.json --val "$WEEKS" > ../logs/fwd_$2.log 2>&1 &
    prev=$1
  done
  wait
fi
cd "$ROOT/src"
ASG2_DB="$FWD/assignment2/data/db" $PY forward_decide.py --fwd "$FWD/results" --data-end "$END"   # outages detected on the fresh release
