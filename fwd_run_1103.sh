#!/usr/bin/env bash
# forward_check.sh split in two: CPU build now, then the backtests on the first truly idle GPU (not GPU 1).
set -euo pipefail
cd "$(dirname "$0")"; ROOT=$PWD; FWD=$ROOT/fwd; PY=$ROOT/.venv/bin/python
mkdir -p "$FWD/logs" "$FWD/configs"
rm -rf "$FWD/assignment2"
git clone -q --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git "$FWD/assignment2"
MSG=$(git -c safe.directory="$FWD/assignment2" -C "$FWD/assignment2" log -1 --format=%s)
END=$(echo "$MSG" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}' | tr T ' ')
echo "release: $MSG"
rm -rf "$FWD/src" && cp -r "$ROOT/src" "$FWD/src" && rm -rf "$FWD/src/__pycache__"
cp "$ROOT"/configs/final_config_{v1_baseline,p3,p4}_imp.json "$FWD/configs/"
cd "$FWD/src"
WEEKS=$($PY forward_decide.py --list-weeks --data-end "$END"); echo "weeks: $WEEKS"
$PY build_dataset.py --jobs 16 > ../logs/build_dataset.log
$PY build_dataset.py --impute --jobs 16 > ../logs/build_dataset_imp.log
echo "build done $(date +%H:%M)"
while :; do   # first idle GPU: utilization < 5 % and < 2 GB used, GPU 1 excluded (ECC error 10-31)
  G=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
      | awk -F', ' '$1 != 1 && $2 < 5 && $3 < 2000 {print $1; exit}')
  [ -n "$G" ] && break; sleep 60
done
echo "using GPU $G from $(date +%H:%M)"
for j in "v1_imp v1_baseline_imp" "p3_imp p3_imp" "p4_imp p4_imp"; do
  set -- $j
  CUDA_VISIBLE_DEVICES=$G $PY pipeline.py --backtest --name fwd_$1 --config configs/final_config_$2.json --val "$WEEKS" > ../logs/fwd_$1.log 2>&1
  echo "fwd_$1 done $(date +%H:%M)"
done
cd "$ROOT/src"
ASG2_DB="$FWD/assignment2/data/db" $PY forward_decide.py --fwd "$FWD/results" --data-end "$END"
