#!/usr/bin/env bash
# Build the tables from GH Archive and write the submission file.
# Usage: ./run_final.sh <studentID> <gpu_index> [config] [cutoff]
#   config  default configs/final_config_xgb_d3.json (52 features, XGBoost depth 3 x 3 seeds; chosen by the 6-week rolling backtest)
#   cutoff  last UTC hour used, exclusive; default 2026-11-21T12:00 (= 20:00 Taiwan, 4 h before the
#           test week 11-22 00:00 Taiwan).  An earlier cutoff gives a rehearsal file.
# Check nvidia-smi for an idle GPU first.
set -euo pipefail
SID=${1:?student id}; GPU=${2:?gpu}; CFG=${3:-configs/final_config_xgb_d3.json}; CUT=${4:-2026-11-21T12:00}
cd "$(dirname "$0")"
unset ASG2_DB ASG2_ART ASG2_OUTAGE_RULE   # the submission always uses data/db and artifacts/ (recorded by manifest.py)
[ -f "$CFG" ] || { echo "config not found: $CFG"; exit 1; }
PY=$PWD/.venv/bin/python
RAW=${RAW:-$HOME/gharchive}          # GH Archive hour files (kept between runs)
mkdir -p logs
# 1. course repository (preprocess.py, evaluate.py, test_input.csv) at a fixed commit; delete and clone again
COURSE_COMMIT=${COURSE_COMMIT:-af9f335301d9badb482dfafa10c4534eabac845a}   # 2026-11-07 revision
rm -rf assignment2
git clone -q https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
HEAD_NOW=$(git -c safe.directory="*" -C assignment2 rev-parse HEAD)
[ "$HEAD_NOW" = "$COURSE_COMMIT" ] || echo "WARNING: course repo has newer commits ($HEAD_NOW); using pinned $COURSE_COMMIT - check them, set COURSE_COMMIT to use another"
git -c safe.directory="*" -C assignment2 -c advice.detachedHead=false checkout -q "$COURSE_COMMIT"
# 2. GH Archive hour files 2025-12-01 .. cutoff (only missing files are downloaded; stops if any hour is missing)
./download_gharchive.sh "$RAW" 2025-12-01 "$CUT" | tee logs/final_download.log
# 3. official preprocessing on exactly those hours -> data/db (the previous tables are removed)
SEL=$(mktemp -d); trap 'rm -rf "$SEL"' EXIT
$PY - "$RAW" "$SEL" "$CUT" <<'EOF'
import os, re, sys
from datetime import datetime
raw, sel, cut = sys.argv[1], sys.argv[2], datetime.fromisoformat(sys.argv[3])
for f in os.listdir(raw):
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})-(\d{1,2})\.json\.gz$", f)
    if m and datetime(2025, 12, 1) <= datetime(*map(int, m.groups())) < cut:   # exactly [2025-12-01, cutoff)
        os.symlink(os.path.join(os.path.abspath(raw), f), os.path.join(sel, f))
EOF
rm -rf data
$PY src/run_preprocess.py --raw "$SEL" --out data | tee logs/final_preprocess.log
cd src
# 4. samples, features at the cutoff, leakage audit, prediction
# weekly Saturday 12:00 UTC cutoffs from 2026-03-21 (build_dataset defaults); imputation method from the
# config's dataset name: dataset_impW = mean of the previous 7 days, dataset_impD = previous day
IMP=$($PY -c "import json,sys; d=json.load(open(sys.argv[1]))['dataset']; print({'dataset_impW.parquet':'week','dataset_impD.parquet':'prevday','dataset_imp.parquet':'copy'}[d])" "../$CFG")
$PY build_dataset.py --impute "$IMP" --jobs 16 --cutoff "$CUT" | tee ../logs/final_build_imp.log
$PY leakage_audit.py --impute="$IMP" | tee ../logs/final_leakage_audit.log
grep -q "LEAKAGE AUDIT PASSED" ../logs/final_leakage_audit.log || { echo "leakage audit failed"; exit 1; }
CUDA_VISIBLE_DEVICES=$GPU $PY predict.py --student-id "$SID" --config "$CFG" --cutoff "$CUT" | tee ../logs/final_predict.log
# 5. manifest: course commit, code/config/table/output hashes, hour files (size + sha256), versions
CUDA_VISIBLE_DEVICES=$GPU $PY manifest.py --raw "$RAW" --start 2025-12-01 --cutoff "$CUT" --config "$CFG" \
    --hash-hours --out ../results/manifest.json | tee ../logs/final_manifest.log
