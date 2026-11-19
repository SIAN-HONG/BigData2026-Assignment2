#!/usr/bin/env bash
# Formal version decision, docs/FORWARD_CHECK.md addendum 7 (fixed snapshot: hours < 2026-11-17 12:00 UTC).
# Usage: ./run_formal.sh <gpu_v1> <gpu_p3> <gpu_p4>   (check nvidia-smi first)
# Steps: course repo at af9f335 -> GH Archive hours -> official preprocess.py -> build_dataset (both, --cutoff)
#        -> leakage audit -> backtests f17_{v1,p3,p4}_imp (12 dev weeks incl. 09-02 + 4 fresh weeks)
#        -> forward_decide.py --new --formal.  The previous tables / datasets are kept in data_b15/, artifacts_b15/.
set -euo pipefail
GA=${1:?gpu}; GB=${2:?gpu}; GC=${3:?gpu}
CUT=2026-11-17T12:00
FRESH=2026-10-14,2026-10-21,2026-10-28,2026-11-04
cd "$(dirname "$0")"
PY=$PWD/.venv/bin/python
RAW=${RAW:-$HOME/gharchive}
COURSE_COMMIT=af9f335301d9badb482dfafa10c4534eabac845a
mkdir -p logs
echo "== $(date -u '+%F %T') UTC: start, snapshot hours < $CUT"
# 1. course repository at the pinned commit
[ "$(git -c safe.directory='*' -C assignment2 rev-parse HEAD)" = "$COURSE_COMMIT" ] || { echo "assignment2 is not at $COURSE_COMMIT"; exit 1; }
# 2. hour files (stops if any hour is missing; then rerun later with the same range)
./download_gharchive.sh "$RAW" 2025-12-01 "$CUT" | tee logs/f17_download.log
# 3. keep the 11-15 snapshot, then official preprocessing on exactly the hours < CUT
[ -e data_b15 ] || cp -a data data_b15
mkdir -p artifacts_b15
for f in dataset dataset_imp events final_features final_features_imp; do
  [ -e artifacts_b15/$f.parquet ] || cp -a artifacts/$f.parquet artifacts_b15/
done
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
$PY src/run_preprocess.py --raw "$SEL" --out data | tee logs/f17_preprocess.log
cd src
# 4. datasets at the cutoff and leakage audit
$PY build_dataset.py --jobs 16 --cutoff "$CUT" | tee ../logs/f17_build.log
$PY build_dataset.py --impute --jobs 16 --cutoff "$CUT" | tee ../logs/f17_build_imp.log
$PY leakage_audit.py --impute "2026-05-13 12:00" "2026-09-23 12:00" "2026-10-28 12:00" "2026-11-04 12:00" "2026-11-16 12:00" | tee ../logs/f17_leakage_audit.log
grep -q "LEAKAGE AUDIT PASSED" ../logs/f17_leakage_audit.log || { echo "leakage audit failed"; exit 1; }
cd ..
# 5. backtests of the three candidates
echo "== $(date -u '+%F %T') UTC: backtests"
PREFIX=f17 ./backtest_new.sh "$GA" "$GB" "$GC" "$FRESH"
# 6. the decision (stops unless all fresh weeks are mature and scored for every candidate)
cd src
DEV=2026-05-06,2026-05-27,2026-06-17,2026-07-08,2026-07-29,2026-08-19,2026-09-02,2026-09-09,2026-09-16,2026-09-23,2026-09-30,2026-10-07
$PY forward_decide.py --new --formal --prefix f17 --data-end "2026-11-17 12:00" --fresh "$FRESH" --weeks "$DEV" | tee ../logs/f17_decide.log
echo "== $(date -u '+%F %T') UTC: done"
