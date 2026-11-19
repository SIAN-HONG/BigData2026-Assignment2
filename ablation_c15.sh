#!/usr/bin/env bash
# Corrected feature-group ablation (round 09 consensus with the reviewer), for the Q2 / Q3 slides only;
# it does not touch the version selection of docs/FORWARD_CHECK.md addendum 7.
#  - training labels censored at each validation cutoff (exp.py --censor), as in pipeline.fit_predict
#  - outage-imputed dataset_imp.parquet (11-15 snapshot), every 3rd cutoff
#  - 8 mature Wednesdays 09-09 .. 10-28; fixed v1 tree parameters (depth 10, min_child_weight 50)
#  - 5 groups x 3 seeds, XGBoost on CPU (no idle GPU on 2026-11-15; all arms use the same backend)
# Scored afterwards by src/ablation_report.py (outage-aware, joint plug-in, seeds averaged).
# Usage: ./ablation_c15.sh
set -euo pipefail
cd "$(dirname "$0")/src"; PY=../.venv/bin/python
VAL=2026-09-09,2026-09-16,2026-09-23,2026-09-30,2026-10-07,2026-10-14,2026-10-21,2026-10-28
P='{"max_depth": 10, "min_child_weight": 50, "device": "cpu", "n_jobs": 48}'
stream() {
  s=$1
  for g in T0 T0,T1 T0,T1,R1 T0,T1,R1,R2 T0,R1,R2; do
    n=c15_$(echo $g | tr -d ,)_s$s
    $PY exp.py --name $n --groups $g --seed $s --every 3 --censor --data dataset_imp.parquet --val $VAL \
       --params "$P" > ../logs/$n.log 2>&1
    echo "$n done $(date +%H:%M)"
  done
}
stream 0 & stream 1 & stream 2 & wait
echo "all done $(date +%H:%M)"
