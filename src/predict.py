"""Official prediction for 2026-11-21 16:00 -> 11-28 16:00 UTC (11/22-11/29 Taiwan).

  python src/predict.py --student-id <ID>

Uses artifacts/dataset.parquet (training samples) and artifacts/final_features.parquet
(features of all cohort users at the latest release cutoff), both written by
build_dataset.py from the tables in data/db (preprocess.py on GH Archive).  The model and decision rule are
pipeline.fit_predict / pipeline.decide - the exact code path of the backtest.

If assignment2/submission/test_input.csv is not released yet, the official selection rule
is applied to write a REHEARSAL file (output_<ID>_rehearsal.csv) - not a submission.
The output is always checked with the official evaluate.py; a failed check exits non-zero.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from common import ART, DAY, FINAL_CUTOFF, RES, ROOT, TARGETS, build_events, eligible_users, load_tables, s
from pipeline import decide, fit_predict, load_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True)
    ap.add_argument("--config", default=None, help="config json relative to the project root (default final_config.json)")
    ap.add_argument("--cutoff", default=None,
                    help="expected cutoff (UTC); stop if final_features was built at another one")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    df = pd.read_parquet(ART / cfg["dataset"])
    ev = pd.read_parquet(ART / "events.parquet")
    fin = pd.read_parquet(ART / cfg["dataset"].replace("dataset", "final_features"))
    cut = fin.cutoff.iloc[0]
    if a.cutoff is not None and cut != pd.Timestamp(a.cutoff):
        sys.exit(f"ERROR: final features are at {cut}, expected --cutoff {pd.Timestamp(a.cutoff)}")
    print(f"cutoff (data end) {cut} UTC; target window {cut + pd.Timedelta(hours=4)} .. "
          f"{cut + pd.Timedelta(hours=4, days=7)} UTC")
    if cut != FINAL_CUTOFF:
        print(f"NOTE: cutoff is not {FINAL_CUTOFF} - result is a rehearsal of the pipeline")

    test_path = ROOT / "assignment2" / "submission" / "test_input.csv"
    if test_path.exists():
        users = pd.read_csv(test_path).UserID.to_numpy()
        out_path = ROOT / f"output_{a.student_id}.csv"
        if cut != FINAL_CUTOFF:
            out_path = ROOT / f"output_{a.student_id}_rehearsal_{cut:%m%d}.csv"
    else:
        T = load_tables()
        u = T["Users"]
        el = eligible_users(build_events(T), u.user_id[u.is_cohort].to_numpy(), s(cut))
        users = u.github_user_id.to_numpy()[el]
        tag = "" if a.config is None else "_" + Path(a.config).stem
        out_path = ROOT / f"output_{a.student_id}_rehearsal{tag}.csv"
        print("test_input.csv not released -> rehearsal users from the official selection rule")

    rows = fin.set_index("UserID").reindex(users)
    miss = rows.uid.isna()
    if miss.any():
        print(f"WARNING: {miss.sum()} test users are not cohort users of this release -> zero features")
        rows = rows.fillna(0)
    rows = rows.reset_index()
    rows["uid"] = rows.uid.astype(int)
    P, _ = fit_predict(df, ev, cut, rows, cfg)
    pred = decide(P, cfg)

    out = pd.DataFrame(pred, columns=TARGETS)
    out.insert(0, "UserID", users.astype("int64"))
    out.to_csv(out_path, index=False)
    pd.DataFrame(P, columns=TARGETS).assign(UserID=users).to_csv(RES / "final_probabilities.csv", index=False)
    info = {"cutoff": str(cut), "n_users": int(len(users)), "output": out_path.name, "cfg": cfg,
            "predicted_positive_rate": dict(zip(TARGETS, pred.mean(0).round(4).tolist())),
            "mean_probability": dict(zip(TARGETS, P.mean(0).round(4).tolist()))}
    json.dump(info, open(RES / "final_prediction_info.json", "w"), indent=2)
    print(json.dumps(info["predicted_positive_rate"]))
    print("wrote", out_path)

    cmd = [sys.executable, str(ROOT / "assignment2/scripts/evaluate.py"), "--pred", str(out_path)]
    if test_path.exists():
        cmd += ["--input", str(test_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip())
    if r.returncode != 0:
        sys.exit("OFFICIAL FORMAT CHECK FAILED")


if __name__ == "__main__":
    main()
