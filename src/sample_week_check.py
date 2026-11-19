"""Rehearsal on the official example week (sample_truth.csv: 2026-09-27 13:00 -> 10-04 13:00 UTC).

Features are built at t = 2026-09-27 13:00 UTC for the 4,785 example users; the models are
trained only on cutoffs <= t - 8 days, i.e. on labels that were complete before t.  The
output CSV is checked and scored with the official evaluate.py.  This week overlaps the
09-23 / 09-30 validation folds, so it is a format and pipeline check, not an independent
held-out estimate.
  CUDA_VISIBLE_DEVICES=0 python src/sample_week_check.py
"""
import subprocess
import sys

import numpy as np
import pandas as pd

import build_dataset as B
from common import ART, RES, ROOT, TARGETS, s
from pipeline import decide, fit_predict, load_cfg

if __name__ == "__main__":
    B._init()
    truth = pd.read_csv(ROOT / "assignment2/evaluation/examples/sample_truth.csv")
    u = B._G["T"]["Users"]
    uid = pd.Series(u.user_id.values, index=u.github_user_id.values).loc[truth.UserID].to_numpy()
    t = s("2026-09-27 13:00")
    rows = B.make_sample(t, uids=uid, with_labels=False)
    df = pd.read_parquet(ART / "dataset.parquet")
    P, _ = fit_predict(df, B._G["ev"], pd.Timestamp(t, unit="s"), rows, load_cfg())
    pred = decide(P, load_cfg())
    out = pd.DataFrame(pred, columns=TARGETS)
    out.insert(0, "UserID", rows.UserID.to_numpy())
    path = RES / "sample_week_output.csv"
    out.to_csv(path, index=False)
    r = subprocess.run([sys.executable, str(ROOT / "assignment2/evaluation/evaluate.py"), "--pred", str(path),
                        "--truth", str(ROOT / "assignment2/evaluation/examples/sample_truth.csv")],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr)
    (RES / "sample_week_eval.txt").write_text(r.stdout + r.stderr)
