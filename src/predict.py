"""Official prediction for 2026-11-21 16:00 -> 11-28 16:00 UTC (11/22-11/29 Taiwan).

  python src/predict.py --student-id <ID>

Uses artifacts/dataset.parquet (training samples) and artifacts/final_features.parquet
(features of all cohort users at the latest release cutoff), both written by
build_dataset.py from the tables in data/db (preprocess.py on GH Archive).  The model and decision rule are
pipeline.fit_predict / pipeline.decide - the exact code path of the backtest.

Official mode = the final features are at FINAL_CUTOFF (2026-11-21 12:00 UTC).  In official mode the run
stops with an error (never a silent fallback) if assignment2/submission/test_input.csv is missing or
malformed, differs from the pinned course file (sha256, 4,465 users), or a test user has no features.
Only at an earlier cutoff (rehearsal) may the users come from the official selection rule when the file
is missing; that output is named output_<ID>_rehearsal*.csv and is not a submission.
Every output is checked here (6 activity columns, 0/1 only, same users in the same order as the test
list) and with the official evaluate.py; a failed check exits non-zero.
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

# assignment2/submission/test_input.csv of the pinned course commit af9f335 (run_final.sh COURSE_COMMIT)
OFFICIAL_TEST_SHA256 = "3bc4f0e2250f44bad45dfcd6dbc9517e0dd4aa5959d9ec11540859e052931f8e"
OFFICIAL_TEST_USERS = 4465


def read_test_users(path: Path, official: bool) -> np.ndarray:
    """UserIDs of test_input.csv in file order; exits with an error if the file is malformed (any mode)
    or, in official mode, if it is not the pinned course file."""
    import hashlib
    if not path.exists():
        sys.exit(f"ERROR: {path} not found")
    try:
        d = pd.read_csv(path, dtype=str)
    except Exception as e:
        sys.exit(f"ERROR: cannot read {path}: {e}")
    if list(d.columns) != ["UserID"]:
        sys.exit(f"ERROR: {path} must have exactly one column 'UserID', got {list(d.columns)}")
    u = d.UserID
    if u.isna().any() or not u.str.fullmatch(r"\d+").all():
        sys.exit(f"ERROR: {path} has empty or non-integer UserIDs")
    if u.duplicated().any():
        sys.exit(f"ERROR: {path} has duplicate UserIDs")
    if official:
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        if len(u) != OFFICIAL_TEST_USERS or h != OFFICIAL_TEST_SHA256:
            sys.exit(f"ERROR: {path} is not the pinned course file ({len(u)} users, sha256 {h[:12]}...; expected "
                     f"{OFFICIAL_TEST_USERS} users, {OFFICIAL_TEST_SHA256[:12]}...). Check the course commit.")
    return u.astype("int64").to_numpy()


def check_output(path: Path, users: np.ndarray):
    """Re-read the written file: exact header, 0/1 in all six activity columns, same users in the same order."""
    o = pd.read_csv(path, dtype=str)
    if list(o.columns) != ["UserID"] + TARGETS:
        sys.exit(f"ERROR: output header {list(o.columns)}")
    if o.isna().any().any() or not o[TARGETS].isin(["0", "1"]).all().all():
        sys.exit("ERROR: output has empty cells or values other than 0/1")
    if not (o.UserID.astype("int64").to_numpy() == users).all() or len(o) != len(users):
        sys.exit("ERROR: output users differ from the test list or are in another order")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True)
    ap.add_argument("--config", default=None, help="config json relative to the project root (default: the submitted configuration, configs/final_config_xgb_d3.json)")
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

    official = cut == FINAL_CUTOFF
    test_path = ROOT / "assignment2" / "submission" / "test_input.csv"
    if official and not test_path.exists():
        sys.exit(f"ERROR: official run but {test_path} is missing - no fallback to the selection rule")
    if test_path.exists():
        users = read_test_users(test_path, official)
        out_path = ROOT / f"output_{a.student_id}.csv"
        if not official:
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
    if miss.any() and official:
        sys.exit(f"ERROR: {miss.sum()} test users have no features at the official cutoff")
    if miss.any():
        print(f"WARNING (rehearsal only): {miss.sum()} users are not cohort users of this release -> zero features")
        rows = rows.fillna(0)
    rows = rows.reset_index()
    rows["uid"] = rows.uid.astype(int)
    P, _ = fit_predict(df, ev, cut, rows, cfg)
    pred = decide(P, cfg)
    if pred.shape != (len(users), len(TARGETS)) or not np.isin(pred, [0, 1]).all():
        sys.exit(f"ERROR: predictions have shape {pred.shape} or non-binary values")

    out = pd.DataFrame(pred, columns=TARGETS)
    out.insert(0, "UserID", users.astype("int64"))
    out.to_csv(out_path, index=False)
    check_output(out_path, users.astype("int64"))
    pd.DataFrame(P, columns=TARGETS).assign(UserID=users).to_csv(RES / "final_probabilities.csv", index=False)
    info = {"cutoff": str(cut), "official": bool(official), "n_users": int(len(users)), "output": out_path.name, "cfg": cfg,
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
