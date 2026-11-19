"""Descriptive numbers used on the slides, computed from the current artifacts (one snapshot), with
their definitions, so that slides/build_slides.py types none of them by hand.

  python src/slide_stats.py [--censor]      # --censor adds the late-label check (a few minutes)
Writes results/slide_stats.json.

- test_users: rows of assignment2/submission/test_input.csv
- pos_rate_{early,late}: positive rate per activity over the user-weeks of the Wednesday cutoffs
  2026-01-07 .. 06-24 (early) and 08-05 .. 10-28 (late) in dataset_imp.parquet (labels from raw events)
- outage_days: UTC days from 2025-12-01 to the last complete day flagged by common.outage_flags on the
  daily counts of all users' events (raw event log), per activity
- events: rows of artifacts/events.parquet (derived event rows, 6 target + 7 relational kinds)
- dataset_rows / dataset_cutoffs / last_cutoff: dataset_imp.parquet
- n_feats: feature_groups of the dataset columns, groups T0 / T1 / R1 / R2
- censor_changed (with --censor): per validation week V, training rows (every 3rd cutoff <= V - 8 d)
  whose six labels change when the labels are censored at V (modeling.censor_labels)
- snapshot: outage-aware week scores of the same code on the 11-07 (forward_check_a6.json) and 11-15
  (forward_check_b15.json) snapshots; per version the largest |change| over the weeks both contain
  (09-02 left out), and the change of each candidate's difference to v1
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import ART, RES, ROOT, TARGETS, outage_flags
from features import feature_groups
from modeling import YCOLS, censor_labels, feature_cols, train_window

VAL8 = ["2026-09-09", "2026-09-16", "2026-09-23", "2026-09-30", "2026-10-07", "2026-10-14", "2026-10-21", "2026-10-28"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--censor", action="store_true")
    a = ap.parse_args()
    out = {}
    out["test_users"] = int(len(pd.read_csv(ROOT / "assignment2/submission/test_input.csv")))
    df = pd.read_parquet(ART / "dataset_imp.parquet")
    out["dataset_rows"] = int(len(df))
    out["dataset_cutoffs"] = int(df.cutoff.nunique())
    out["first_cutoff"], out["last_cutoff"] = str(df.cutoff.min()), str(df.cutoff.max())
    G = feature_groups(feature_cols(df))
    out["n_feats"] = {g: len(G[g]) for g in ("T0", "T1", "R1", "R2")}
    wed = df[df.cutoff.dt.dayofweek == 2]
    early = wed[(wed.cutoff >= "2026-01-07") & (wed.cutoff <= "2026-06-24 23:59")]
    late = wed[(wed.cutoff >= "2026-08-05") & (wed.cutoff <= "2026-10-28 23:59")]
    out["pos_rate_early"] = {k: 100 * float(early[f"y_{k}"].mean()) for k in TARGETS}
    out["pos_rate_late"] = {k: 100 * float(late[f"y_{k}"].mean()) for k in TARGETS}
    out["pos_rate_def"] = ("user-weeks of Wednesday cutoffs 01-07..06-24 (early) vs 08-05..10-28 (late), "
                           "dataset_imp.parquet")
    ev = pd.read_parquet(ART / "events.parquet")
    out["events"] = int(len(ev))
    e = ev[ev.kind < 6]
    day = pd.to_datetime(e.ts, unit="s").dt.floor("D")
    end = pd.to_datetime(ev.avail.max(), unit="s").floor("D")
    c = pd.crosstab(day, e.kind)
    c = c.reindex(pd.date_range(c.index.min(), end - pd.Timedelta(days=1)), fill_value=0)
    flags = c.apply(lambda col: pd.Series(outage_flags(col.to_numpy())[0], index=col.index))
    flags = flags[flags.index >= "2025-12-01"]
    out["outage_days"] = {TARGETS[k]: int(flags[k].sum()) for k in range(6)}
    out["outage_last_day"] = str((end - pd.Timedelta(days=1)).date())
    pushd = flags.index[flags[5]]
    out["push_outage_days_since_1027"] = int((pushd >= "2026-10-27").sum())
    if a.censor:
        out["censor_changed"] = {}
        out["censor_train_rows"] = {}
        for d in VAL8:
            v = pd.Timestamp(f"{d} 12:00")
            tr = train_window(df, v, every=3)
            cz = censor_labels(tr, ev, v)
            out["censor_changed"][d] = int((tr[YCOLS].to_numpy() != cz[YCOLS].to_numpy()).any(1).sum())
            out["censor_train_rows"][d] = int(len(tr))
    old, new = json.load(open(RES / "forward_check_a6.json")), json.load(open(RES / "forward_check_b15.json"))
    snap = {}
    for n in ("v1", "p3", "p4"):
        ws = [w for w in old["scores"][n] if w in new["scores"][n] and w != "2026-09-02"]
        d = [new["scores"][n][w] - old["scores"][n][w] for w in ws]
        snap[n] = {"weeks": len(ws), "max_abs_change": float(np.max(np.abs(d))), "mean_change": float(np.mean(d))}
        if n != "v1":
            dd = [(new["scores"][n][w] - new["scores"]["v1"][w]) - (old["scores"][n][w] - old["scores"]["v1"][w]) for w in ws]
            snap[n]["max_abs_change_of_delta_vs_v1"] = float(np.max(np.abs(dd)))
    out["snapshot"] = snap
    out["snapshot_def"] = "same code, outage-aware week scores, 11-07 vs 11-15 snapshot, weeks in both except 09-02"
    import hashlib, time
    out["inputs"] = {f: {"mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime((ART / f).stat().st_mtime)),
                         "sha256_first_1MB": hashlib.sha256(open(ART / f, "rb").read(1 << 20)).hexdigest()}
                     for f in ("dataset_imp.parquet", "events.parquet")}
    out["data_end"] = str(pd.Timestamp(ev.avail.max(), unit="s"))
    (RES / "slide_stats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
