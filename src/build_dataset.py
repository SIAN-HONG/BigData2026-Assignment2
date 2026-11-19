"""Build the supervised (user, cutoff) samples.

A sample mirrors the official setting exactly:
  * cutoff t    = last data timestamp available (12:00 UTC, i.e. 20:00 Taiwan release)
  * users       = cohort users with >= 3 activity records in the 90 days before t - 17 days
                  (official test list is selected on 11-04 12:00 UTC for the 11-21 12:00 cutoff)
  * features    = only rows with availability time < t
  * labels      = activity in [t + 4h, t + 4h + 7d)   (official: 11-21 16:00 -> 11-28 16:00 UTC)

Usage:  python src/build_dataset.py [--start 2026-01-07] [--every 1] [--jobs 16]
Writes artifacts/dataset.parquet (one row per user x cutoff) and, for inference,
artifacts/final_features.parquet built at the latest available cutoff.
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import get_context

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from common import (ART, DAY, GAP_SEC, TARGETS, build_events, eligible_users, label_matrix,
                    load_tables, s)
from features import FeatureBuilder

_G = {}


def _init():
    T = load_tables()
    ev = build_events(T)
    _G["T"], _G["ev"] = T, ev
    _G["fb"] = FeatureBuilder(T, ev)
    u = T["Users"]
    _G["cohort"] = u.user_id[u.is_cohort].to_numpy()
    _G["gid"] = u.github_user_id.to_numpy()
    _G["data_end"] = int(ev.avail.max()) + 1


def make_sample(t: int, uids=None, with_labels=True) -> pd.DataFrame:
    ev, fb = _G["ev"], _G["fb"]
    if uids is None:
        uids = eligible_users(ev, _G["cohort"], t)
    X = fb.build(t, uids).copy()
    X.insert(0, "cutoff", pd.Timestamp(t, unit="s"))
    X.insert(1, "UserID", _G["gid"][uids])
    if with_labels:
        start, end = t + GAP_SEC, t + GAP_SEC + 7 * DAY
        slack = max(0, min(5 * DAY, _G["data_end"] - end))
        Y = label_matrix(ev, uids, start, end, slack=slack)
        for i, c in enumerate(TARGETS):
            X[f"y_{c}"] = Y[:, i]
    return X.reset_index()


def _work(t):
    t0 = time.time()
    df = make_sample(t)
    print(f"  cutoff {pd.Timestamp(t, unit='s')}  users={len(df)}  {time.time() - t0:.1f}s", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-07 12:00")
    ap.add_argument("--every", type=int, default=1, help="days between cutoffs")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", default=str(ART / "dataset.parquet"))
    ap.add_argument("--final-out", default=str(ART / "final_features.parquet"))
    ap.add_argument("--impute", action="store_true",
                    help="fill collection outages first (features.impute_outages); default outputs *_imp.parquet")
    ap.add_argument("--cutoff", default=None,
                    help="prediction cutoff (UTC, e.g. 2026-11-21T12:00) for final_features; the data must reach "
                         "it (stop otherwise). Default: the latest 12:00 UTC <= data end (+1 h tolerance)")
    args = ap.parse_args()
    if args.impute:
        args.out = args.out.replace("dataset.parquet", "dataset_imp.parquet")
        args.final_out = args.final_out.replace("final_features.parquet", "final_features_imp.parquet")
    _init()
    _G["fb"].impute = args.impute
    data_end = _G["data_end"]
    print("data end:", pd.Timestamp(data_end, unit="s"))
    # features for the newest possible cutoff (the official one: 11-21 12:00 UTC)
    # releases contain data up to 12:00 UTC; take the latest 12:00 UTC <= data end (+1h tolerance)
    last = (data_end + 3600 - 12 * 3600) // DAY * DAY + 12 * 3600
    if args.cutoff is not None:
        want = s(args.cutoff)
        if want != last:  # e.g. the last hour file was missing -> the default would fall back one day
            raise SystemExit(f"ERROR: --cutoff {pd.Timestamp(want, unit='s')} but the data end "
                             f"{pd.Timestamp(data_end, unit='s')} gives {pd.Timestamp(last, unit='s')}")
    # checked before anything is written
    # raw (not imputed) event log, the same with or without --impute; always rewritten so it matches data/db
    _G["ev"].to_parquet(ART / "events.parquet")  # used by predict/pipeline (GNN edges, label censoring) and figures
    # every cutoff at 12:00 UTC whose 7-day label window ends before the data end
    cut = []
    t = s(args.start)
    while t + GAP_SEC + 7 * DAY <= data_end:
        cut.append(t)
        t += args.every * DAY
    print(f"{len(cut)} cutoffs: {pd.Timestamp(cut[0], unit='s')} .. {pd.Timestamp(cut[-1], unit='s')}")
    with get_context("fork").Pool(args.jobs) as pool:
        parts = pool.map(_work, cut, chunksize=1)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(args.out)
    print("dataset", df.shape, "->", args.out)
    fin = make_sample(last, uids=np.array(sorted(set(_G["cohort"]))), with_labels=False)
    fin.to_parquet(args.final_out)
    print("final features at", pd.Timestamp(last, unit="s"), fin.shape)


if __name__ == "__main__":
    main()
