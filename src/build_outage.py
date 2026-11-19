"""Build one extra feature group (default O, outage-corrected temporal features; also R4) for the
rows of an existing dataset.

Group O is computed with the same FeatureBuilder and the same point-in-time rule as the other
groups, for exactly the (cutoff, uid) rows of --data, and written to a separate parquet that
exp2.py merges on (cutoff, uid).  Also writes the detected outage days per cutoff for the report.

  python src/build_outage.py --data ../artifacts/dataset_p2.parquet --out ../artifacts/outage_O.parquet
  python src/build_outage.py --group R4 --out ../artifacts/group_R4.parquet
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import get_context

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import build_dataset as bd
from common import ART, KIND_NAMES, RES
from features import outage_intervals

_KEYS = None
_GROUP = "O"


def _work(t):
    t0 = time.time()
    uids = _KEYS[t]
    X = bd._G["fb"].build(t, uids, groups=(_GROUP,))
    X.insert(0, "cutoff", pd.Timestamp(t, unit="s"))
    n_av = np.searchsorted(bd._G["fb"].ev_avail, t, side="left")
    past = bd._G["ev"].iloc[:n_av]
    iv = outage_intervals(past[past.ts < t], t)
    days = {KIND_NAMES[k]: [str(pd.Timestamp(a, unit="s").date()) for a, _ in v if a >= t - 90 * 86400]
            for k, v in iv.items()}
    print(f"  {pd.Timestamp(t, unit='s')}  {time.time() - t0:.1f}s  {_GROUP}", flush=True)
    return X.reset_index(), {"cutoff": str(pd.Timestamp(t, unit="s")), **{k: " ".join(v) for k, v in days.items()}}


def main():
    global _KEYS, _GROUP
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="O", help="feature group to build (O, R4, ...)")
    ap.add_argument("--data", default=str(ART / "dataset_p2.parquet"))
    ap.add_argument("--out", default=str(ART / "outage_O.parquet"))
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    _GROUP = a.group
    keys = pd.read_parquet(a.data, columns=["cutoff", "uid"])
    _KEYS = {int(c.value // 10**9): g.uid.to_numpy() for c, g in keys.groupby("cutoff")}
    bd._init()
    with get_context("fork").Pool(a.jobs) as pool:
        parts = pool.map(_work, sorted(_KEYS), chunksize=1)
    df = pd.concat([p[0] for p in parts], ignore_index=True)
    df.to_parquet(a.out)
    if _GROUP == "O":
        pd.DataFrame([p[1] for p in parts]).to_csv(RES / "phase2" / "outage_days_by_cutoff.csv", index=False)
    print("group", _GROUP, df.shape, "->", a.out)


if __name__ == "__main__":
    main()
