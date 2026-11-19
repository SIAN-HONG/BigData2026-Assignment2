"""Experiment runner: rolling-origin validation for one named configuration.

Every run appends one row to results/experiments.tsv and writes the per-week table to
results/exp_<name>.csv and the fold probabilities to artifacts/folds_<name>.pkl.

  python src/exp.py --name full --groups T0,T1,R1,R2
  python src/exp.py --name t0_only --groups T0
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import pandas as pd

from common import ART, RES
from features import feature_groups
from modeling import BACKEND, feature_cols, rolling_eval, summarize

VAL = [pd.Timestamp(f"{d} 12:00") for d in
       ["2026-09-02", "2026-09-09", "2026-09-16", "2026-09-23", "2026-09-30", "2026-10-07"]]

_DF = None


def load(path=ART / "dataset.parquet"):
    global _DF
    if _DF is None:
        _DF = pd.read_parquet(path)
    return _DF


def run(name, groups=("T0", "T1", "R1", "R2"), extra_feats=None, drop_feats=(), params=None,
        start=None, every=1, wednesday_only=False, seed=0, df=None, note="", val=None, censor=False):
    df = load() if df is None else df
    ev = pd.read_parquet(ART / "events.parquet") if censor else None
    allf = feature_cols(df)
    G = feature_groups(allf)
    feats = [f for g in groups for f in G[g]] + list(extra_feats or [])
    feats = [f for f in feats if f not in set(drop_feats)]
    t0 = time.time()
    print(f"[{name}] {len(feats)} features, groups={groups}", flush=True)
    folds = rolling_eval(df, feats, val or VAL, params=params, start=start, every=every,
                         wednesday_only=wednesday_only, seed=seed, log=lambda m: print(m, flush=True),
                         censor_ev=ev)
    S = summarize(folds)
    S["table"].to_csv(RES / f"exp_{name}.csv", index=False)
    pickle.dump(folds, open(ART / f"folds_{name}.pkl", "wb"))
    row = {"name": name, "groups": "+".join(groups), "n_feats": len(feats), "start": start, "every": every,
           "wed_only": wednesday_only, "seed": seed, "params": json.dumps(params or {}),
           "honest_score": round(S["honest_score"], 5), "honest_macro": round(S["honest_macro"], 5),
           "honest_micro": round(S["honest_micro"], 5), "oracle_score": round(S["oracle_score"], 5),
           "prevweek_th_score": round(S["prev_score"], 5),
           "minutes": round((time.time() - t0) / 60, 1), "note": note,
           "backend": (params or {}).get("backend", BACKEND)}
    p = RES / "experiments.tsv"
    pd.DataFrame([row]).to_csv(p, sep="\t", mode="a", header=not p.exists(), index=False)
    print(S["table"].round(4).to_string(), flush=True)
    print(json.dumps(row), flush=True)
    return S, folds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--groups", default="T0,T1,R1,R2")
    ap.add_argument("--start", default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--wed", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--note", default="")
    ap.add_argument("--data", default="dataset.parquet")
    ap.add_argument("--val", default=None, help="comma-separated validation dates (12:00 UTC); default VAL")
    ap.add_argument("--censor", action="store_true", help="censor training labels at each validation cutoff")
    a = ap.parse_args()
    val = [pd.Timestamp(f"{d} 12:00") for d in a.val.split(",")] if a.val else None
    note = a.note + (" censored" if a.censor else "") + (f" data={a.data}" if a.data != "dataset.parquet" else "")
    run(a.name, tuple(a.groups.split(",")), params=json.loads(a.params), start=a.start,
        every=a.every, wednesday_only=a.wed, seed=a.seed, note=note.strip(), df=load(ART / a.data),
        val=val, censor=a.censor)
