"""What if pushes are missing just before the prediction cutoff?

The course announced (10-31) that some GH Archive data is lost for good, e.g. pushes on several October
days (10-27 .. 10-30 at least).  If that lasts until the 11-11 cutoff, every push feature at prediction
time is computed from a hole, and the push features also feed the other five activities.

Simulation on a past validation week V: delete the pushes with created_at in [V - N days, V), rebuild
the dataset rows of every training cutoff whose features or labels touch that hole (and of V itself),
and run the unchanged method (pipeline.fit_predict) on it.  Compare with the same run on intact data.
Push is left out of the score (the course drops an activity with an outage in the test week), the
6-activity score is reported as well.

  python src/sim_push_outage.py --weeks 2026-09-23 --ns 3,7,14 --config configs/final_config_p3.json
  (a config with dataset_imp.parquet rebuilds the damaged rows with outage imputation as well)
Output: results/sim_push_outage_<tag>.csv
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import build_dataset as B
from common import ART, DAY, RES, TARGETS, build_events, load_tables, s
from features import FeatureBuilder
from modeling import joint_plugin_predict, official_score, train_window
from pipeline import fit_predict, load_cfg

PUSH = TARGETS.index("push")


def with_hole(T, ev0, a, b):
    """Tables and events with the pushes created in [a, b) removed (seconds)."""
    from common import ts2s
    pu = T["Pushes"]
    t = ts2s(pu.created_at)
    T2 = {**T, "Pushes": pu[~((t >= a) & (t < b))].reset_index(drop=True)}
    ev2 = build_events(T2)
    return T2, ev2


def rebuild(df, T2, ev2, data_end, cuts, impute=False):
    """Replace the rows of the given cutoffs by rows built from the damaged data (same users)."""
    B._G.update(T=T2, ev=ev2, fb=FeatureBuilder(T2, ev2, impute=impute), data_end=data_end)
    parts = []
    for c in cuts:
        old = df[df.cutoff == c]
        new = B.make_sample(s(c), uids=old.uid.to_numpy())
        parts.append(new[old.columns])
    return pd.concat([df[~df.cutoff.isin(cuts)], *parts], ignore_index=True)


def score(P, Y):
    d = joint_plugin_predict(P)
    keep = [j for j in range(6) if j != PUSH]
    return official_score(d, Y)["score"] * 100, official_score(d[:, keep], Y[:, keep])["score"] * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", required=True)
    ap.add_argument("--ns", default="3,7,14")
    ap.add_argument("--config", default="configs/final_config_p3.json")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    df = pd.read_parquet(ART / cfg["dataset"])
    ev = pd.read_parquet(ART / "events.parquet")
    T = load_tables()
    B._init()
    data_end = B._G["data_end"]
    rows = []
    for w in a.weeks.split(","):
        V = pd.Timestamp(f"{w} 12:00")
        va = df[df.cutoff == V]
        Y = va[[f"y_{c}" for c in TARGETS]].to_numpy()
        t0 = time.time()
        P0, _ = fit_predict(df, ev, V, va, cfg, log=lambda *x, **k: None)
        s6, s5 = score(P0, Y)
        rows.append({"week": w, "hole_days": 0, "S6": s6, "S5": s5, "minutes": (time.time() - t0) / 60})
        print(rows[-1], flush=True)
        used = sorted(train_window(df, V, every=cfg["every"]).cutoff.unique())
        for n in map(int, a.ns.split(",")):
            t0 = time.time()
            a_s, b_s = s(V) - n * DAY, s(V)
            T2, ev2 = with_hole(T, ev, a_s, b_s)
            # features see the hole if c > V - n; labels (window + 5-day slack) if c > V - n - 12 d
            cuts = [c for c in used if c > V - pd.Timedelta(days=n + 12)] + [V]
            df2 = rebuild(df, T2, ev2, data_end, cuts, impute="_imp" in cfg["dataset"])
            va2 = df2[df2.cutoff == V].set_index("uid").loc[va.uid].reset_index()
            P, _ = fit_predict(df2, ev2, V, va2, cfg, log=lambda *x, **k: None)
            s6, s5 = score(P, Y)                      # true labels from the intact data
            rows.append({"week": w, "hole_days": n, "S6": s6, "S5": s5,
                         "mean_p": dict(zip(TARGETS, (P.mean(0) / P0.mean(0)).round(3).tolist())),
                         "rebuilt_cutoffs": len(cuts), "minutes": (time.time() - t0) / 60})
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(RES / f"sim_push_outage{a.tag}.csv", index=False)


if __name__ == "__main__":
    main()
