"""XGBoost grid on the S3 features, rolling refit on the last 6 labelled Saturdays (same folds as the backtest).

  CUDA_VISIBLE_DEVICES=5 ASG2_DB=../data_n19/db ASG2_ART=../artifacts_n19 ../.venv/bin/python grid_xgb.py
Grid (user request 2026-11-20): max_depth {3,5,7,9} x learning_rate {0.005,0.01,0.05} x n_estimators
{100,300,500,1000}; subsample 0.7, colsample_bytree 0.7; everything else as configs/final_config_s3.json
(min_child_weight 50, reg_lambda 1, max_bin 128, hist on GPU), seed 0, no MLP, no GNN.
For each validation Saturday V: train on cutoffs <= V - 14 d (pipeline.train_window), one model per activity,
predict V, joint plug-in decision, official score; all-six and outage-aware (pipeline.outage_targets).
n_estimators: a 1000-tree model is trained once per (depth, lr, fold, activity) and the first n trees are used
(iteration_range); boosting is sequential, so this equals training n trees (checked: --check).
Writes results/grid_xgb.csv (per config and week), results/grid_xgb_summary.csv and the probabilities
artifacts/grid_xgb_probs.pkl (for ensembling with the GNN later).
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from common import ART, RES, ROOT, TARGETS
from modeling import XGB_PARAMS, YCOLS, official_score, train_window
from pipeline import decide, outage_targets, val_weeks

DEPTHS = [3, 5, 7, 9]
LRS = [0.005, 0.01, 0.05]
NTREES = [100, 300, 500, 1000]


def params(depth, lr, n, seed=0):
    return {**XGB_PARAMS, "max_depth": depth, "learning_rate": lr, "n_estimators": n, "subsample": 0.7,
            "colsample_bytree": 0.7, "min_child_weight": 50, "random_state": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only verify that iteration_range == training n trees")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depths", type=int, nargs="+", default=DEPTHS)
    ap.add_argument("--lrs", type=float, nargs="+", default=LRS)
    ap.add_argument("--ntrees", type=int, nargs="+", default=NTREES)
    ap.add_argument("--name", default="grid_xgb", help="output name (results/<name>.csv, _summary.csv, artifacts/<name>_probs.pkl)")
    a = ap.parse_args()
    cfg = json.load(open(ROOT / "configs/final_config_s3.json"))
    fx = cfg["xgb_features"]
    df = pd.read_parquet(ART / cfg["dataset"])
    weeks = val_weeks("last6", df)
    if a.check:
        v = weeks[0]
        tr, va = train_window(df, v, every=1), df[df.cutoff == v]
        for c in ["open_pr", "push"]:
            m1 = xgb.XGBClassifier(**params(5, 0.05, 300)).fit(tr[fx], tr[f"y_{c}"])
            m2 = xgb.XGBClassifier(**params(5, 0.05, 1000)).fit(tr[fx], tr[f"y_{c}"])
            p1 = m1.predict_proba(va[fx])[:, 1]
            p2 = m2.predict_proba(va[fx], iteration_range=(0, 300))[:, 1]
            print(c, "max |300 trees - first 300 of 1000|", float(np.abs(p1 - p2).max()), flush=True)
        return
    ev = pd.read_parquet(ART / "events.parquet")
    drop = outage_targets(ev, weeks)
    rows, probs = [], {}
    t0 = time.time()
    for v in weeks:
        tr, va = train_window(df, v, every=1), df[df.cutoff == v]
        assert tr.cutoff.max() + pd.Timedelta(hours=4, days=12) <= v
        Y = va[YCOLS].to_numpy()
        wk = str(v.date())
        keep = [j for j, c in enumerate(TARGETS) if c not in drop[wk]]
        for d, lr in itertools.product(a.depths, a.lrs):
            P = {n: np.zeros((len(va), 6)) for n in a.ntrees}
            for j, c in enumerate(TARGETS):
                m = xgb.XGBClassifier(**params(d, lr, max(a.ntrees), a.seed)).fit(tr[fx], tr[f"y_{c}"])
                for n in a.ntrees:
                    P[n][:, j] = m.predict_proba(va[fx], iteration_range=(0, n))[:, 1]
            for n in a.ntrees:
                pred = decide(P[n], cfg)
                s = official_score(pred, Y)
                rows.append({"seed": a.seed, "depth": d, "lr": lr, "n_estimators": n, "val": wk, "score": s["score"],
                             "macro_f1": s["macro_f1"], "micro_f1": s["micro_f1"],
                             "outage_aware_score": official_score(pred[:, keep], Y[:, keep])["score"],
                             "dropped": ",".join(drop[wk])})
                probs[(d, lr, n, wk)] = P[n].astype(np.float32)
            print(f"{wk} depth {d} lr {lr}: " + ", ".join(
                f"n{r['n_estimators']} {100 * r['score']:.2f}/{100 * r['outage_aware_score']:.2f}" for r in rows[-len(a.ntrees):])
                + f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        pd.DataFrame(rows).to_csv(RES / f"{a.name}.csv", index=False)
    R = pd.DataFrame(rows)
    R.to_csv(RES / f"{a.name}.csv", index=False)
    S = (R.groupby(["depth", "lr", "n_estimators"])[["score", "outage_aware_score", "macro_f1", "micro_f1"]].mean() * 100)
    S["weeks"] = R.groupby(["depth", "lr", "n_estimators"]).size()
    S = S.sort_values("score", ascending=False)
    S.round(3).to_csv(RES / f"{a.name}_summary.csv")
    pickle.dump({"weeks": [str(w.date()) for w in weeks], "uids": {str(w.date()): df[df.cutoff == w].uid.to_numpy() for w in weeks},
                 "probs": probs}, open(ART / f"{a.name}_probs.pkl", "wb"))
    print(S.round(2).head(15).to_string())
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
