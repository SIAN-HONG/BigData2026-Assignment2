"""Phase 2 diagnostics of a saved fold-prediction file (default: the v1 baseline ensemble).

  python src/diagnose.py [folds_name]   -> results/phase2/diag_<name>_*.csv and printed summary

Per activity: precision / recall / F1 / prevalence / predicted-positive rate / FP / FN, calibration
(mean p vs prevalence, reliability bins), ranking quality (AUC, average precision), and the F1 that
the best possible top-k would reach with the same ranking (ranking ceiling).  Breakdowns by user
activity level and by week.
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import ART, RES, TARGETS
from modeling import EXCLUDE_WEEKS, joint_plugin_predict

OUT = RES / "phase2"
OUT.mkdir(exist_ok=True)


def prf(p, y):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return tp, fp, fn, pr, rc, f1


def best_k_f1(p, y):
    o = np.argsort(-p); ys = y[o]; tp = np.cumsum(ys); k = np.arange(1, len(ys) + 1)
    f1 = 2 * tp / (k + ys.sum()); i = int(np.argmax(f1))
    return f1[i], (i + 1) / len(ys)


def main(name="final_backtest_v1_baseline"):
    folds = pickle.load(open(ART / f"folds_{name}.pkl", "rb"))
    df = pd.read_parquet(ART / "dataset.parquet",
                         columns=["cutoff", "UserID", "c28d_push", "rec_any", "adays28_any", "n_repos28",
                                  "call_push", "c90d_open_pr", "c90d_issue_comment", "c90d_pr_review"])
    rows, grp, cal = [], [], []
    for v, P, Y, U in folds:
        wk = str(v.date())
        D = joint_plugin_predict(P)
        meta = df[df.cutoff == v].set_index("UserID").loc[U]
        act = meta.adays28_any.to_numpy()
        bucket = pd.cut(act, [-1, 0, 2, 7, 14, 28], labels=["0d", "1-2d", "3-7d", "8-14d", "15-28d"]).astype(str)
        for j, c in enumerate(TARGETS):
            y, p, d = Y[:, j], P[:, j], D[:, j]
            tp, fp, fn, pr, rc, f1 = prf(d, y)
            bf1, brate = best_k_f1(p, y)
            rows.append(dict(week=wk, target=c, n=len(y), prevalence=y.mean(), pred_rate=d.mean(), mean_p=p.mean(),
                             tp=tp, fp=fp, fn=fn, precision=pr, recall=rc, f1=f1,
                             auc=roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
                             ap=average_precision_score(y, p) if y.sum() else np.nan,
                             best_topk_f1=bf1, best_topk_rate=brate))
            for b in np.unique(bucket):
                m = bucket == b
                tp, fp, fn, pr, rc, f1 = prf(d[m], y[m])
                grp.append(dict(week=wk, target=c, bucket=b, n=int(m.sum()), prevalence=y[m].mean(),
                                pred_rate=d[m].mean(), tp=tp, fp=fp, fn=fn))
            q = pd.qcut(p, 10, labels=False, duplicates="drop")
            for b in np.unique(q):
                m = q == b
                cal.append(dict(week=wk, target=c, bin=int(b), mean_p=p[m].mean(), obs=y[m].mean(), n=int(m.sum())))
    R = pd.DataFrame(rows); G = pd.DataFrame(grp); C = pd.DataFrame(cal)
    R.to_csv(OUT / f"diag_{name}_per_week_target.csv", index=False)
    G.to_csv(OUT / f"diag_{name}_by_activity_bucket.csv", index=False)
    C.to_csv(OUT / f"diag_{name}_calibration.csv", index=False)
    E = R[~R.week.isin(EXCLUDE_WEEKS)]
    pd.set_option("display.width", 220)
    agg = E.groupby("target")[["prevalence", "pred_rate", "mean_p", "precision", "recall", "f1", "auc", "ap",
                               "best_topk_f1", "best_topk_rate", "fp", "fn"]].mean()
    print("== mean over 5 weeks, per target\n", agg.round(4).to_string())
    print("\n== F1 per week\n", R.pivot(index="week", columns="target", values="f1").round(4).to_string())
    print("\n== prevalence / pred_rate / mean_p per week\n",
          R.pivot(index="week", columns="target", values="prevalence").round(3).to_string(), "\n",
          R.pivot(index="week", columns="target", values="pred_rate").round(3).to_string(), "\n",
          R.pivot(index="week", columns="target", values="mean_p").round(3).to_string())
    g = G[~G.week.isin(EXCLUDE_WEEKS)].groupby(["target", "bucket"])[["n", "tp", "fp", "fn"]].sum()
    g["prev"] = (g.tp + g.fn) / g.n; g["pred"] = (g.tp + g.fp) / g.n
    g["f1"] = 2 * g.tp / (2 * g.tp + g.fp + g.fn).clip(lower=1)
    g["share_of_fn"] = g.fn / g.groupby(level=0).fn.transform("sum")
    print("\n== by active days in last 28d (pooled 5 weeks)\n", g.round(3).to_string())
    return R, G, C


if __name__ == "__main__":
    main(*sys.argv[1:])
