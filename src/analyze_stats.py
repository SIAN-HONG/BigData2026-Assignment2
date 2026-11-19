"""Statistical feature diagnostics (Phase 3b): why new features add nothing, and which existing
features are unstable.

For the validation weeks 09-09 and 10-07 (same training window as the backtest, v1 XGB params,
groups T0+T1+R1+R2+R4):
  1. SHAP importance (exact TreeSHAP, xgboost pred_contribs) per activity on the validation rows,
     per feature and per group; rank stability between the two weeks (Spearman)
  2. redundancy: R^2 of each R4 feature predicted from the existing T0+T1+R1+R2 features
     (XGBoost regressor fit on training rows, R^2 on the validation rows)
  3. drift: population stability index (PSI) of every feature, training window vs validation week
Outputs: results/analysis/S_*.csv

  CUDA_VISIBLE_DEVICES=<g> python src/analyze_stats.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from common import RES, TARGETS
from exp2 import EV, load
from features import feature_groups
from modeling import XGB_PARAMS, censor_labels, feature_cols, train_window

OUT = RES / "analysis"
WEEKS = [pd.Timestamp("2026-09-09 12:00"), pd.Timestamp("2026-10-07 12:00")]
PARAMS = {**XGB_PARAMS, "max_depth": 10, "min_child_weight": 50, "random_state": 0}
BASE = ["T0", "T1", "R1", "R2"]


def psi(a, b, bins=10):
    """PSI of b (validation) against a (training), quantile bins of a; constant / tied values merge."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(b, a[0]) else np.inf
    pa = np.histogram(np.clip(a, edges[0], edges[-1]), edges)[0] / len(a)
    pb = np.histogram(np.clip(b, edges[0], edges[-1]), edges)[0] / len(b)
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(((pb - pa) * np.log(pb / pa)).sum())


def main():
    df = load()
    G = feature_groups(feature_cols(df))
    base = [f for g in BASE for f in G[g]]
    feats = base + G["R4"]
    grp = {f: g for g in BASE + ["R4"] for f in G[g]}
    imp_rows, red_rows, psi_rows = [], [], []
    for v in WEEKS:
        tr = censor_labels(train_window(df, v, every=3), EV, v)
        te = df[df.cutoff == v]
        print(v.date(), "train", len(tr), "val", len(te), flush=True)
        for c in TARGETS:
            m = xgb.XGBClassifier(**PARAMS).fit(tr[feats], tr[f"y_{c}"])
            sh = m.get_booster().predict(xgb.DMatrix(te[feats]), pred_contribs=True)[:, :-1]
            for f, s in zip(feats, np.abs(sh).mean(0)):
                imp_rows.append({"week": str(v.date()), "target": c, "feature": f, "group": grp[f], "mean_abs_shap": s})
        # redundancy of the R4 features
        for f in G["R4"]:
            r = xgb.XGBRegressor(**{**XGB_PARAMS, "objective": "reg:squarederror", "max_depth": 8}).fit(tr[base], tr[f])
            p = r.predict(te[base])
            y = te[f].to_numpy()
            red_rows.append({"week": str(v.date()), "feature": f, "r2_from_existing": 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()})
        # drift
        for f in feats:
            psi_rows.append({"week": str(v.date()), "feature": f, "group": grp[f],
                             "psi": psi(tr[f].to_numpy(np.float64), te[f].to_numpy(np.float64))})
    I = pd.DataFrame(imp_rows)
    I.to_csv(OUT / "S_shap_importance.csv", index=False)
    pd.DataFrame(red_rows).to_csv(OUT / "S_r4_redundancy.csv", index=False)
    P = pd.DataFrame(psi_rows)
    P.to_csv(OUT / "S_psi.csv", index=False)
    # summaries
    tot = I.groupby(["week", "target"]).mean_abs_shap.transform("sum")
    I["share"] = I.mean_abs_shap / tot
    print("\nSHAP share by group (mean over weeks):")
    print(I.groupby(["target", "group"]).share.sum().unstack().div(len(WEEKS)).round(3).to_string())
    a = I.pivot_table(index=["target", "feature"], columns="week", values="mean_abs_shap")
    print("\nSpearman rank stability of importance 09-09 vs 10-07, per target:")
    for c in TARGETS:
        x = a.loc[c]
        print(f"  {c:18s} {x.iloc[:, 0].rank().corr(x.iloc[:, 1].rank()):.3f}")
    print("\nR4 redundancy (R^2 from existing features):")
    print(pd.DataFrame(red_rows).pivot(index="feature", columns="week", values="r2_from_existing").round(3).to_string())


if __name__ == "__main__":
    main()
