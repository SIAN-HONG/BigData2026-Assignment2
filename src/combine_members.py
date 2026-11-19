"""Evaluate a small, pre-declared set of member combinations from saved backtest members.

members_p2b_backtest.pkl : 5 weighted XGB seeds, then gnn s0, gnn s1, mlp s0, mlp s1 (weighted)
members_p2c_backtest.pkl : gnn s0, gnn s1, mlp s0, mlp s1 (unweighted)
Writes results/phase2/member_combinations.csv
"""
import pickle

import numpy as np
import pandas as pd

from common import ART, RES, TARGETS
from exp2 import summarize_p2

B = pickle.load(open(ART / "members_p2b_backtest.pkl", "rb"))
C = pickle.load(open(ART / "members_p2c_backtest.pkl", "rb"))
_extra = ART / "members_p2b_extra.pkl"
if _extra.exists():   # extra weeks 05-06 .. 08-19 (same members, earlier cutoffs)
    B = pickle.load(open(_extra, "rb")) + B
    C = pickle.load(open(ART / "members_p2c_extra.pkl", "rb")) + C


V1 = {v: P for v, P, Y, U in pickle.load(open(ART / "folds_final_backtest_v1_baseline.pkl", "rb"))}
if (ART / "folds_v1_extra.pkl").exists():
    V1.update({v: P for v, P, Y, U in pickle.load(open(ART / "folds_v1_extra.pkl", "rb"))})


def combo(xgb_w, nn_w, nn_unw, w_trees=None, v1=None):
    folds = []
    for (v, mb, Y, U), (v2, mc, Y2, U2) in zip(B, C):
        assert v == v2 and (U == U2).all()
        if v1 is not None:   # blend of the v1 ensemble and the weighted trees
            P = v1 * V1[v] + (1 - v1) * np.mean([mb[i] for i in xgb_w], 0) if xgb_w else V1[v]
            folds.append((v, P, Y, U))
            continue
        parts_t = [mb[i] for i in xgb_w]
        parts_n = [mb[i] for i in nn_w] + [mc[i] for i in nn_unw]
        if w_trees is None:
            P = np.mean(parts_t + parts_n, axis=0)
        else:
            P = w_trees * np.mean(parts_t, 0) + (1 - w_trees) * np.mean(parts_n, 0)
        folds.append((v, P, Y, U))
    return folds


X5, NNW, NNU = [0, 1, 2, 3, 4], [5, 6, 7, 8], [0, 1, 2, 3]
SPECS = {
    "v1 ensemble (reference)": dict(xgb_w=[], nn_w=[], nn_unw=[], v1=1.0),
    "0.5 v1 ensemble + 0.5 xgbW x5": dict(xgb_w=X5, nn_w=[], nn_unw=[], v1=0.5),
    "xgbW x3": dict(xgb_w=[0, 1, 2], nn_w=[], nn_unw=[]),
    "xgbW x5": dict(xgb_w=X5, nn_w=[], nn_unw=[]),
    "xgbW x3 + nnW (gnn,mlp s0) [=p2a]": dict(xgb_w=[0, 1, 2], nn_w=[5, 7], nn_unw=[]),
    "xgbW x5 + nnW x4 equal": dict(xgb_w=X5, nn_w=NNW, nn_unw=[]),
    "xgbW x5 + nnU x4 equal": dict(xgb_w=X5, nn_w=[], nn_unw=NNU),
    "0.7 xgbW x5 + 0.3 nnW x4": dict(xgb_w=X5, nn_w=NNW, nn_unw=[], w_trees=0.7),
    "0.7 xgbW x5 + 0.3 nnU x4": dict(xgb_w=X5, nn_w=[], nn_unw=NNU, w_trees=0.7),
    "nnW x4 only": dict(xgb_w=[], nn_w=NNW, nn_unw=[]),
    "nnU x4 only": dict(xgb_w=[], nn_w=[], nn_unw=NNU),
}

if __name__ == "__main__":
    rows = []
    for name, sp in SPECS.items():
        F = combo(**sp)
        main = [f for f in F if f[0] >= pd.Timestamp("2026-09-01")]
        extra = [f for f in F if f[0] < pd.Timestamp("2026-09-01")]
        r = summarize_p2(main, "combo_tmp")
        row = {"combination": name, **{k: r[k] for k in ("score", "macro_f1", "micro_f1", "score_6wk",
                                                          "week_sd", "oracle_score", "per_week")},
               **{f"f1_{c}": r[f"f1_{c}"] for c in TARGETS}}
        if extra:
            re_ = summarize_p2(extra, "combo_tmp")
            row["score_extra6"] = re_["score"]
            row["score_11wk"] = (5 * r["score"] + 6 * re_["score"]) / 11
            row["per_week_extra"] = re_["per_week"]
        rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(RES / "phase2" / "member_combinations.csv", index=False)
    (RES / "phase2" / "weeks_combo_tmp.csv").unlink(missing_ok=True)
    pd.set_option("display.width", 250)
    print(R.drop(columns=[c for c in R if c.startswith("f1_")]).round(4).to_string(index=False))
