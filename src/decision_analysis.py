"""Decision-rule analysis (Phase 2).

1. Approximation: the joint plug-in maximises S(E[TP], E[FP], E[FN]) (ratio of expectations), not
   E[S] (expected ratio).  Monte Carlo: draw labels y ~ Bernoulli(p) from the model's own
   probabilities (i.e. assuming calibration) and compare E[S] at the plug-in choice with the best
   E[S] over a neighbourhood of k (each target's k scaled by 0.8 .. 1.25).
2. Loss decomposition on the saved baseline folds:
   joint plug-in  ->  same rule after shifting each activity's probabilities so that their mean
   equals the TRUE prevalence (oracle level, ranking unchanged)  ->  oracle thresholds.
   The first gap is the cost of mis-estimated base rates, the second the cost of the rule given
   perfect levels, and the rest is ranking quality.
  python src/decision_analysis.py [folds_name]
"""
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from common import ART, RES
from modeling import EXCLUDE_WEEKS, joint_plugin_predict, official_score, tune_thresholds


def topk_pred(P, ks):
    out = np.zeros(P.shape, int)
    for j, k in enumerate(ks):
        out[np.argsort(-P[:, j], kind="stable")[:k], j] = 1
    return out


def mc_expected_score(P, D, n=200, rng=None):
    rng = rng or np.random.default_rng(0)
    return np.mean([official_score(D, (rng.random(P.shape) < P).astype(int))["score"] for _ in range(n)])


def shift_to(p, target):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p))
    b = brentq(lambda b: (1 / (1 + np.exp(-(lg + b)))).mean() - target, -15, 15)
    return 1 / (1 + np.exp(-(lg + b)))


def main(name="final_backtest_v1_baseline"):
    folds = pickle.load(open(ART / f"folds_{name}.pkl", "rb"))
    rows = []
    for v, P, Y, _ in folds:
        wk = str(v.date())
        D = joint_plugin_predict(P)
        ks = D.sum(0)
        # 1. Monte Carlo check of the plug-in approximation (assumes calibrated p)
        rng = np.random.default_rng(0)
        es_plugin = mc_expected_score(P, D, rng=rng)
        best = es_plugin
        for f in (0.8, 0.9, 1.1, 1.25):
            for j in range(6):
                k2 = ks.copy(); k2[j] = max(1, int(round(ks[j] * f)))
                best = max(best, mc_expected_score(P, topk_pred(P, k2), n=100, rng=rng))
        # 2. decomposition
        s_joint = official_score(D, Y)["score"]
        Q = np.column_stack([shift_to(P[:, j], Y[:, j].mean()) for j in range(6)])
        s_level = official_score(joint_plugin_predict(Q), Y)["score"]
        s_oracle = official_score((P >= tune_thresholds(P, Y)).astype(int), Y)["score"]
        rows.append(dict(week=wk, mc_E_score_plugin=es_plugin, mc_best_E_score_nearby=best,
                         joint=s_joint, joint_with_true_levels=s_level, oracle_thresholds=s_oracle))
    R = pd.DataFrame(rows)
    E = R[~R.week.isin(EXCLUDE_WEEKS)]
    R.loc[len(R)] = {"week": "mean (5 wk)", **E.drop(columns="week").mean().to_dict()}
    R.to_csv(RES / "phase2" / f"decision_rule_analysis_{name}.csv", index=False)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
