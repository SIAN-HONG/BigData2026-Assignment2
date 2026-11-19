"""Uncertainty-aware decision rule (Phase 2 test).

The joint plug-in rule trusts the model's probability LEVEL; the decomposition showed the level is
the largest loss.  Here k (positives per activity) maximises the expected official score averaged
over a common level multiplier m ~ LogNormal(0, sigma) applied to every activity's odds-shifted
probabilities (quadrature over 9 quantiles).  sigma is estimated from TRAINING label history only:
sd of log(prevalence_t / prevalence_{t-7d}) over training cutoffs (week-over-week level changes).
  python src/robust_decision.py
"""
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from common import ART, RES, TARGETS
from modeling import YCOLS, joint_plugin_predict, official_score


def _shift(p, mult):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p))
    target = min(0.97, p.mean() * mult)
    b = brentq(lambda b: (1 / (1 + np.exp(-(lg + b)))).mean() - target, -20, 20)
    return 1 / (1 + np.exp(-(lg + b)))


def robust_predict(P, sigma, nq=9, iters=2):
    n, m = P.shape
    if sigma <= 0:
        return joint_plugin_predict(P)
    qs = norm.ppf((np.arange(nq) + 0.5) / nq) * sigma
    order = [np.argsort(-P[:, j], kind="stable") for j in range(m)]
    # cumulative expected TP along each activity's ranking, for every scenario
    C = np.zeros((nq, m, n + 1))
    for s, q in enumerate(qs):
        for j in range(m):
            C[s, j, 1:] = np.cumsum(_shift(P[:, j], np.exp(q))[order[j]])
    tot = C[:, :, -1]

    def exp_score(ks):
        ks = np.asarray(ks)
        tp = C[:, np.arange(m), ks]                 # (nq, m)
        fp = ks[None, :] - tp
        fn = tot - tp
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
        micro = 2 * tp.sum(1) / (2 * tp.sum(1) + fp.sum(1) + fn.sum(1))
        return np.mean((f1.mean(1) + micro) / 2)

    ks = list(joint_plugin_predict(P).sum(0))
    grid = np.unique(np.geomspace(1, n, 300).astype(int))
    for _ in range(iters):
        for j in range(m):
            best_s, best_k = exp_score(ks), ks[j]
            for k in grid:
                ks2 = list(ks); ks2[j] = int(k)
                sc = exp_score(ks2)
                if sc > best_s:
                    best_s, best_k = sc, int(k)
            ks[j] = best_k
    out = np.zeros(P.shape, int)
    for j in range(m):
        out[order[j][:ks[j]], j] = 1
    return out


def sigma_from_history(df, cut):
    """sd of week-over-week log prevalence ratio over training cutoffs (<= cut - 8d)."""
    prev = df[df.cutoff <= cut - pd.Timedelta(days=8)].groupby("cutoff")[YCOLS].mean()
    lr = np.log((prev + 1e-3) / (prev.shift(7) + 1e-3)).dropna()
    return float(lr.iloc[:, :5].stack().std())     # the five rare activities


if __name__ == "__main__":
    df = pd.read_parquet(ART / "dataset.parquet", columns=["cutoff"] + YCOLS)
    sets = {"v1": pickle.load(open(ART / "folds_v1_extra.pkl", "rb"))
                  + pickle.load(open(ART / "folds_final_backtest_v1_baseline.pkl", "rb"))}
    mb = pickle.load(open(ART / "members_p2b_extra.pkl", "rb")) + pickle.load(open(ART / "members_p2b_backtest.pkl", "rb"))
    sets["xgbW5"] = [(v, np.mean(m[:5], 0), Y, U) for v, m, Y, U in mb]
    rows = []
    for name, F in sets.items():
        for v, P, Y, U in F:
            sg = sigma_from_history(df, v)
            r = {"set": name, "week": str(v.date()), "sigma_hist": sg,
                 "joint": official_score(joint_plugin_predict(P), Y)["score"]}
            for s in (0.15, 0.3):
                r[f"robust_{s}"] = official_score(robust_predict(P, s), Y)["score"]
            r["robust_hist"] = official_score(robust_predict(P, sg), Y)["score"]
            rows.append(r)
            print(r, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(RES / "phase2" / "robust_decision.csv", index=False)
    E = R[R.week != "2026-09-02"]
    print(E.groupby("set")[["joint", "robust_0.15", "robust_0.3", "robust_hist", "sigma_hist"]].mean().round(4))
    E2 = E[E.week >= "2026-09-01"]
    print("main 5 weeks\n", E2.groupby("set")[["joint", "robust_0.15", "robust_0.3", "robust_hist"]].mean().round(4))
