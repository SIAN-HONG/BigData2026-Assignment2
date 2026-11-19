"""Label-shift prior adaptation on the unlabeled target week (Phase 2b).

Saerens, Latinne & Decaestecker (2002) EM: given source posteriors p(y=1|x) and source prior pi_s,
iterate  pi <- mean_i p'_i,  p'_i = (pi/pi_s) p_i / ((pi/pi_s) p_i + ((1-pi)/(1-pi_s)) (1-p_i)).
Uses only the target week's FEATURES (through the model's predictions), never its labels.
pi_s = positive rate of the training samples the model was fitted on (cutoffs <= V - 8 d).
Variants: full EM, damped (geometric mean of EM prior and model prior), EM after
bias-corrected temperature scaling is NOT possible without held-out source predictions, so skipped.
  python src/label_shift.py
"""
import pickle

import numpy as np
import pandas as pd

from common import ART, RES, TARGETS
from modeling import YCOLS, joint_plugin_predict, official_score


def em_prior(p, pi_s, iters=200, tol=1e-7):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pi = p.mean()
    for _ in range(iters):
        a = pi / pi_s * p
        q = a / (a + (1 - pi) / (1 - pi_s) * (1 - p))
        new = q.mean()
        if abs(new - pi) < tol:
            break
        pi = new
    return pi, q


def adjust(p, pi_s, pi_t):
    a = pi_t / pi_s * p
    return a / (a + (1 - pi_t) / (1 - pi_s) * (1 - p))


if __name__ == "__main__":
    df = pd.read_parquet(ART / "dataset.parquet", columns=["cutoff"] + YCOLS)
    sets = {"v1": pickle.load(open(ART / "folds_v1_extra.pkl", "rb"))
                  + pickle.load(open(ART / "folds_final_backtest_v1_baseline.pkl", "rb"))}
    mb = pickle.load(open(ART / "members_p2b_extra.pkl", "rb")) + pickle.load(open(ART / "members_p2b_backtest.pkl", "rb"))
    sets["xgbW5"] = [(v, np.mean(m[:5], 0), Y, U) for v, m, Y, U in mb]
    rows = []
    for name, F in sets.items():
        for v, P, Y, U in F:
            tr = df[(df.cutoff <= v - pd.Timedelta(days=8))]
            tr = tr[((v - pd.Timedelta(days=8) - tr.cutoff).dt.days % 3) == 0]
            pi_s = tr[YCOLS].mean().to_numpy()
            Q, Qd = np.zeros_like(P), np.zeros_like(P)
            pis = []
            for j in range(6):
                pi_t, Q[:, j] = em_prior(P[:, j], pi_s[j])
                pis.append(pi_t)
                Qd[:, j] = adjust(np.clip(P[:, j], 1e-6, 1 - 1e-6), pi_s[j], np.sqrt(pi_t * P[:, j].mean()))
            r = {"set": name, "week": str(v.date()),
                 "joint": official_score(joint_plugin_predict(P), Y)["score"],
                 "em": official_score(joint_plugin_predict(Q), Y)["score"],
                 "em_damped": official_score(joint_plugin_predict(Qd), Y)["score"]}
            for j, c in enumerate(TARGETS):
                r[f"true_{c}"], r[f"meanp_{c}"], r[f"em_{c}"] = Y[:, j].mean(), P[:, j].mean(), pis[j]
            rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(RES / "phase2" / "label_shift_em.csv", index=False)
    E = R[R.week != "2026-09-02"]
    print(E.groupby("set")[["joint", "em", "em_damped"]].mean().round(4))
    print("main 5 wk\n", E[E.week >= "2026-09-01"].groupby("set")[["joint", "em", "em_damped"]].mean().round(4))
    # does EM move the level toward the truth?
    for c in TARGETS:
        e = E[E.set == "xgbW5"]
        print(f"{c:18s} |meanp-true| {np.abs(e[f'meanp_{c}'] - e[f'true_{c}']).mean():.4f}   |em-true| {np.abs(e[f'em_{c}'] - e[f'true_{c}']).mean():.4f}")
