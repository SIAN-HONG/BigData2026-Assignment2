"""Per-activity hyperparameters, offline from the saved fold predictions of the Phase-2c random search
(R00..R11) and the reference (O0_ref_all = depth 10 / min_child_weight 50).

Each activity has its own XGBoost model, so column j of the probability matrix can come from a
different configuration.  Honest protocol: select on one block of weeks, score on the other.
  select on extra 6 (05-06..08-19)  ->  score on main 5 (09-09..10-07), and the reverse.
Selection criteria:
  ap     : per activity, the config with the highest mean average precision (ranking quality)
  greedy : coordinate ascent on the mean official score (joint plug-in decision) over the
           selection weeks, starting from the reference
Output: results/phase2/per_activity_params.csv

  python src/per_activity_params.py                                  # Phase-2c random search
  python src/per_activity_params.py --ref X_d10_m50 --names "X_*" --out xgb_grid   # any set of runs
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle

import numpy as np
import pandas as pd

from pathlib import Path

from common import ART, RES, TARGETS
from modeling import joint_plugin_predict, official_score

REF = "O0_ref_all"
NAMES = [REF] + [f"R{i:02d}" for i in range(12)]
MAIN = [f"2026-{d}" for d in ("09-09", "09-16", "09-23", "09-30", "10-07")]
EXTRA = [f"2026-{d}" for d in ("05-06", "05-27", "06-17", "07-08", "07-29", "08-19")]


def ap(p, y):
    o = np.argsort(-p, kind="stable")
    ys = y[o]
    tp = np.cumsum(ys)
    return float((tp / np.arange(1, len(ys) + 1))[ys == 1].mean()) if ys.sum() else np.nan


def load_folds():
    F = {}
    for n in NAMES:
        path = ART / f"folds_p2_{n}.pkl"
        path = path if path.exists() else ART / f"folds_{n}.pkl"      # gnn.py runs
        F[n] = {str(v.date()): (P, Y, U) for v, P, Y, U in pickle.load(open(path, "rb"))}
    for w in MAIN + EXTRA:   # same rows in the same order in every run
        u0 = F[REF][w][2]
        assert all(np.array_equal(F[n][w][2], u0) for n in NAMES), w
    return F


def score(F, choice, weeks):
    s = []
    for w in weeks:
        P = np.column_stack([F[choice[j]][w][0][:, j] for j in range(6)])
        s.append(official_score(joint_plugin_predict(P), F[REF][w][1])["score"] * 100)
    return float(np.mean(s)), s


def select_ap(F, weeks):
    ch = []
    for j in range(6):
        m = {n: np.nanmean([ap(F[n][w][0][:, j], F[REF][w][1][:, j]) for w in weeks]) for n in NAMES}
        ch.append(max(m, key=m.get))
    return ch


def select_greedy(F, weeks, rounds=2):
    ch = [REF] * 6
    best = score(F, ch, weeks)[0]
    for _ in range(rounds):
        for j in range(6):
            for n in NAMES:
                c2 = list(ch); c2[j] = n
                sc = score(F, c2, weeks)[0]
                if sc > best + 1e-9:
                    best, ch = sc, c2
    return ch


def main():
    global REF, NAMES
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--ref", default=REF)
    ap_.add_argument("--names", default=None, help="glob over fold files, e.g. 'X_*' or 'N_mlp_*'")
    ap_.add_argument("--out", default="per_activity_params")
    a = ap_.parse_args()
    if a.names:
        found = sorted({Path(f).stem.replace("folds_p2_", "").replace("folds_", "")
                        for f in glob.glob(str(ART / f"folds_p2_{a.names}.pkl")) + glob.glob(str(ART / f"folds_{a.names}.pkl"))})
        REF, NAMES = a.ref, [a.ref] + [n for n in found if n != a.ref]
    print("runs:", NAMES, flush=True)
    F = load_folds()
    T = pd.DataFrame({n: [np.nanmean([ap(F[n][w][0][:, j], F[REF][w][1][:, j]) for w in MAIN + EXTRA]) for j in range(6)]
                      for n in NAMES}, index=TARGETS)
    T.mul(100).round(2).to_csv(RES / "phase2" / f"{a.out}_ap.csv")
    S = pd.Series({n: score(F, [n] * 6, MAIN + EXTRA)[0] for n in NAMES}, name="score_11wk")
    S.round(3).to_csv(RES / "phase2" / f"{a.out}_shared_score.csv")
    print((T * 100).round(2).to_string()); print(S.round(2).to_string())
    rows = []
    for sel, ev in (("extra", "main"), ("main", "extra")):
        sw, ew = (EXTRA, MAIN) if sel == "extra" else (MAIN, EXTRA)
        ref_ev, ref_weeks = score(F, [REF] * 6, ew)
        for crit, fn in (("ap", select_ap), ("greedy", select_greedy)):
            ch = fn(F, sw)
            sc, weeks = score(F, ch, ew)
            d = np.array(weeks) - np.array(ref_weeks)
            rows.append({"select_on": sel, "score_on": ev, "criterion": crit, "choice": json.dumps(dict(zip(TARGETS, ch))),
                         "score": round(sc, 3), "reference": round(ref_ev, 3), "delta": round(sc - ref_ev, 3),
                         "weeks_better": int((d > 0).sum()), "n_weeks": len(d)})
            print(rows[-1], flush=True)
        # single best shared config chosen on the selection weeks (what Phase 2c did, split honestly)
        m = {n: score(F, [n] * 6, sw)[0] for n in NAMES}
        b = max(m, key=m.get)
        sc, weeks = score(F, [b] * 6, ew)
        rows.append({"select_on": sel, "score_on": ev, "criterion": "shared", "choice": b, "score": round(sc, 3),
                     "reference": round(ref_ev, 3), "delta": round(sc - ref_ev, 3),
                     "weeks_better": int((np.array(weeks) > np.array(ref_weeks)).sum()), "n_weeks": len(weeks)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "phase2" / f"{a.out}.csv", index=False)


if __name__ == "__main__":
    main()
