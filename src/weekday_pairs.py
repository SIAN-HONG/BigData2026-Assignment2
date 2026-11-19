"""Is a Wednesday validation week representative of a Saturday prediction cutoff (the test cutoff
2026-11-21 is a Saturday)?  Five fixed pairs, each Saturday with the Wednesday 3 days earlier
(same every=3 sampling phase of modeling.train_window), one data snapshot, the chosen version p3.

  CUDA_VISIBLE_DEVICES=<gpu> python pipeline.py --backtest --name wd_p3_imp \
      --config configs/final_config_p3_imp.json --val <the 10 dates>
  python weekday_pairs.py --folds wd_p3_imp

Per date and activity: positive rate, predicted positive rate, AP, F1; the week score on the activities
kept on both days of a pair (an activity with an outage day in either label window is left out of both),
and the same score on the users present on both days.  Writes results/weekday_pairs.{csv,md}.
The label windows of a pair overlap by 4 of 7 days: five pairs describe the difference, they are not
five independent samples.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import ART, RES, TARGETS
from forward_decide import outage_targets
from modeling import joint_plugin_predict, official_score

def md(df: pd.DataFrame, index=True) -> str:
    """Markdown table without the optional tabulate dependency."""
    d = df.reset_index() if index else df
    d.columns = [" ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c) for c in d.columns]
    rows = [list(map(str, d.columns)), ["---"] * len(d.columns)] + d.astype(str).values.tolist()
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


PAIRS = [("2026-09-09", "2026-09-12"), ("2026-09-16", "2026-09-19"), ("2026-09-23", "2026-09-26"),
         ("2026-09-30", "2026-10-03"), ("2026-10-07", "2026-10-10")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="wd_p3_imp")
    a = ap.parse_args()
    folds = {str(v.date()): (P, Y, u) for v, P, Y, u in pickle.load(open(ART / f"folds_{a.folds}.pkl", "rb"))}
    ev_end = pd.read_parquet(ART / "events.parquet", columns=["ts"]).ts.max()
    drop = outage_targets([d for p in PAIRS for d in p], pd.Timestamp(ev_end, unit="s"))
    rows, pair_rows = [], []
    for wed, sat in PAIRS:
        keep = [j for j, k in enumerate(TARGETS) if k not in set(drop[wed]) | set(drop[sat])]
        five = [j for j, k in enumerate(TARGETS) if k != "push"]   # fixed set: the five non-push activities
        if any(TARGETS[j] in set(drop[wed]) | set(drop[sat]) for j in five):
            raise SystemExit(f"outage in a non-push activity for pair {wed}/{sat}: fixed-five score undefined")
        sc, sc5 = {}, {}
        for day, d in (("Wed", wed), ("Sat", sat)):
            P, Y, u = folds[d]
            D = joint_plugin_predict(P)
            f1 = official_score(D, Y)
            for j, k in enumerate(TARGETS):
                rows.append({"pair": wed, "day": day, "date": d, "activity": k, "n_users": len(u),
                             "pos_rate": Y[:, j].mean(), "pred_rate": D[:, j].mean(),
                             "ap": average_precision_score(Y[:, j], P[:, j]) if Y[:, j].any() else np.nan,
                             "f1": f1[f"f1_{k}"], "outage": k in drop[d]})
            sc[day] = official_score(D[:, keep], Y[:, keep])["score"] * 100
            sc5[day] = official_score(D[:, five], Y[:, five])["score"] * 100
        # users present on both days (decisions as made on each full day)
        (Pw, Yw, uw), (Ps, Ys, us) = folds[wed], folds[sat]
        com = np.intersect1d(uw, us)
        iw, is_ = np.isin(uw, com), np.isin(us, com)
        Dw, Ds = joint_plugin_predict(Pw)[iw], joint_plugin_predict(Ps)[is_]
        pair_rows.append({"wed": wed, "sat": sat, "kept": ",".join(TARGETS[j] for j in keep),
                          "n_wed": len(uw), "n_sat": len(us), "n_common": len(com),
                          "score_wed": sc["Wed"], "score_sat": sc["Sat"], "delta_sat_minus_wed": sc["Sat"] - sc["Wed"],
                          "five_wed": sc5["Wed"], "five_sat": sc5["Sat"], "five_delta": sc5["Sat"] - sc5["Wed"],
                          "common_users_wed": official_score(Dw[:, keep], Yw[iw][:, keep])["score"] * 100,
                          "common_users_sat": official_score(Ds[:, keep], Ys[is_][:, keep])["score"] * 100})
    R, S = pd.DataFrame(rows), pd.DataFrame(pair_rows)
    S["common_delta"] = S.common_users_sat - S.common_users_wed
    R.to_csv(RES / "weekday_pairs_activity.csv", index=False)
    S.to_csv(RES / "weekday_pairs.csv", index=False)
    piv = R.pivot_table(index="activity", columns="day", values=["pos_rate", "pred_rate", "ap", "f1"]).reindex(TARGETS)
    piv = piv * 100
    lines = ["# Wednesday vs Saturday cutoffs (p3, one data snapshot)", "",
             "Score on the activities kept on both days (points):", "", md(S.round(2), index=False), "",
             f"Mean Sat − Wed: {S.delta_sat_minus_wed.mean():+.2f} (better on Sat in "
             f"{(S.delta_sat_minus_wed > 0).sum()} of {len(S)}); common users: {S.common_delta.mean():+.2f}; "
             f"fixed five non-push activities: {S.five_delta.mean():+.2f} (better on Sat in {(S.five_delta > 0).sum()} of {len(S)})", "",
             "Per activity, mean over the five dates of each weekday (%):", "", md(piv.round(2)), "",
             "Label windows of a pair overlap by 4 of 7 days: descriptive, not independent samples."]
    (RES / "weekday_pairs.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
