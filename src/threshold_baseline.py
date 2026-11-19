"""Label-free decision rule vs thresholds learned from an earlier week whose labels are mature at the
prediction cutoff (no training; existing backtest folds).

  python src/threshold_baseline.py --folds b15_p3_imp

For a validation week V the most recent week with mature labels at V 12:00 is S = V - 14 d (S's label
window ends S + 4 h + 7 d and the 5-day slack ends before V).  Baseline: the same-week threshold search
(modeling.tune_thresholds, joint) run on S, restricted to the activities without an outage in S's label
window (known at V); activities with an outage in S fall back to the label-free per-activity rule.
The thresholds are applied to V's probabilities.  Scoring afterwards leaves out the activities with an
outage in V's label window (as the official scoring does).  Also reported: the per-activity and joint
label-free rules, and the same-week threshold search on V itself (uses V's labels: a reference, not an
available method, and a grid / coordinate search, not a proven upper bound).
Writes results/threshold_baseline_<folds>.{csv,md}.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from common import ART, RES, TARGETS
from forward_decide import outage_targets, status
from modeling import EXCLUDE_WEEKS, joint_plugin_predict, official_score, plugin_predict, tune_thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="b15_p3_imp")
    a = ap.parse_args()
    F = {str(v.date()): (P, Y) for v, P, Y, _ in pickle.load(open(ART / f"folds_{a.folds}.pkl", "rb"))}
    data_end = pd.Timestamp(pd.read_parquet(ART / "events.parquet", columns=["ts"]).ts.max(), unit="s") + pd.Timedelta(seconds=1)
    weeks = [w for w in sorted(F) if w not in EXCLUDE_WEEKS and status(w, data_end) == "complete"]
    pairs = [(w, str((pd.Timestamp(w) - pd.Timedelta(days=14)).date())) for w in weeks]
    pairs = [(w, s) for w, s in pairs if s in F and s not in EXCLUDE_WEEKS]
    drop = outage_targets(sorted({x for p in pairs for x in p}), data_end)
    rows = []
    for w, s in pairs:
        P, Y = F[w]
        Ps, Ys = F[s]
        src = [j for j, k in enumerate(TARGETS) if k not in drop[s]]          # known at V
        D_base = plugin_predict(P)                                            # fallback for the others
        th = tune_thresholds(Ps[:, src], Ys[:, src], True)
        D_base[:, src] = (P[:, src] >= th).astype(int)
        keep = [j for j, k in enumerate(TARGETS) if k not in drop[w]]         # scoring only
        sc = lambda D: official_score(D[:, keep], Y[:, keep])["score"] * 100
        th_v = tune_thresholds(P[:, keep], Y[:, keep], True)
        rows.append({"week": w, "threshold_source": s, "scored": ",".join(TARGETS[j] for j in keep),
                     "fallback": ",".join(TARGETS[j] for j in range(6) if j not in src),
                     "mature_past_threshold": sc(D_base), "per_activity_plugin": sc(plugin_predict(P)),
                     "joint_plugin": sc(joint_plugin_predict(P)),
                     "same_week_search": official_score((P[:, keep] >= th_v).astype(int), Y[:, keep])["score"] * 100})
    R = pd.DataFrame(rows)
    R["joint_minus_past"] = R.joint_plugin - R.mature_past_threshold
    R.to_csv(RES / f"threshold_baseline_{a.folds}.csv", index=False)
    num = ["mature_past_threshold", "per_activity_plugin", "joint_plugin", "same_week_search", "joint_minus_past"]
    lines = [f"# Decision rules on {a.folds} (points; outage-aware scoring)", "",
             "| " + " | ".join(R.columns) + " |", "|" + "---|" * len(R.columns)]
    lines += ["| " + " | ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in r) + " |" for r in R.values.tolist()]
    m = R[num].mean()
    lines += ["", "Mean: " + ", ".join(f"{c} {m[c]:.2f}" for c in num),
              f"Joint plug-in better in {(R.joint_minus_past > 0).sum()} of {len(R)} weeks; "
              f"mean difference without the largest week: {R.joint_minus_past.drop(R.joint_minus_past.idxmax()).mean():+.2f}."]
    # probability scale vs positive rate in the source and target week (why a numeric threshold may not transfer)
    lines += ["", "Positive rate vs mean predicted probability (%), source week -> week:", ""]
    for w, s in pairs:
        P, Y = F[w]
        Ps, Ys = F[s]
        lines.append(f"- {s} -> {w}: " + "; ".join(
            f"{k} {100 * Ys[:, j].mean():.2f}/{100 * Ps[:, j].mean():.2f} -> {100 * Y[:, j].mean():.2f}/{100 * P[:, j].mean():.2f}"
            for j, k in enumerate(TARGETS)))
    (RES / f"threshold_baseline_{a.folds}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
