"""Per-activity error analysis from existing backtest folds (no training, no GPU).

  python src/activity_errors.py --prefix b15            # folds_b15_{v1,p3,p4}_imp.pkl
  python src/activity_errors.py --prefix f17 --focus p4 # after the formal decision

Weeks: every week in the folds except modeling.EXCLUDE_WEEKS and weeks whose labels are not mature
(label window + 5-day slack after the data end; forward_decide.status).  An activity with an outage day
in a week's label window (forward_decide.outage_targets, as in the official scoring) is left out of
that week.  Decisions: the label-free joint plug-in rule used for the submission.

Part A (all versions): per activity, the mean over the kept weeks of positive rate, predicted-positive
rate, precision, recall, F1 and average precision (AP).
Part B (one version, --focus): the positives of each kept (week, activity) split by how many days before
the cutoff the user last did that activity (dataset column rec_<activity>, imputed events; 999 = none observed):
share of the positives, recall, and precision of the users predicted 1 in that bucket.
Writes results/activity_errors_<prefix>.{md,csv}, results/activity_errors_<prefix>_recency.csv and
figures/activity_errors_<prefix>.png.
"""
from __future__ import annotations

import argparse
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import ART, RES, ROOT, TARGETS
from forward_decide import outage_targets, status
from modeling import EXCLUDE_WEEKS, joint_plugin_predict

VERSIONS = ["v1", "p3", "p4"]
NEVER = 999.0   # features.py: rec_<activity> = days since the last event, fillna(999) when none is observed
BUCKETS = [(0, 7, "< 7 d"), (7, 28, "7-28 d"), (28, 90, "28-90 d"), (90, 1e9, "≥ 90 d"), (None, None, "none observed")]


def md(df: pd.DataFrame) -> str:
    rows = [list(map(str, df.columns)), ["---"] * len(df.columns)] + [[str(x) for x in r] for r in df.values.tolist()]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="b15")
    ap.add_argument("--focus", default="p3", help="version for the recency breakdown (part B)")
    a = ap.parse_args()

    folds = {v: {str(w.date()): (P, Y, u) for w, P, Y, u in pickle.load(open(ART / f"folds_{a.prefix}_{v}_imp.pkl", "rb"))}
             for v in VERSIONS}
    ev_end = pd.Timestamp(pd.read_parquet(ART / "events.parquet", columns=["ts"]).ts.max(), unit="s")
    data_end = ev_end + pd.Timedelta(seconds=1)
    weeks = sorted(folds[VERSIONS[0]])
    immature = [w for w in weeks if status(w, data_end) != "complete"]
    weeks = [w for w in weeks if w not in EXCLUDE_WEEKS and w not in immature]
    drop = outage_targets(weeks, data_end)

    # Part A
    rows = []
    for v in VERSIONS:
        for w in weeks:
            P, Y, _ = folds[v][w]
            D = joint_plugin_predict(P)
            for j, k in enumerate(TARGETS):
                if k in drop[w]:
                    continue
                y, d = Y[:, j].astype(bool), D[:, j].astype(bool)
                tp = (y & d).sum()
                prec = tp / d.sum() if d.sum() else 0.0
                rec = tp / y.sum() if y.sum() else 0.0
                rows.append({"version": v, "week": w, "activity": k, "n_users": len(y),
                             "pos_rate": y.mean(), "pred_rate": d.mean(), "precision": prec, "recall": rec,
                             "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                             "ap": average_precision_score(y, P[:, j]) if y.any() else np.nan})
    A = pd.DataFrame(rows)
    A.to_csv(RES / f"activity_errors_{a.prefix}.csv", index=False)
    S = (A.groupby(["activity", "version"])[["pos_rate", "pred_rate", "precision", "recall", "f1", "ap"]].mean() * 100)
    S["weeks"] = A.groupby(["activity", "version"]).size()
    S = S.reindex(pd.MultiIndex.from_product([TARGETS, VERSIONS], names=["activity", "version"])).reset_index()

    # Part B
    cut_ts = {w: pd.Timestamp(f"{w} 12:00") for w in weeks}
    cols = ["cutoff", "UserID"] + [f"rec_{k}" for k in TARGETS]
    X = pd.read_parquet(ART / "dataset_imp.parquet", columns=cols,
                        filters=[("cutoff", "in", list(cut_ts.values()))])
    brow = []
    for w in weeks:
        P, Y, u = folds[a.focus][w]
        D = joint_plugin_predict(P)
        x = X[X.cutoff == cut_ts[w]].set_index("UserID").reindex(u)
        if x.isna().any().any():
            raise SystemExit(f"recency features missing for some users of week {w}")
        for j, k in enumerate(TARGETS):
            if k in drop[w]:
                continue
            r = x[f"rec_{k}"].to_numpy()
            y, d = Y[:, j].astype(bool), D[:, j].astype(bool)
            for lo, hi, name in BUCKETS:
                m = (r == NEVER) if lo is None else ((r >= lo) & (r < hi) & (r != NEVER))
                brow.append({"week": w, "activity": k, "bucket": name, "users": int(m.sum()),
                             "pos": int((y & m).sum()), "pred": int((d & m).sum()), "tp": int((y & d & m).sum())})
    B = pd.DataFrame(brow)
    B.to_csv(RES / f"activity_errors_{a.prefix}_recency.csv", index=False)
    g = B.groupby(["activity", "bucket"])[["users", "pos", "pred", "tp"]].sum()   # pooled over kept weeks
    tot = g.groupby(level=0).pos.transform("sum")
    g["pos_share"] = 100 * g.pos / tot
    g["recall"] = 100 * g.tp / g.pos.where(g.pos > 0)
    g["precision"] = 100 * g.tp / g.pred.where(g.pred > 0)
    order = [b[2] for b in BUCKETS]
    g = g.reindex(pd.MultiIndex.from_product([TARGETS, order], names=["activity", "bucket"])).reset_index()
    missed_new = g[g.bucket.isin(["≥ 90 d", "none observed"])].groupby("activity").apply(
        lambda t: 100 * (t.pos - t.tp).sum() / max(1, (g[g.activity == t.name].pos - g[g.activity == t.name].tp).sum()))

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x0 = np.arange(len(TARGETS))
    for i, v in enumerate(VERSIONS):
        f = S[S.version == v].set_index("activity").reindex(TARGETS)
        ax[0].bar(x0 + (i - 1) * 0.27, f.f1, width=0.27, label=v)
    ax[0].set_xticks(x0, [t.replace("_", "\n") for t in TARGETS], fontsize=8)
    ax[0].set_ylabel("F1 (%), mean over kept weeks")
    ax[0].set_title("F1 per activity")
    ax[0].legend(fontsize=8)
    for k in TARGETS:
        t = g[g.activity == k]
        ax[1].plot(order, t.recall, marker="o", label=k)
    ax[1].set_ylabel(f"recall of positives (%), {a.focus}")
    ax[1].set_xlabel("days since the user's last observed action of this kind (imputed features, at the cutoff)")
    ax[1].set_title("Who is missed: recall by recency")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"activity_errors_{a.prefix}.png", dpi=180)

    lines = [f"# Per-activity errors ({a.prefix} folds, label-free joint plug-in decisions)", "",
             f"Data end {data_end}. Weeks kept ({len(weeks)}): {', '.join(weeks)}.",
             f"Left out: {', '.join(sorted(EXCLUDE_WEEKS))} (reported-only week)"
             + (f"; immature labels: {', '.join(immature)}" if immature else "") + ".",
             "Outage activity-weeks left out: " + "; ".join(f"{w}: {', '.join(d)}" for w, d in drop.items() if d) + ".", "",
             "## A. Mean over kept weeks (%)", "", md(S.round(2)), "",
             f"## B. {a.focus}: positives by days since the user's last action of the same kind (pooled over kept weeks)", "",
             md(g.round(1)), "",
             "Rows of part B are user-weeks pooled over the kept weeks (not distinct users); part A averages per week. "
             "rec_<activity> is computed on the outage-imputed events, as the model sees it; 'none observed' = "
             "no event of that kind in the available data (value 999), not 'never in the user's life'.", "",
             "Share of all missed positives (false negatives) from users with the last action ≥ 90 d ago or none "
             "observed: " + ", ".join(f"{k} {missed_new[k]:.1f} %" for k in TARGETS) + "."]
    (RES / f"activity_errors_{a.prefix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
