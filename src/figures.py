"""Figures for the report (all generated from data / experiment ledger).

  ASG2_ART=fwd/artifacts ASG2_DB=fwd/assignment2/data/db python src/figures.py
    figures/activity_timeline.png  weekly cohort activity per type, outage days marked (Q1 / Q3)
    figures/action_relations.png   co-occurrence, lag-1 lift and residual correlation of the 6 actions (Q1)
    figures/outage_imputation.png  push-outage simulation and rehearsal probabilities, with / without imputation (Q3)
    figures/weekly_scores.png      outage-aware week scores of v1 / p3 / p4 (Q4)
    figures/ablation.png           validation score of the experiment ledger (Q2 / Q3)
ASG2_ART selects the release whose artifacts are drawn (default artifacts/).
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ART, FIG, KIND_NAMES, RES, ROOT as ROOT_DIR, TARGETS, load_tables, outage_flags

BLUE, GREY, INK, MUTED = "#2a78d6", "#d9d8d4", "#0b0b0b", "#52514e"
ORANGE, AQUA, RED = "#eb6834", "#1baf7a", "#e34948"
SHORT = ["open_pr", "open_issue", "issue_cmt", "pr_review", "pr_rev_cmt", "push"]
REL = Path(os.environ.get("ASG2_ART", ART))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#b0afab",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
T0 = pd.Timestamp("2025-12-01")
VAL_WEEKS = ["2026-05-06", "2026-05-27", "2026-06-17", "2026-07-08", "2026-07-29", "2026-08-19", "2026-09-09",
             "2026-09-16", "2026-09-23", "2026-09-30", "2026-10-07", "2026-10-14", "2026-10-21", "2026-10-28", "2026-11-04"]
# outage days: common.outage_flags, the same rule as forward_decide.outage_targets


def activity_timeline():
    ev = pd.read_parquet(REL / "events.parquet")
    u = load_tables()["Users"]
    coh = set(u.user_id[u.is_cohort])
    e = ev[ev.kind < 6]
    day_all = pd.to_datetime(e.ts, unit="s").dt.floor("D")
    end = pd.to_datetime(ev.avail.max(), unit="s").floor("D")        # last complete UTC day = end - 1
    c = pd.crosstab(day_all, e.kind)
    c = c.reindex(pd.date_range(c.index.min(), end - pd.Timedelta(days=1)), fill_value=0)
    out = c.apply(lambda col: pd.Series(outage_flags(col.to_numpy())[0], index=col.index))   # common.outage_flags
    e = e[e.uid.isin(coh)]
    day = pd.to_datetime(e.ts, unit="s").dt.floor("D")
    d = pd.crosstab(day, e.kind)
    d = d[(d.index >= "2025-12-01") & (d.index < end)]
    wk = d.resample("W-WED").sum().iloc[:-1]     # weeks ending Wednesday (cutoff weekday)
    print("weeks", wk.index.min(), wk.index.max())
    fig, axs = plt.subplots(2, 3, figsize=(13, 5.8), sharex=True)
    months = pd.date_range("2025-12-01", end, freq="2MS")
    for k, ax in zip(range(6), axs.flat):
        # numeric axis (days since 2025-12-01) with explicit month labels; each weekly
        # total is drawn at the middle of its week (weeks end Wednesday 00:00 UTC)
        x = ((wk.index - pd.Timedelta(days=3, hours=12)) - T0).total_seconds() / 86400
        ax.plot(x, wk[k].to_numpy(), color=BLUE, lw=2)
        for v in VAL_WEEKS:
            s = (pd.Timestamp(v) + pd.Timedelta(hours=16) - T0).total_seconds() / 86400
            ax.axvspan(s, s + 7, color=GREY, alpha=0.5, lw=0)
        od = out.index[out[k] & (out.index >= T0)]
        for o in od:
            ax.axvspan((o - T0).days, (o - T0).days + 1, ymin=0, ymax=0.06, color=RED, lw=0)
        ax.set_title(f"{TARGETS[k]}   ({len(od)} outage days)", color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", color="#ecebe8", lw=0.8)
        ax.set_ylim(0, None)
        ax.set_xticks((months - T0).days, [m.strftime("%b\n%Y") for m in months])
    fig.suptitle(f"Weekly activity of the 9,000-user cohort (events per week; grey = {len(VAL_WEEKS)} validation label windows, "
                 f"red = outage days; data to {end.date()})", color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "activity_timeline.png", dpi=200)
    print("wrote", FIG / "activity_timeline.png")


def _heat(ax, M, title, fmt, vmax, diag=True):
    M = np.array(M, dtype=float)
    if not diag:
        np.fill_diagonal(M, np.nan)
    ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax)
    for i in range(6):
        for j in range(6):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if M[i, j] > 0.6 * vmax else INK)
    ax.set_xticks(range(6), SHORT, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(6), SHORT, fontsize=8.5)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)


def action_relations(last="2026-10-28"):
    """Wednesday cutoffs up to `last` (complete labels).  Residuals from the p3 backtest folds."""
    Y = [f"y_{t}" for t in TARGETS]
    d = pd.read_parquet(REL / "dataset_imp.parquet", columns=["cutoff", "uid"] + Y)
    d = d[(d.cutoff.dt.dayofweek == 2) & (d.cutoff <= pd.Timestamp(f"{last} 12:00"))]
    A = d[Y].to_numpy().astype(float)
    cond = [[A[A[:, i] == 1, j].mean() for j in range(6)] for i in range(6)]
    prev = d.assign(cutoff=d.cutoff + pd.Timedelta(days=7))
    m = d.merge(prev, on=["cutoff", "uid"], suffixes=("", "_p"))
    lift = [[(m.loc[(m[Y[i] + "_p"] == 1) & (m[Y[j] + "_p"] == 0), Y[j]].mean()
              / m.loc[m[Y[j] + "_p"] == 0, Y[j]].mean()) if i != j else np.nan for j in range(6)] for i in range(6)]
    P, L = [], []
    for f in [ART / "folds_a6_p3_imp.pkl"]:
        if f.exists():
            for _, p, y, _ in pickle.load(open(f, "rb")):
                P.append(p); L.append(y.astype(float))
    R = np.vstack(L) - np.vstack(P)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.9))
    _heat(axs[0], cond, "A  P(column this week | row this week)", "{:.2f}", 1.0, diag=False)
    _heat(axs[1], lift, "B  lift of column this week if row last week\n    (users without column last week)",
          "{:.1f}", 8.0, diag=False)
    _heat(axs[2], np.corrcoef(R.T), f"C  correlation of residuals y - p after the model\n"
          f"    ({len(R):,} validation user-weeks)", "{:.2f}", 0.5, diag=False)
    fig.suptitle(f"How the six actions relate ({len(d):,} user-weeks, {d.cutoff.nunique()} Wednesday cutoffs "
                 f"{d.cutoff.min().date()} .. {d.cutoff.max().date()})", color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "action_relations.png", dpi=200, bbox_inches="tight")
    print("wrote", FIG / "action_relations.png")


SIM_FILES = {"noimp": ["sim_push_outage_new_noimp.csv", "sim_push_outage_a6_noimp.csv"],
             "old": ["sim_push_outage_a6_old.csv", "sim_push_outage_new_imp.csv"],
             "new": ["sim_push_outage_a6_new.csv", "sim_push_outage_a6_new37.csv"]}


def sim_deltas(kind):
    """Change of the 5-activity score per (week, hole_days) against the intact run of the same files
    (first file holds hole_days 0).  old = imputation with the detection before addendum 6."""
    f = [pd.read_csv(RES / n) for n in SIM_FILES[kind]]
    base = f[0][f[0].hole_days == 0].set_index("week").S5
    d = pd.concat(f).query("hole_days > 0").drop_duplicates(["week", "hole_days"])
    return d.assign(d=d.S5 - d.week.map(base))


def outage_imputation(reh2=RES / "rehearsal_1106"):
    """A: score of the 5 non-push activities when the last N days of pushes before the cutoff are deleted
    (mean over 4 weeks, sim_push_outage.py on the GH Archive tables).  B: mean predicted probability of
    the 11-06 rehearsal (test users, pushes missing since 10-27)."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axs[0]
    for d0, lab, col, dy in [(sim_deltas("noimp"), "no imputation", ORANGE, -0.3),
                             (sim_deltas("old"), "imputation, detection before the fix", MUTED, 0),
                             (sim_deltas("new"), "imputation, corrected detection", BLUE, 0.3)]:
        g = d0.groupby("hole_days").d.mean()
        ax.plot(g.index, g.values, "-o", color=col, lw=2, ms=7, label=lab)
        ax.text(g.index[-1] + 0.6, g.values[-1] + dy, f"{g.values[-1]:+.1f}", va="center", color=INK, fontsize=9)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xticks([0, 3, 7, 14, 21, 28])
    ax.set_xlim(-0.5, 31.5)
    ax.set_xlabel("days of pushes missing just before the cutoff")
    ax.set_ylabel("change of the 5-activity score (points)")
    ax.grid(axis="y", color="#ecebe8", lw=0.8)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("A  Simulated push outage (mean of 4 validation weeks)", color=INK, fontsize=10, loc="left")
    ax = axs[1]
    p0 = pd.read_csv(reh2 / "prob_p3.csv")[TARGETS].mean()
    p1 = pd.read_csv(reh2 / "prob_p3_imp.csv")[TARGETS].mean()
    Y = [f"y_{t}" for t in TARGETS]
    d = pd.read_parquet(REL / "dataset_imp.parquet", columns=["cutoff"] + Y)
    rec = d[(d.cutoff.dt.dayofweek == 2) & (d.cutoff >= "2026-09-30") & (d.cutoff <= "2026-10-28 12:00")]
    r = rec.groupby("cutoff")[Y].mean()
    x = np.arange(6)
    ax.vlines(x, 100 * r.min().values, 100 * r.max().values, color=GREY, lw=9, label="recorded positive rate (labels), weeks 09-30 .. 10-28")
    ax.plot(x - 0.12, 100 * p0.values, "o", color=ORANGE, ms=8, label="predicted, no imputation")
    ax.plot(x + 0.12, 100 * p1.values, "o", color=BLUE, ms=8, label="predicted, outage imputation (corrected detection)")
    ax.set_xticks(x, SHORT, fontsize=9)
    ax.set_ylabel("mean probability / positive rate (%)")
    ax.set_ylim(0, None)
    ax.grid(axis="y", color="#ecebe8", lw=0.8)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("B  Rehearsal at 11-06 12:00 UTC, test users (pushes missing since 10-27)", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "outage_imputation.png", dpi=200)
    print("wrote", FIG / "outage_imputation.png")


def weekly_scores():
    """Outage-aware week scores (activities with an outage in the label window left out), from the
    fold pickles that results/forward_check_a6.json was computed with (addenda 5 and 6)."""
    import forward_decide as F
    fc = json.load(open(RES / "forward_check_a6.json"))
    F.DROP.update(fc["dropped"])
    S = {n: F.scores([f"a6_{n}_imp"], RES).sort_index() for n in ["v1", "p3", "p4"]}
    S = {n: s[~s.index.isin(F.EXCLUDE_WEEKS)] for n, s in S.items()}
    F.FRESH = fc["fresh"]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(len(S["v1"]))
    for (n, s), col, lab in zip(S.items(), [BLUE, ORANGE, AQUA],
                                ["v1 (baseline)", "p3 (per-activity XGB params, chosen)", "p4 (shallow XGB + NN 256)"]):
        ax.plot(x, s.values, "-o", color=col, lw=2, ms=6, label=lab)
    for i, w in enumerate(S["v1"].index):
        if w in F.FRESH:
            ax.axvspan(i - 0.5, i + 0.5, color=GREY, alpha=0.5, lw=0)
    lbl = [w[5:] + ("".join("\n−" + SHORT[TARGETS.index(k)] for k in fc["dropped"].get(w, []))
                    if fc["dropped"].get(w) else "") for w in S["v1"].index]
    ax.set_xticks(x, lbl, fontsize=8)
    ax.set_ylabel("Score (%)")
    ax.grid(axis="y", color="#ecebe8", lw=0.8)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    m = {n: s.mean() for n, s in S.items()}
    fig.suptitle(f"Weekly validation score, outage-aware (grey = fresh weeks; mean v1 {m['v1']:.2f}, "
                 f"p3 {m['p3']:.2f}, p4 {m['p4']:.2f})", color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "weekly_scores.png", dpi=200)
    print("wrote", FIG / "weekly_scores.png")


def ablation(rows: list[tuple[str, str]], fname="ablation.png", title=""):
    """rows: (experiment name in the ledger, display label), drawn top to bottom."""
    L = read_ledger().set_index("name")
    lab = [r[1] for r in rows]
    val = [float(L.loc[r[0], "honest_score"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(rows) + 1.3))
    y = np.arange(len(rows))[::-1]
    v100 = 100 * np.array(val)
    # dot plot: the axis does not start at zero, so no bars (length would mislead)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#ecebe8", lw=0.8)
    ax.hlines(y, v100.min() - 1, v100, color="#d9d8d4", lw=1)
    ax.plot(v100, y, "o", color=BLUE, ms=8)
    for yi, v in zip(y, v100):
        ax.text(v + 0.12, yi, f"{v:.2f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, lab)
    ax.set_xlim(v100.min() - 1, v100.max() + 0.8)
    ax.set_xlabel("validation Score (%) = (macro-F1 + micro-F1)/2, mean of 5 weekly folds")
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=200)
    print("wrote", FIG / fname)


def read_ledger() -> pd.DataFrame:
    return pd.read_csv(RES / "experiments.tsv", sep="\t")


if __name__ == "__main__":
    activity_timeline()
    action_relations()
    outage_imputation()
    weekly_scores()


def ablations_new():
    """Q2 / Q3 ablations on the GH Archive tables (ablation_new.sh; ledger rows new_*)."""
    ablation([
        ("new_abl_T0", "T0 counts 7 / 28 / 90 d"),
        ("new_abl_T0R1R2", "T0 + R1 + R2 (relational, no rich temporal)"),
        ("new_abl_T0T1", "T0 + T1 rich temporal"),
        ("new_abl_T0T1R1", "T0 + T1 + R1 user-centric relational"),
        ("new_abl_full", "T0 + T1 + R1 + R2 repo / collaborator 2-hop"),
        ("new_mlp_T0T1R1", "NN on T0 + T1 + R1, no message passing (MLP)"),
        ("new_gnn_T0T1R1", "NN on T0 + T1 + R1 + user-repo message passing (GNN)"),
    ], "ablation_relational.png", "Relational features and message passing (XGBoost depth 8 unless noted; seed sd about 0.08)")
    ablation([
        ("new_heuristic_last7d", "rule: activity in the last 7 days"),
        ("new_abl_T0", "T0 counts 7 / 28 / 90 d"),
        ("new_abl_T0T1", "+ T1 rich temporal (recency, weekly lags, decay, trend)"),
        ("new_abl_full", "+ R1 + R2 (all features)"),
        ("new_full_start0601", "all features, train only on cutoffs since 06-01"),
        ("new_full_hl90", "all features, recency weighting (half-life 90 d)"),
        ("new_full_xgb_d10", "all features, depth 10 (used in the final model)"),
    ], "ablation_temporal.png", "Temporal features and training window (XGBoost depth 8 unless noted)")


def data_period():
    """Q1: the dataset time period used by the model (GH Archive hours -> tables -> samples)."""
    D = lambda s: pd.Timestamp(s)
    rows = [  # label, start, end, colour, note
        ("GH Archive hours read by preprocess.py", D("2025-12-01"), D("2026-11-21 12:00"), GREY,
         f"2025-12-01 00:00 – 2026-11-21 12:00 UTC ({int((D('2026-11-21 12:00') - D('2025-12-01')) / pd.Timedelta(hours=1)):,} hourly files)"),
        ("Feature history (rows known before t)", D("2025-12-01"), D("2026-11-21 12:00"), BLUE,
         "every feature uses only rows with row time < cutoff t"),
        ("Training cutoffs t (every 3rd day, 12:00 UTC)", D("2026-01-08"), D("2026-11-13 12:00"), ORANGE,
         "submission: 01-08 .. 11-13 (104 cutoffs); training labels censored at the fitting time V (submission V = 11-21 12:00 UTC)"),
        ("Validation weeks (rolling origin)", None, None, AQUA,
         "one bar per Wednesday cutoff: {dev} development, {fresh} fresh (dark); 09-02 reported only (hatched)"),
        ("Prediction cutoff → test week", D("2026-11-21 12:00"), D("2026-11-28 16:00"), RED,
         "t = 11-21 12:00 UTC; test week 11-21 16:00 – 11-28 16:00 UTC"),
    ]
    from modeling import EXCLUDE_WEEKS
    fc = json.load(open(RES / "forward_check_b15.json"))          # weeks of the version comparison
    weeks, fresh = sorted(fc["dropped"]), set(fc["fresh"])
    n_dev = len([w for w in weeks if w not in fresh and w not in EXCLUDE_WEEKS])
    rows = [(l, a, b, c, n.format(dev=n_dev, fresh=len(fresh)) if a is None else n) for l, a, b, c, n in rows]
    fday = lambda x: (x - T0).days + (x - T0).seconds / 86400
    fig, ax = plt.subplots(figsize=(12, 3.6))
    for i, (lab, a, b, col, note) in enumerate(rows):
        y = len(rows) - 1 - i
        if a is None:   # validation: label window [V + 4 h, V + 4 h + 7 d) of each week
            for w in weeks:
                v = D(w) + pd.Timedelta(hours=16)
                ax.barh(y, 7, left=fday(v), height=0.5, color=col if w not in EXCLUDE_WEEKS else "white",
                        edgecolor=col, hatch="////" if w in EXCLUDE_WEEKS else None, lw=0.6,
                        alpha=1.0 if w in fresh else 0.55)
            a, b = D(weeks[0]), D(weeks[-1]) + pd.Timedelta(days=7, hours=16)
        else:
            ax.barh(y, fday(b) - fday(a), left=fday(a), color=col, height=0.5, alpha=0.9 if col != GREY else 0.6)
        ax.text(-3, y, lab, ha="right", va="center", color=INK, fontsize=10)
        late = a > D("2026-10-01") or i >= 2      # long notes end at their bar's right edge
        ax.text((b - T0).days if late else (a - T0).days + 2, y - 0.42, note, ha="right" if late else "left",
                va="center", color=MUTED, fontsize=8.5)
    months = pd.date_range("2025-12-01", "2026-12-01", freq="MS")
    ax.set_xticks((months - T0).days, [m.strftime("%b\n%Y") if m.month in (12, 1) else m.strftime("%b") for m in months],
                  fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(0, (D("2026-12-03") - T0).days)
    ax.set_ylim(-0.8, len(rows) - 0.5)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#ecebe8", lw=0.8)
    fig.suptitle("Dataset time period used by the model (all times UTC)", color=INK, fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "data_period.png", dpi=200, bbox_inches="tight")
    print("wrote", FIG / "data_period.png")
