"""Phase 2 figures (from results/phase2/experiments_p2.tsv and backtest CSVs).
  figures/p2_experiments.png   single-model experiments under the final decision rule
  figures/p2_weekly.png        v1 vs Phase-2 final, per validation week (main + extra weeks)
  figures/p2_decision.png      decision-rule loss decomposition (v1 vs Phase-2)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, RES

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#ecebe8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#b0afab",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
P2 = RES / "phase2"


def dotplot(labels, vals, fname, title, xlabel, ref=None):
    v = 100 * np.array(vals)
    y = np.arange(len(v))[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(v) + 1.3))
    ax.set_axisbelow(True); ax.grid(axis="x", color=GRID, lw=0.8)
    lo = v.min() - 0.8
    ax.hlines(y, lo, v, color="#d9d8d4", lw=1)
    ax.plot(v, y, "o", color=BLUE, ms=8)
    if ref is not None:
        ax.axvline(100 * ref, color=MUTED, lw=1, ls="--")
        ax.text(100 * ref, len(v) - 0.4, " reference", color=MUTED, fontsize=8, va="bottom")
    for yi, vi in zip(y, v):
        ax.text(vi + 0.06, yi, f"{vi:.2f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, labels); ax.set_xlim(lo, v.max() + 0.6); ax.set_xlabel(xlabel)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left"); fig.tight_layout()
    fig.savefig(FIG / fname, dpi=200); print("wrote", FIG / fname)


def experiments():
    L = pd.read_csv(P2 / "experiments_p2.tsv", sep="\t").drop_duplicates("name", keep="last").set_index("name")
    rows = [("A0_ref_censored", "reference: XGB d10, uniform weights"),
            ("A2_T2", "+ T2 temporal v2 (gaps, burstiness, streaks)"),
            ("A3_R3", "+ R3 threads + branch->PR features"),
            ("A1_N", "+ N regime-normalised features"),
            ("B1_cat", "CatBoost instead of XGBoost"),
            ("A8_qw_hard04", "drop drought label weeks (hard filter)"),
            ("A6_qw_pow05", "label-quality weights, sqrt"),
            ("A7_qw_cap2", "label-quality weights, cap 2"),
            ("A5_qw", "label-quality weights (linear, cap 1)"),
            ("A5_qw_s1", "label-quality weights, seed 1"),
            ("A12_qw_T2R3", "weights + T2 + R3"),
            ("A16_qw_multi", "weights + multi-task trees"),
            ("A14_qw_d6", "weights + depth 6"),
            ("A15_qw_d12", "weights + depth 12")]
    dotplot([r[1] for r in rows], [L.loc[r[0], "score"] for r in rows], "p2_experiments.png",
            "Phase 2: single XGBoost runs, joint plug-in rule, 5 weekly folds (seed sd ~ 0.1 pt)",
            "validation Score (%)", ref=L.loc["A0_ref_censored", "score"])


def weekly():
    a = pd.concat([pd.read_csv(RES / "final_backtest_v1_baseline.csv"), pd.read_csv(RES / "v1_extra.csv")])
    b = pd.concat([pd.read_csv(RES / "p2d_backtest.csv"), pd.read_csv(RES / "p2d_extra.csv")])
    a, b = a.set_index("val").sort_index(), b.set_index("val").sort_index()
    x = np.arange(len(a))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=0.8)
    ax.plot(x, 100 * a.score, "o-", color=MUTED, lw=1.5, ms=6, label="v1 ensemble (55.96)")
    ax.plot(x, 100 * b.score, "o-", color=BLUE, lw=2, ms=7, label="Phase 2c candidate p2d: 0.6 v1 + 0.2 CNN + 0.2 Transformer sequence models")
    ax.set_xticks(x, [d[5:] for d in a.index]); ax.set_ylabel("Score (%)")
    ax.axvspan(-0.5, 5.5, color="#f4f3f0", lw=0, zorder=0)
    ax.text(0, ax.get_ylim()[1] - 1, "extra weeks (May-Aug)", color=MUTED, fontsize=9)
    ax.text(6, ax.get_ylim()[1] - 1, "main weeks (09-02 = push outage)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.22, 0.88))
    fig.suptitle("Per-week validation Score, v1 vs Phase 2", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "p2_weekly.png", dpi=200); print("wrote", FIG / "p2_weekly.png")


def decision():
    a = pd.read_csv(P2 / "decision_rule_analysis_final_backtest_v1_baseline.csv").set_index("week").loc["mean (5 wk)"]
    b = pd.read_csv(P2 / "decision_rule_analysis_p2_xgbW5.csv").set_index("week").loc["mean (5 wk)"]
    cols = [("joint", "joint plug-in rule (deployable)"), ("joint_with_true_levels", "same rule, true weekly base rates"),
            ("oracle_thresholds", "oracle thresholds (upper bound)")]
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.set_axisbelow(True); ax.grid(axis="x", color=GRID, lw=0.8)
    y = np.arange(len(cols))[::-1]
    for s, col, lab, off in [(a, MUTED, "v1", 0.12), (b, BLUE, "Phase 2", -0.12)]:
        v = [100 * s[c] for c, _ in cols]
        ax.plot(v, y + off, "o", color=col, ms=8, label=lab)
        for yi, vi in zip(y + off, v):
            ax.text(vi + 0.05, yi, f"{vi:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [c[1] for c in cols]); ax.set_xlabel("Score (%), mean of 5 weekly folds")
    ax.set_xlim(None, ax.get_xlim()[1] + 0.3)
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Where the remaining decision loss comes from", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "p2_decision.png", dpi=200); print("wrote", FIG / "p2_decision.png")


if __name__ == "__main__":
    experiments()
    decision()
    weekly()
