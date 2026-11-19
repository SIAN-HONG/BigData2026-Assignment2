"""Score the corrected feature-group ablation (ablation_c15.sh) and the ensemble-member comparison.

  python src/ablation_report.py --prefix c15 --members b15_p3_imp

Per group: the probabilities of the three seeds are averaged per week (as the final tree members are),
then the label-free joint plug-in rule decides; activities with an outage day in a week's label window
are left out (forward_decide.outage_targets).  Also each seed alone (mean, sd).  Differences between
groups are paired by week: mean, weeks better, t (descriptive: 8 weeks, not independent samples).
Gains are conditional on the groups already present and do not add up.
Members: per-member probabilities of the formal pipeline backtest (censored labels), same weeks and
scoring: the three trees averaged, GNN, MLP, full ensemble.
Writes results/ablation_<prefix>.{md,csv} and figures/ablation_<prefix>.png.
"""
from __future__ import annotations

import argparse
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ART, RES, ROOT, TARGETS
from forward_decide import outage_targets
from modeling import joint_plugin_predict, official_score

GROUPS = ["T0", "T0T1", "T0T1R1", "T0T1R1R2", "T0R1R2"]
LABEL = {"T0": "T0", "T0T1": "T0+T1", "T0T1R1": "T0+T1+R1", "T0T1R1R2": "T0+T1+R1+R2 (all)", "T0R1R2": "T0+R1+R2 (no T1)"}
STEPS = [("T0", "T0T1", "add T1 (rich temporal) to T0"),
         ("T0T1", "T0T1R1", "add R1 (user-centric relational)"),
         ("T0T1R1", "T0T1R1R2", "add R2 (repo / collaborator, 2-hop)"),
         ("T0", "T0R1R2", "add R1+R2 without T1"),
         ("T0R1R2", "T0T1R1R2", "add T1 when R1+R2 are present")]


def md(df):
    rows = [list(map(str, df.columns)), ["---"] * len(df.columns)] + [[str(x) for x in r] for r in df.values.tolist()]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="c15")
    ap.add_argument("--members", default="b15_p3_imp")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    runs = {g: [pickle.load(open(ART / f"folds_{a.prefix}_{g}_s{s}.pkl", "rb")) for s in seeds] for g in GROUPS}
    weeks = [str(v.date()) for v, *_ in runs[GROUPS[0]][0]]
    expect = ["2026-09-09", "2026-09-16", "2026-09-23", "2026-09-30", "2026-10-07", "2026-10-14", "2026-10-21", "2026-10-28"]
    if a.prefix == "c15" and weeks != expect:
        raise SystemExit(f"weeks {weeks} != the agreed 8 weeks")
    ref = runs[GROUPS[0]][0]
    for g in GROUPS:          # every group and seed: same weeks, users, labels; finite probabilities
        for r in runs[g]:
            for (v0, _, Y0, u0), (v, P, Y, u) in zip(ref, r):
                if v != v0 or not np.array_equal(u, u0) or not np.array_equal(Y, Y0) or not np.isfinite(P).all():
                    raise SystemExit(f"{g}: week {v.date()} differs from {GROUPS[0]} in users / labels or has non-finite P")
    data_end = pd.Timestamp(pd.read_parquet(ART / "events.parquet", columns=["ts"]).ts.max(), unit="s") + pd.Timedelta(seconds=1)
    drop = outage_targets(weeks, data_end)

    def score(P, Y, w):
        keep = [j for j, k in enumerate(TARGETS) if k not in drop[w]]
        return official_score(joint_plugin_predict(P)[:, keep], Y[:, keep])["score"] * 100

    rows = []
    for g in GROUPS:
        for i, w in enumerate(weeks):
            fs = [r[i] for r in runs[g]]
            if any(str(f[0].date()) != w or not np.array_equal(f[3], fs[0][3]) for f in fs):
                raise SystemExit(f"{g} {w}: seeds disagree on week or user order")
            Y = fs[0][2]
            rows.append({"group": g, "week": w, "avg_seeds": score(np.mean([f[1] for f in fs], axis=0), Y, w),
                         **{f"seed{s}": score(f[1], Y, w) for s, f in zip(seeds, fs)}})
    A = pd.DataFrame(rows)
    A.to_csv(RES / f"ablation_{a.prefix}.csv", index=False)
    sc = [f"seed{s}" for s in seeds]
    T = A.groupby("group").agg(avg_seeds=("avg_seeds", "mean"), **{c: (c, "mean") for c in sc}).reindex(GROUPS)
    T["seed_sd"] = T[sc].std(axis=1)
    T.insert(0, "features", [LABEL[g] for g in GROUPS])
    W = A.pivot(index="week", columns="group", values="avg_seeds")
    D = []
    for g0, g1, txt in STEPS:
        d = W[g1] - W[g0]
        D.append({"step": txt, "from": LABEL[g0], "to": LABEL[g1], "mean_delta": d.mean(), "median_delta": d.median(),
                  "largest_week": d.idxmax(), "largest_delta": d.max(),
                  "mean_delta_without_largest": d.drop(d.idxmax()).mean(),
                  "weeks_better": f"{(d > 0).sum()} of {len(d)}", "t": d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))})
    D = pd.DataFrame(D)

    # per activity: F1 of each activity under the same joint decisions (k chosen for all six together,
    # so an activity's F1 also depends on the others' probabilities); outage activity-weeks left out
    from sklearn.metrics import f1_score
    prow = []
    for g in GROUPS:
        for i, w in enumerate(weeks):
            fs = [r[i] for r in runs[g]]
            P, Y = np.mean([f[1] for f in fs], axis=0), fs[0][2]
            dec = joint_plugin_predict(P)
            for j, k in enumerate(TARGETS):
                if k not in drop[w]:
                    prow.append({"group": g, "week": w, "activity": k, "f1": 100 * f1_score(Y[:, j], dec[:, j], zero_division=0)})
    PA = pd.DataFrame(prow).pivot_table(index=["activity", "week"], columns="group", values="f1")
    AD = []
    for g0, g1, txt in STEPS[:3]:
        d = (PA[g1] - PA[g0]).groupby(level="activity")
        for k, x in d:
            AD.append({"step": txt, "activity": k, "mean_f1_delta": x.mean(), "weeks_better": int((x > 0).sum()), "weeks": len(x)})
    AD = pd.DataFrame(AD)
    AD.to_csv(RES / f"ablation_{a.prefix}_activity.csv", index=False)

    # ensemble members of the formal pipeline backtest, same weeks
    M = {str(v.date()): (mem, Y) for v, mem, Y, _ in pickle.load(open(ART / f"members_{a.members}.pkl", "rb"))}
    mrows = []
    for w in weeks:
        mem, Y = M[w]
        mrows.append({"week": w, "trees (3 seeds averaged)": score(np.mean(mem[:3], axis=0), Y, w),
                      "GNN": score(mem[3], Y, w), "MLP": score(mem[4], Y, w), "ensemble": score(np.mean(mem, axis=0), Y, w)})
    Mm = pd.DataFrame(mrows)
    Mm.to_csv(RES / f"ablation_{a.prefix}_members.csv", index=False)
    D.to_csv(RES / f"ablation_{a.prefix}_steps.csv", index=False)
    T.to_csv(RES / f"ablation_{a.prefix}_groups.csv")
    mm = Mm.drop(columns="week").mean()

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.1), gridspec_kw={"width_ratios": [1, 1.15]})
    ys = np.arange(len(GROUPS))[::-1]           # dot plot: the axis does not start at 0, so no bars
    ax[0].scatter(T.avg_seeds, ys, color=["#4c78a8"] * 4 + ["#c9a27a"], s=40, zorder=3)
    for y, (g, v) in zip(ys, T.avg_seeds.items()):
        ax[0].text(v + 0.08, y, f"{v:.2f}", va="center", fontsize=10)
        for sd in [c for c in T.columns if c.startswith("seed") and c != "seed_sd"]:
            ax[0].scatter(T.loc[g, sd], y, color="#999999", s=8, zorder=2)
    ax[0].set_yticks(ys, [LABEL[g] for g in GROUPS], fontsize=10)
    ax[0].set_xlim(T.avg_seeds.min() - 1, T.avg_seeds.max() + 1)
    ax[0].set_xlabel("score (%), mean of 8 weeks (axis does not start at 0)")
    ax[0].set_title("Feature groups: 3 seeds averaged (dots), single seeds (grey)", fontsize=11)
    ax[0].grid(axis="x", color="#eeeeee")
    # per activity: mean ΔF1 of the two relational steps (the per-week view is in the text: weeks better)
    acts = list(TARGETS)[::-1]
    ya = np.arange(len(acts))
    for off, (step_txt, lab, col, mk) in zip((0.17, -0.17), [
            (STEPS[1][2], "+R1 on T0+T1", "#2a78d6", "o"), (STEPS[2][2], "+R2 on T0+T1+R1", "#eb6834", "D")]):
        s = AD[AD.step == step_txt].set_index("activity").reindex(acts)
        ax[1].scatter(s.mean_f1_delta, ya + off, color=col, marker=mk, s=34, zorder=3, label=lab,
                      edgecolors="white", linewidths=1)
        for y, (v, b, n) in zip(ya + off, s[["mean_f1_delta", "weeks_better", "weeks"]].itertuples(index=False)):
            ax[1].text(v + (0.07 if v >= 0 else -0.07), y, f"{b}/{n} wk", va="center", ha="left" if v >= 0 else "right",
                       fontsize=9, color="#555555")
    ax[1].axvline(0, color="#888888", lw=1)
    ax[1].set_yticks(ya, acts, fontsize=10)
    rel = AD[AD.step.isin([STEPS[1][2], STEPS[2][2]])].mean_f1_delta
    ax[1].set_xlim(rel.min() - 0.6, rel.max() + 0.6)
    ax[1].set_xlabel("mean ΔF1 (percentage points); label = weeks better / weeks scored")
    ax[1].grid(axis="x", color="#eeeeee")
    ax[1].legend(fontsize=11, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
                 borderaxespad=0.2, title="Per activity (push: 3 weeks scored)", title_fontsize=11,
                 alignment="left")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"ablation_{a.prefix}.png", dpi=180)

    lines = [f"# Corrected feature-group ablation ({a.prefix})", "",
             f"Weeks ({len(weeks)}): {', '.join(weeks)}. Outage activity-weeks left out: "
             + ("; ".join(f"{w}: {', '.join(d)}" for w, d in drop.items() if d) or "none") + ".",
             "XGBoost depth 10, min_child_weight 50 (v1 tree parameters), every 3rd cutoff, training labels censored at "
             "each validation cutoff, dataset_imp.parquet; seeds averaged in probability, then the joint plug-in rule.", "",
             md(T.reset_index().round(2)), "",
             "Paired by week (seed-averaged; gains are conditional on the groups present and do not add up; "
             "t is descriptive):", "", md(D.round(2)), "",
             "Per activity (F1 in points of each activity under the same joint decisions; descriptive):", "",
             md(AD.round(2)), "",
             f"Ensemble members of the formal backtest ({a.members}, censored labels), same weeks and scoring:", "",
             md(Mm.round(2)), "", "Mean: " + ", ".join(f"{k} {v:.2f}" for k, v in mm.items())]
    (RES / f"ablation_{a.prefix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
