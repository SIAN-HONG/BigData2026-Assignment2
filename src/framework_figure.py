"""Q4 framework figures.  Every box of the detailed figure names the file/function that implements it.
python src/framework_figure.py [--config configs/final_config_p3_imp.json]
    -> figures/framework.{svg,png,pdf}  (graphviz, detailed)
    -> figures/framework_slide.{png,pdf}  (matplotlib, fixed layout for the slide: three parallel paths)
The tree-model box is written from the config that is actually used (so the figure follows the
version decided under docs/FORWARD_CHECK.md addendum 7).
"""
import argparse
import json

from common import ROOT
from graphs import render


def xgb_label(cfg: dict, slide: bool = False) -> str:
    per = cfg.get("xgb_per_target") or {}
    base = cfg["xgb_params"]
    if per:
        rare = {v.get("params", {}).get("max_depth") for k, v in per.items() if k != "push"}
        push = per.get("push", {})
        d = (f"per-activity params: depth {'/'.join(map(str, sorted(rare)))} (5 rarer),\n"
             f"depth {push.get('params', {}).get('max_depth')}"
             + (" + label-quality weights" if push.get("weights", "").startswith("quality") else "") + " (push)")
    else:
        d = f"depth {base['max_depth']}, min_child_weight {base['min_child_weight']}" + (
            ", early stopping" if base.get("early_stop") else "")
    n = len(cfg["xgb_seeds"])
    if slide:
        return f"XGBoost x {n} per activity"
    return f"XGBoost x {n} seeds, one model per activity\n{d}\ninput " + "+".join(cfg["xgb_groups"])

DOT = r'''
digraph F {
graph [rankdir=TB, fontname="Helvetica", nodesep=0.3, ranksep=0.35, splines=true, compound=true, newrank=true,
       label="Framework: point-in-time features with outage imputation + tree / relational-NN ensemble + label-free decision rule", labelloc=t, fontsize=17];
node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, fillcolor="#ffffff", color="#6b6a66"];
edge [color="#6b6a66", arrowsize=0.7];

subgraph cluster_in { label="1  Input: course preprocess.py on GH Archive hours"; style="rounded,filled"; fillcolor="#eef4fc"; color="#c9d9ef"; fontsize=12;
  db [label="8 Parquet tables\nUsers, Repositories, PullRequests, Issues,\nIssueComments, PRReviews, PRReviewComments, Pushes\n(UTC, FK = row index)"];
}
subgraph cluster_pre { label="2  Preprocessing  (common.py)"; style="rounded,filled"; fillcolor="#eef7f2"; color="#c6e3d3"; fontsize=12;
  evlog [label="Unified event log (build_events)\nderived rows: user, repo, kind, ts, avail\n6 target activities + 7 relational kinds\n(merged_by, comments/reviews received, own PR/issue closed)"];
  pit [label="Point-in-time rule\nuse row only if avail < t\navail = first_seen_at or created_at;\nmerged_at, closed_at, dismissed_at only if < t\n(tables keep the latest metadata: see limits)"];
  lab [label="Labels (label_matrix, raw events)\nactivity in [t+4h, t+4h+7d);\nour 5-day availability slack;\ntraining labels censored at V (censor_labels)"];
  imp [label="Outage imputation (features.impute_outages)\nday count < 10% of the reference = outage\n(common.outage_flags: median of the latest 28\nnon-outage days within 56 d, needs >= 7);\nfill the hole with the same activity's events\nfrom the same span whole weeks earlier\n(point-in-time, only rows known before t)", fillcolor="#fbe9e7"];
  sel [label="Samples (build_dataset.py --impute)\none row per (user, daily cutoff t = 12:00 UTC) from 01-07\nusers: >=3 raw-event records in the 90 d before t-17d\n(eligible_users; the rule that gives test_input.csv)\neach row: features at t + labels"];
}
subgraph cluster_feat { label="3  Feature construction at cutoff t  (features.py)"; style="rounded,filled"; fillcolor="#fdf6e7"; color="#efdcae"; fontsize=12;
  t0 [label="T0  basic temporal\ncounts 7/28/90d per activity"];
  t1 [label="T1  rich temporal\n1/3/14/56d + all-time counts, recency,\nweekly-lag indicators (8 wk), active days,\ndecayed counts (3d/14d half-life), trend"];
  r1 [label="R1  user-centric relational\nreviews/comments received on own PRs/issues,\nPRs merged by the user, own open PRs/issues\n(first seen within 90 d; PR / issue state from the tables)"];
  r2 [label="R2  repo / collaborator (2-hop)\nuser->repo->users aggregates, open-PR\nreview demand, collaborators' activity\n(repo metadata from the tables)"];
  leak [label="Leakage audit (leakage_audit.py)\nfeatures from tables truncated at t vs full tables\n(with imputation): no column differs > 1e-4", fillcolor="#fbe9e7"];
}
subgraph cluster_model { label="4  Models  (pipeline.fit_predict; all trained on the censored labels of the samples)"; style="rounded,filled"; fillcolor="#f3effb"; color="#d8cff0"; fontsize=12;
  xgb [label="XGB_LABEL"];
  gnn [label="Relational GNN (gnn.py)\nuser <-> repo graph per cutoff: raw events known before t,\nlast 28 d, sample users; 7 log-count edge channels\n(6 activities + merged_by) + repo aggregates;\nuser->repo->user message passing, 6-label head\ninput NN_GROUPS user features, NN_HIDDEN"];
  mlp [label="MLP baseline (no message passing)\ninput NN_GROUPS, NN_HIDDEN"];
  avg [label="AVG_LABEL"];
}
subgraph cluster_val { label="5  Training / validation"; style="rounded,filled"; fillcolor="#f1f0ed"; color="#d9d8d4"; fontsize=12;
  roll [label="Rolling origin: Wednesday weeks V (05-06 .. 11-04)\ntrain on every 3rd cutoff <= V - 8d,\nlabels censored at V; score = (macro-F1 + micro-F1) / 2,\nactivities with an outage in the label window left out;\nversion fixed by a pre-registered rule"];
}
subgraph cluster_out { label="6  Decision + output  (predict.py)"; style="rounded,filled"; fillcolor="#e9f5fb"; color="#c4e0ee"; fontsize=12;
  dec [label="Label-free joint plug-in rule\nchoose top-k per activity by a plug-in\napproximation of the official score from the\nmodel's own probabilities (no labels of week t)"];
  out [label="output_<ID>.csv\n0/1 x 6 per test user\nchecked by official evaluate.py"];
}
{rank=same; t0; t1; r1; r2; leak}
{rank=same; xgb; gnn; mlp; roll}
{rank=same; evlog; lab; pit}
{rank=same; imp; sel}
t0 -> t1 -> r1 -> r2 -> leak [style=invis];
db -> evlog; evlog -> pit; evlog -> lab; pit -> imp; imp -> sel; lab -> sel;
sel -> t0; sel -> t1; sel -> r1; sel -> r2;
t0 -> xgb; t1 -> xgb; r1 -> xgb; r2 -> xgb;
t0 -> gnn; t1 -> gnn; r1 -> gnn;
pit -> gnn [style=dashed, constraint=false, label=" raw events known before t: graph edges", fontsize=9]; t0 -> mlp; t1 -> mlp; r1 -> mlp;
xgb -> avg; gnn -> avg; mlp -> avg; avg -> dec -> out;
evlog -> sel [style=dashed, constraint=false, label=" user selection (raw events)", fontsize=9];
roll -> xgb [style=dashed, constraint=false, label=" protocol", fontsize=9];
}
'''

def nn_text(cfg: dict) -> str:
    """NN width / depth: pipeline defaults 128 / 1 when the config does not set them (pipeline.py)."""
    return f"hidden {cfg.get('nn_hidden', 128)}" + (f" x {cfg['nn_layers']} layers" if cfg.get("nn_layers", 1) > 1 else "")


def avg_text(cfg: dict) -> str:
    """pipeline.fit_predict: one probability vector per XGB seed and per NN variant and seed, equal mean."""
    nx, nv = len(cfg["xgb_seeds"]), len(cfg["nn_variants"]) * len(cfg.get("nn_seeds", [0]))
    names = " + ".join(v.upper() for v in cfg["nn_variants"])
    return f"Equal mean of {nx + nv} probability vectors\n({nx} XGB + {names})"


def xgb_short(cfg: dict) -> str:
    per = cfg.get("xgb_per_target") or {}
    base = cfg["xgb_params"]
    if per:
        rare = sorted({v["params"]["max_depth"] for k, v in per.items() if k != "push"})
        push = per.get("push", {})
        w = " + label-quality weights" if push.get("weights", "").startswith("quality") else ""
        return f"depth {'/'.join(map(str, rare))} (5 rarer activities),\ndepth {push['params']['max_depth']} (push{w})"
    return f"depth {base['max_depth']}" + (", early stopping" if base.get("early_stop") else "")


def slide_figure(cfg: dict):
    """Slide version, fixed layout: input -> three paths (features / graph / labels) -> models -> decision.
    Arrows are placed so that none cross; tags say which question discusses the box."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    fig, ax = plt.subplots(figsize=(13, 5.3))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.3)
    ax.axis("off")
    INK, EDGE = "#222222", "#6b6a66"
    TAG = {"Q2": "#2a78d6", "Q3": "#eb6834", "Q4": "#555555"}
    # stage colours (same hues as the schema / entity graphs; white text contrast >= 4.5:1)
    C_IN, C_EV, C_FE, C_GR, C_LB, C_MD, C_OUT = "#2f67ad", "#2e7d55", "#8c6a00", "#7558ad", "#2e7d55", "#4f5d6b", "#226e8a"

    def box(x0, y0, x1, y1, title, body="", fill="#ffffff", tags=(), fs=11, accent=None):
        # styling only: a coloured title bar (accent) clipped to the rounded box; text and positions unchanged
        ac = accent or EDGE
        ax.add_patch(FancyBboxPatch((x0 + 0.03, y0 - 0.04), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="#00000014", ec="none", zorder=1.5))   # soft shadow
        shape = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.08",
                               fc=fill, ec=ac, lw=1.3, zorder=2)
        ax.add_patch(shape)
        if title:
            bar = Rectangle((x0, y1 - 0.36), x1 - x0, 0.36, fc=ac, ec="none", zorder=2.1)
            ax.add_patch(bar)
            bar.set_clip_path(shape)
            ax.text((x0 + x1) / 2, y1 - 0.18, title, ha="center", va="center", fontsize=fs + 0.5, weight="bold",
                    color="white", zorder=3)
        if body:
            ax.text((x0 + x1) / 2, y1 - 0.47, body, ha="center", va="top", fontsize=fs, color=INK,
                    linespacing=1.2, zorder=3)
        for i, t in enumerate(tags):
            ax.text(x1 - 0.06 - 0.42 * i, y1 + 0.02, t, ha="right", va="center", fontsize=10, color="white",
                    weight="bold", zorder=4, bbox=dict(boxstyle="round,pad=0.18", fc=TAG[t], ec="white", lw=1))
        return (x0, y0, x1, y1)

    def arrow(p, q, style="-", label=None, lpos=0.5, ldy=0.12, color=EDGE):
        ax.annotate("", xy=q, xytext=p, zorder=1,
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color=color, linestyle=style, shrinkA=0, shrinkB=0,
                                    mutation_scale=13))
        if label:
            ax.text(p[0] + (q[0] - p[0]) * lpos, p[1] + (q[1] - p[1]) * lpos + ldy, label, ha="center", va="bottom",
                    fontsize=10, color="#444444", style="italic")

    # column 1: input
    tb = box(0.1, 3.75, 2.45, 4.85, "8 tables", "GH Archive hours ->\ncourse preprocess.py", fill="#eef4fc", accent=C_IN)
    ev = box(0.1, 1.95, 2.45, 3.35, "Event log", "action + known-at time;\n6 target + 7 relational\nkinds (FK joins)",
             fill="#eef7f2", tags=("Q2",), accent=C_EV)
    arrow((1.275, 3.75), (1.275, 3.35))
    ax.text(1.275, 1.75, "one sample per (user,\ndaily cutoff t = 12:00 UTC);\ntest week from t + 4 h",
            ha="center", va="top", fontsize=10, color="#444444", linespacing=1.2)

    # column 2: three parallel paths
    fe = box(2.95, 3.75, 6.15, 4.85, "Features at t",
             "known before t -> imputation\nT0/T1 temporal | R1/R2 relational\n(events + table state/metadata)",
             fill="#fdf6e7", tags=("Q3", "Q2"), accent=C_FE)
    gr = box(2.95, 2.3, 6.15, 3.35, "User–repo graph per t",
             "raw events known before t,\nlast 28 d; 7 log-count edge\nchannels + repo aggregates",
             fill="#f3effb", tags=("Q2",), accent=C_GR)
    lb = box(2.95, 0.85, 6.15, 1.9, "Labels (raw events)",
             "activity in [t+4h, t+4h+7d),\nour 5-day slack; training\nlabels censored at V",
             fill="#eef7f2", tags=("Q3",), accent=C_LB)
    arrow((2.45, 4.45), (2.95, 4.45))
    ax.text(2.7, 4.33, "state,\nmetadata", ha="center", va="top", fontsize=9, color="#444444", style="italic",
            linespacing=1.1, zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    arrow((2.45, 3.2), (2.95, 3.9))
    arrow((2.45, 2.8), (2.95, 2.8))
    arrow((2.45, 2.1), (2.95, 1.45))

    # column 3: training container with the three models
    ax.add_patch(FancyBboxPatch((6.6, 0.85), 3.05, 4.15, boxstyle="round,pad=0,rounding_size=0.1",
                                fc="#fafaf8", ec="#b9b8b2", lw=1, ls="--", zorder=1))
    ax.text(6.7, 4.95, "Training (all models)", ha="left", va="top", fontsize=10, color="#555555")
    g = "+".join(cfg["xgb_groups"])
    n = "+".join(cfg["nn_groups"])
    xg = box(6.8, 3.6, 9.45, 4.65, f"XGBoost x {len(cfg['xgb_seeds'])} seeds", f"one model per activity;\ninput {g}", accent=C_MD)
    ml = box(6.8, 2.65, 9.45, 3.35, "MLP (no msg. passing)", f"input {n}", accent=C_MD)
    gn = box(6.8, 1.3, 9.45, 2.5, "Relational GNN", f"{n} user features\n+ graph; user -> repo -> user",
             tags=("Q2",), accent=C_MD)
    arrow((6.15, 4.3), (6.8, 4.2))
    arrow((6.15, 4.05), (6.8, 3.0))
    arrow((6.15, 3.85), (6.8, 2.25))
    arrow((6.15, 2.6), (6.8, 1.75))
    arrow((6.15, 1.05), (6.6, 1.05), style="--", label="targets", ldy=0.03)

    # column 4: combine, decide, output
    nx = len(cfg["xgb_seeds"]); nv = len(cfg["nn_variants"]) * len(cfg.get("nn_seeds", [0]))
    av = box(10.15, 3.75, 12.9, 4.85, "Average", f"equal mean of {nx + nv}\nprobability vectors\n({nx} XGB + GNN + MLP)", fill="#e9f5fb", accent=C_OUT)
    dc = box(10.15, 2.3, 12.9, 3.35, "Label-free decision", "joint plug-in rule: top-k\nper activity, no labels\nof the predicted week",
             fill="#e9f5fb", tags=("Q3",), accent=C_OUT)
    oc = box(10.15, 0.85, 12.9, 1.9, "output_<ID>.csv", "4,465 users x 6;\nformat + user-set check", fill="#e9f5fb", accent=C_OUT)
    arrow((9.45, 4.2), (10.15, 4.3))
    arrow((9.45, 3.0), (10.15, 4.0))
    arrow((9.45, 1.9), (10.15, 3.85))
    arrow((11.525, 3.75), (11.525, 3.35))
    arrow((11.525, 2.3), (11.525, 1.9))

    # bottom: validation / selection
    box(0.1, 0.05, 12.9, 0.62, "", fill="#f1f0ed", tags=("Q4",))
    ax.text(6.5, 0.335, "Validation and selection: for each Wednesday V, train on cutoffs <= V - 8 d (labels censored at V), "
            "score week V outage-aware;\nversion fixed by a pre-registered rule.   Checks: leakage audit (page 7), "
            "reproducibility manifest (page 13).",
            ha="center", va="center", fontsize=10.5, color=INK, linespacing=1.3)
    ax.text(12.9, 5.27, "tags: Q2 relational  ·  Q3 temporal  ·  Q4 validation", ha="right", va="top", fontsize=10,
            color="#555555")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"framework_slide.{ext}", dpi=220 if ext == "png" else None)
    plt.close(fig)
    print("wrote", ROOT / "figures" / "framework_slide.png/.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final_config_p3_imp.json")
    a = ap.parse_args()
    cfg = json.load(open(ROOT / a.config))
    render(DOT.replace("XGB_LABEL", xgb_label(cfg)).replace("NN_GROUPS", "+".join(cfg["nn_groups"]))
           .replace("NN_HIDDEN", nn_text(cfg)).replace("AVG_LABEL", avg_text(cfg)), "framework")
    slide_figure(cfg)
