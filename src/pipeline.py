"""The final method as ONE function, used both for the rolling backtest and for the
official prediction, so the validated code path is exactly the submitted one.

fit_predict(df, ev, cut, target_rows, cfg):
  1. training samples = every (user, cutoff) with cutoff <= cut - 8 days (label windows
     complete before `cut`), cutoffs spaced cfg["every"] days
  2. gradient-boosted trees (XGBoost, GPU) on all feature groups, several seeds
  3. relational neural nets (user<->repo GNN and plain MLP), several seeds
  4. average the probabilities of all members
  5. label-free joint plug-in decision rule maximising the expected official score
Backtest:  python src/pipeline.py --backtest
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd

from common import ART, RES, ROOT, TARGETS
from features import feature_groups
from modeling import (EXCLUDE_WEEKS, YCOLS, censor_labels, feature_cols, joint_plugin_predict, official_score,
                      outage_label_mask, quality_weights,
                      plugin_predict, predict, train_models, train_window)

DEFAULT_CFG = {
    "dataset": "dataset.parquet",
    "weights": "",
    "xgb_groups": ["T0", "T1", "R1", "R2"],
    "xgb_params": {},
    "xgb_seeds": [0, 1, 2],
    "nn_groups": ["T0", "T1", "R1"],
    "nn_variants": ["gnn", "mlp"],
    "nn_seeds": [0],
    "nn_epochs": 8,
    "every": 3,
    "decision": "joint_plugin",
}


def load_cfg(path=None):
    p = ROOT / (path or "final_config.json")
    if not p.is_file():  # a mistyped path must not silently fall back to the defaults
        raise FileNotFoundError(f"config file not found: {p}")
    return {**DEFAULT_CFG, **json.loads(p.read_text())}


def fit_predict(df, ev, cut, target_rows, cfg, log=print):
    """Probabilities (n_target x 6) for target_rows (features at cutoff `cut`)."""
    import gnn  # torch only needed here
    cut = pd.Timestamp(cut)
    G = feature_groups(feature_cols(df))
    members = []
    # ---- trees ---------------------------------------------------------------------------
    fx = [f for g in cfg["xgb_groups"] for f in G[g]]
    tr = censor_labels(train_window(df, cut, every=cfg["every"]), ev, cut)
    W = quality_weights(tr, cfg["weights"]) if cfg.get("weights", "").startswith("quality") else None
    # optional: drop (weight 0) the labels whose window overlaps an outage of that activity
    M = outage_label_mask(tr, ev, cut) if cfg.get("label_mask") == "outage" else None
    mask = (lambda Wd: Wd if M is None else {c: (1.0 if Wd is None else Wd[c]) * M[c] for c in TARGETS})
    W = mask(W)
    per = cfg.get("xgb_per_target")   # optional {activity: {"params": {...}, "weights": "quality" | ""}}
    for sd in cfg["xgb_seeds"]:
        if per:
            models = {}
            for c in TARGETS:
                spec = per.get(c, {})
                w = spec.get("weights", cfg.get("weights", ""))
                Wc = mask(quality_weights(tr, w) if w.startswith("quality") else None)
                models.update(train_models(tr, fx, {**cfg["xgb_params"], **spec.get("params", {})},
                                           targets=[c], seed=sd, weights=Wc))
        else:
            models = train_models(tr, fx, cfg["xgb_params"], seed=sd, weights=W)
        members.append(predict(models, target_rows, fx))
    log(f"    xgb x{len(cfg['xgb_seeds'])}: {len(tr)} rows, cutoffs {tr.cutoff.min().date()}..{tr.cutoff.max().date()}"
        + ("" if M is None else "; outage-masked label share " + ", ".join(f"{c} {1 - M[c].mean():.2f}" for c in TARGETS)))
    # ---- relational neural nets ---------------------------------------------------------
    if cfg["nn_variants"]:
        fn = [f for g in cfg["nn_groups"] for f in G[g]]
        train_cuts = sorted(tr.cutoff.unique())
        graphs = gnn.build_graphs(tr, fn, ev, train_cuts + [cut], rows={cut: target_rows})
        for var in cfg["nn_variants"]:
            for sd in cfg["nn_seeds"]:
                members.append(gnn.run_fold(graphs, train_cuts, cut, var, cfg["nn_epochs"], "cuda", sd,
                                            qweights=cfg.get("nn_weights", cfg.get("weights", "")).startswith("quality"),
                                            hidden=cfg.get("nn_hidden", 128), layers=cfg.get("nn_layers", 1)))
        log(f"    nn {cfg['nn_variants']} x{len(cfg['nn_seeds'])}: {len(train_cuts)} graph snapshots")
    P = np.mean(members, axis=0)
    # ---- optional sequence-model members (Phase 2c), blended with weight cfg["seq_weight"] ---
    if cfg.get("seq_members"):
        import seqmodel
        cache = seqmodel.SeqCache(ev)
        fs = [f for g in cfg.get("seq_groups", ["T0", "T1", "R1"]) for f in G[g]]
        seq_P = []
        for sm in cfg["seq_members"]:
            p_, _ = seqmodel.run_fold(df, cache, cut, fs, every=cfg["every"], weights=cfg.get("seq_weights", "quality"),
                                      epochs=sm.get("epochs", 4), seed=sm.get("seed", 0), arch=sm["arch"],
                                      hidden=sm.get("hidden", 64), te=target_rows, tr=tr)
            seq_P.append(p_)
            members.append(p_)
        log(f"    seq {[s['arch'] for s in cfg['seq_members']]}")
        P = (1 - cfg["seq_weight"]) * P + cfg["seq_weight"] * np.mean(seq_P, axis=0)
    return P, members


def decide(P, cfg):
    return joint_plugin_predict(P) if cfg["decision"] == "joint_plugin" else plugin_predict(P)


def val_weeks(val):
    """'main' | 'extra' | comma-separated dates (validation cutoffs at 12:00 UTC)."""
    from exp import VAL
    from exp2 import VAL_EXTRA
    if val in ("main", "extra"):
        return VAL if val == "main" else VAL_EXTRA
    return [pd.Timestamp(f"{d} 12:00") for d in val.split(",")]


def backtest(name="final_backtest", config=None, val="main"):
    cfg = load_cfg(config)
    df = pd.read_parquet(ART / cfg["dataset"])
    ev = pd.read_parquet(ART / "events.parquet")
    rows, folds, all_members = [], [], []
    t0 = time.time()
    for v in val_weeks(val):
        va = df[df.cutoff == v]
        if not len(va):
            print(f"  {v.date()}  no labelled samples in this release, skipped", flush=True)
            continue
        P, members = fit_predict(df, ev, v, va, cfg)
        Y = va[YCOLS].to_numpy()
        folds.append((v, P, Y, va.UserID.to_numpy()))
        all_members.append((v, members, Y, va.UserID.to_numpy()))
        r = {"val": str(v.date()), **official_score(decide(P, cfg), Y)}
        r["per_target_plugin_score"] = official_score(plugin_predict(P), Y)["score"]
        nx = len(cfg["xgb_seeds"])   # members: xgb..., nn..., seq...
        r["xgb_only_score"] = official_score(decide(np.mean(members[:nx], 0), cfg), Y)["score"] if nx else np.nan
        rows.append(r)
        print(f"  {v.date()}  score {r['score']:.4f}  (macro {r['macro_f1']:.4f}, micro {r['micro_f1']:.4f})", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(RES / f"{name}.csv", index=False)
    pickle.dump(folds, open(ART / f"folds_{name}.pkl", "wb"))
    pickle.dump(all_members, open(ART / f"members_{name}.pkl", "wb"))   # per-member probabilities
    E = R[~R.val.isin(EXCLUDE_WEEKS)]
    summ = {"name": name, "cfg": cfg, "weeks": list(E.val),
            "score": E.score.mean(), "macro_f1": E.macro_f1.mean(), "micro_f1": E.micro_f1.mean(),
            "score_std_over_weeks": E.score.std(), "per_target_plugin_score": E.per_target_plugin_score.mean(),
            "xgb_only_score": E.xgb_only_score.mean(), "minutes": (time.time() - t0) / 60,
            "excluded_outage_week": R[R.val.isin(EXCLUDE_WEEKS)].to_dict("records")}
    json.dump(summ, open(RES / f"{name}.json", "w"), indent=2, default=float)
    print(R.round(4).to_string())
    print(json.dumps({k: v for k, v in summ.items() if k not in ("cfg", "excluded_outage_week")}, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--name", default="final_backtest")
    ap.add_argument("--config", default=None, help="config json relative to the project root")
    ap.add_argument("--val", default="main", help="main | extra | comma-separated dates, e.g. 2026-10-14,2026-10-21")
    a = ap.parse_args()
    if a.backtest:
        backtest(a.name, a.config, a.val)
