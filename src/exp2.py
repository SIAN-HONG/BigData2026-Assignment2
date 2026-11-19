"""Phase 2 experiment runner (same rolling-origin protocol as exp.py / pipeline.py).

Differences to exp.py: uses artifacts/dataset_p2.parquet (+ regime-normalised group N), reports
the final decision rule (joint plug-in) as the main score, logs per-week (including the 09-02
outage week) and per-activity F1 to results/phase2/experiments_p2.tsv, and supports
  --weights quality   per-activity sample weights that down-weight training cutoffs whose
                      label week has an abnormally low positive rate (collection drought)
  --objective rank    XGBoost pairwise ranking within each cutoff; its ranking is combined with
                      the probability levels of the binary model ("transplant")
Fold probabilities: artifacts/folds_p2_<name>.pkl

  CUDA_VISIBLE_DEVICES=3 python src/exp2.py --name p2_base --groups T0,T1,R1,R2
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from common import ART, RES, TARGETS
from exp import VAL
from features import add_regime_normalized, feature_groups
from modeling import (EXCLUDE_WEEKS, XGB_PARAMS, YCOLS, censor_labels, feature_cols, quality_weights, joint_plugin_predict, official_score,
                      plugin_predict, train_window, tune_thresholds)

OUT = RES / "phase2"
OUT.mkdir(exist_ok=True)
_DF = None
EV = pd.read_parquet(ART / "events.parquet")


def load():
    global _DF
    if _DF is None:
        _DF = add_regime_normalized(pd.read_parquet(ART / "dataset_p2.parquet"))
        if (ART / "outage_O.parquet").exists():   # group O (build_outage.py), merged on (cutoff, uid)
            _DF = _DF.merge(pd.read_parquet(ART / "outage_O.parquet"), on=["cutoff", "uid"], how="left",
                            validate="one_to_one")
        if (ART / "group_R4.parquet").exists():   # group R4 (build_outage.py --group R4)
            _DF = _DF.merge(pd.read_parquet(ART / "group_R4.parquet"), on=["cutoff", "uid"], how="left",
                            validate="one_to_one")
    return _DF


CAT_PARAMS = dict(task_type="GPU", devices="0", iterations=1000, learning_rate=0.08, depth=8,
                  l2_leaf_reg=3, border_count=128, verbose=0, allow_writing_files=False)


def fit_predict_cat(tr, te, feats, params, seed, weights=None):
    from catboost import CatBoostClassifier
    P = np.zeros((len(te), 6))
    for j, c in enumerate(TARGETS):
        m = CatBoostClassifier(**{**CAT_PARAMS, **params, "random_seed": seed})
        m.fit(tr[feats].to_numpy(np.float32), tr[f"y_{c}"].to_numpy(),
              sample_weight=None if weights is None else weights[c])
        P[:, j] = m.predict_proba(te[feats].to_numpy(np.float32))[:, 1]
    return P


def fit_predict_multi(tr, te, feats, params, seed, weights=None):
    """Multi-task: one multi-output tree ensemble for all 6 activities (shared splits).
    Sample weight = mean of the per-activity weights."""
    p = {**XGB_PARAMS, **params, "random_state": seed, "multi_strategy": "multi_output_tree"}
    w = None if weights is None else np.mean([weights[c] for c in TARGETS], axis=0)
    m = xgb.XGBClassifier(**p)
    m.fit(tr[feats], tr[YCOLS].to_numpy(), sample_weight=w)
    return m.predict_proba(te[feats])


def fit_predict_stack(tr, te, feats, params, seed, weights=None, k=3):
    """Classifier chain / stacking across activities.
    Stage 1: per-activity models give out-of-fold probabilities on the training rows; folds are
    contiguous blocks of training cutoffs (limits leakage between overlapping label weeks) and the
    target rows get stage-1 probabilities from models fit on all training rows.
    Stage 2: per-activity models on features + the 6 stage-1 probabilities (e.g. pr_review_comment
    sees the predicted chance of pr_review)."""
    cuts = np.sort(tr.cutoff.unique())
    blocks = np.array_split(cuts, k)
    oof = np.zeros((len(tr), 6))
    for b in blocks:
        m = tr.cutoff.isin(b).to_numpy()
        sub_w = None if weights is None else {c: w[~m] for c, w in weights.items()}
        oof[m] = fit_predict_xgb(tr[~m], tr[m], feats, params, seed, sub_w, "binary")
    p1 = fit_predict_xgb(tr, te, feats, params, seed, weights, "binary")
    sc = [f"stack_p_{c}" for c in TARGETS]
    tr2 = pd.concat([tr.reset_index(drop=True), pd.DataFrame(oof, columns=sc)], axis=1)
    te2 = pd.concat([te.reset_index(drop=True), pd.DataFrame(p1, columns=sc)], axis=1)
    return fit_predict_xgb(tr2, te2, feats + sc, params, seed, weights, "binary")


def fit_predict_xgb(tr, te, feats, params, seed, weights=None, objective="binary"):
    if objective == "stack":
        return fit_predict_stack(tr, te, feats, params, seed, weights)
    if objective == "catboost":
        return fit_predict_cat(tr, te, feats, params, seed, weights)
    if objective == "multi":
        return fit_predict_multi(tr, te, feats, params, seed, weights)
    p = {**XGB_PARAMS, **params, "random_state": seed}
    es = p.pop("early_stop", None)
    if es:   # inner validation = training cutoffs of the last 7 days; inner train ends 8 days before it
        iv = (tr.cutoff > tr.cutoff.max() - pd.Timedelta(days=7)).to_numpy()
        it = (tr.cutoff <= tr.cutoff[iv].min() - pd.Timedelta(days=8)).to_numpy()
    P = np.zeros((len(te), 6))
    for j, c in enumerate(TARGETS):
        w = None if weights is None else weights[c]
        if es:   # number of trees from early stopping (logloss), then refit on all training rows
            m = xgb.XGBClassifier(**p, early_stopping_rounds=es, eval_metric="logloss")
            m.fit(tr.loc[it, feats], tr.loc[it, f"y_{c}"], sample_weight=None if w is None else w[it],
                  eval_set=[(tr.loc[iv, feats], tr.loc[iv, f"y_{c}"])],
                  sample_weight_eval_set=None if w is None else [w[iv]], verbose=False)
            BEST_IT.setdefault(c, []).append(m.best_iteration + 1)
            m = xgb.XGBClassifier(**{**p, "n_estimators": m.best_iteration + 1})
        else:
            m = xgb.XGBClassifier(**p)
        m.fit(tr[feats], tr[f"y_{c}"], sample_weight=w)
        pb = m.predict_proba(te[feats])[:, 1]
        if objective == "rank":
            trs = tr.sort_values("cutoff", kind="stable")
            r = xgb.XGBRanker(**{**p, "objective": "rank:pairwise"})
            r.fit(trs[feats], trs[f"y_{c}"], qid=trs.cutoff.astype("int64").to_numpy())
            sc = r.predict(te[feats])
            # transplant: keep the binary model's probability levels, use the ranker's order
            pb = np.sort(pb)[np.argsort(np.argsort(sc))]
        P[:, j] = pb
    return P


BEST_IT = {}   # early stopping: best number of trees per activity, over folds / seeds

VAL_EXTRA = [pd.Timestamp(f"{d} 12:00") for d in
             ["2026-05-06", "2026-05-27", "2026-06-17", "2026-07-08", "2026-07-29", "2026-08-19"]]


def run(name, groups, params=None, seeds=(0,), every=3, weights=None, objective="binary", note="", val=None,
        o_replace=False, drop=()):
    df = load()
    G = feature_groups(feature_cols(df))
    feats = [f for g in groups for f in G[g]]
    if drop:   # drop features by name prefix, e.g. ("glob7_", "tenure_")
        feats = [f for f in feats if not f.startswith(tuple(drop))]
    if o_replace:   # each raw column with an outage-corrected twin o_<name> is replaced by it
        oset = set(G["O"])
        feats = [f for f in feats if f"o_{f}" not in oset] + G["O"]
    t0 = time.time()
    folds = []
    for v in (val or VAL):
        tr = censor_labels(train_window(df, v, every=every), EV, v)
        te = df[df.cutoff == v]
        W = quality_weights(tr, weights) if weights and weights.startswith("quality") else None
        P = np.mean([fit_predict_xgb(tr, te, feats, params or {}, sd, W, objective) for sd in seeds], axis=0)
        folds.append((v, P, te[YCOLS].to_numpy(), te.UserID.to_numpy()))
        print(f"  {v.date()} done ({time.time() - t0:.0f}s)", flush=True)
    pickle.dump(folds, open(ART / f"folds_p2_{name}.pkl", "wb"))
    row = summarize_p2(folds, name)
    row.update({"groups": "+".join(groups), "n_feats": len(feats), "params": json.dumps(params or {}),
                "seeds": ",".join(map(str, seeds)), "every": every, "weights": weights or "", "objective": objective, "o_replace": o_replace, "drop": ",".join(drop),
                "val_weeks": ",".join(str(v.date()) for v in (val or VAL)),
                "train_period": "cutoffs 2026-01-07 .. V-8d", "label_censoring": True, "minutes": round((time.time() - t0) / 60, 1),
                "best_iter": json.dumps({c: int(np.median(v)) for c, v in BEST_IT.items()}) if BEST_IT else "",
                "note": note})
    log_row(row)
    return row, folds


def summarize_p2(folds, name):
    per_week, per_t = {}, []
    rows = []
    for v, P, Y, _ in folds:
        wk = str(v.date())
        sj = official_score(joint_plugin_predict(P), Y)
        sp = official_score(plugin_predict(P), Y)
        so = official_score((P >= tune_thresholds(P, Y)).astype(int), Y)
        rows.append({"week": wk, **sj, "plugin_score": sp["score"], "oracle_score": so["score"]})
    R = pd.DataFrame(rows)
    E = R[~R.week.isin(EXCLUDE_WEEKS)]
    if not len(E):
        E = R
    out = {"name": name, "score": E.score.mean(), "macro_f1": E.macro_f1.mean(), "micro_f1": E.micro_f1.mean(),
           "score_6wk": R.score.mean(), "week_sd": E.score.std(), "plugin_score": E.plugin_score.mean(),
           "oracle_score": E.oracle_score.mean()}
    for c in TARGETS:
        out[f"f1_{c}"] = E[f"f1_{c}"].mean()
    out["per_week"] = json.dumps({r.week: round(r.score, 4) for r in R.itertuples()})
    mainw = R[R.week >= "2026-09-09"]; extraw = R[R.week < "2026-09-01"]
    if len(mainw) and len(extraw):
        out["score_main5"] = mainw.score.mean()
        out["score_extra6"] = extraw.score.mean()
        out["score_11wk"] = pd.concat([mainw, extraw]).score.mean()
    R.to_csv(OUT / f"weeks_{name}.csv", index=False)
    return out


def log_row(row):
    import fcntl
    p = OUT / "experiments_p2.tsv"
    r = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}
    with open(OUT / ".ledger.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        old = pd.read_csv(p, sep="\t") if p.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame([r])], ignore_index=True).to_csv(p, sep="\t", index=False)
    print(json.dumps({k: r[k] for k in ("name", "score", "macro_f1", "micro_f1", "score_6wk", "plugin_score",
                                        "oracle_score", "per_week") if k in r}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--groups", default="T0,T1,R1,R2")
    ap.add_argument("--params", default='{"max_depth": 10, "min_child_weight": 50}')
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--objective", default="binary")
    ap.add_argument("--note", default="")
    ap.add_argument("--drop", default="", help="comma-separated feature-name prefixes to leave out")
    ap.add_argument("--o-replace", action="store_true", help="replace raw temporal columns by their group-O twins")
    ap.add_argument("--val", default="main", help="main (6 Wednesdays 09-02..10-07) or extra (6 earlier Wednesdays)")
    a = ap.parse_args()
    run(a.name, a.groups.split(","), json.loads(a.params), tuple(int(x) for x in a.seeds.split(",")),
        a.every, a.weights, a.objective, a.note,
        {"extra": VAL_EXTRA, "all": VAL_EXTRA + list(VAL)}.get(a.val), a.o_replace,
        tuple(x for x in a.drop.split(",") if x))
