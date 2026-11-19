"""Model training, rolling-origin validation and threshold selection.

Validation protocol (time-based, no random split):
  * validation cutoffs V are Wednesdays 12:00 UTC (same weekday/hour as the official cutoff
    2026-11-11 12:00 UTC);
  * a model validated at V is trained only on cutoffs t with t + 4h + 7d <= V, i.e. every
    training label window is complete before the validation features are computed
    (t <= V - 8 days).  At test time the same rule gives training cutoffs <= 11-03;
  * decision thresholds applied to week V are tuned on the previous validation week
    (V - 7d), never on V itself ("honest" thresholds).  Oracle thresholds are reported
    separately only as an upper bound.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd

from common import DAY, ROOT, TARGETS

sys.path.insert(0, str(ROOT / "assignment2" / "scripts"))
YCOLS = [f"y_{c}" for c in TARGETS]
META = ["uid", "cutoff", "UserID"] + YCOLS

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=200,
                  feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                  n_estimators=400, verbose=-1, n_jobs=32)
# GPU gradient boosting (same role as LGB_PARAMS; much faster on the shared server)
XGB_PARAMS = dict(objective="binary:logistic", tree_method="hist", device="cuda", learning_rate=0.05,
                  max_depth=8, min_child_weight=20, subsample=0.7, colsample_bytree=0.5,
                  reg_lambda=1.0, n_estimators=400, max_bin=128, n_jobs=8)
BACKEND = "xgb"


def official_score(P: np.ndarray, Y: np.ndarray) -> dict:
    """Same formulas as scripts/evaluate.py (tp/fp/fn summed over users)."""
    tp = ((P == 1) & (Y == 1)).sum(0)
    fp = ((P == 1) & (Y == 0)).sum(0)
    fn = ((P == 0) & (Y == 1)).sum(0)
    f1 = np.where(tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 1.0)
    macro = f1.mean()
    micro = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())
    return {"macro_f1": float(macro), "micro_f1": float(micro), "score": float((macro + micro) / 2),
            **{f"f1_{c}": float(f) for c, f in zip(TARGETS, f1)}}


def best_f1_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximising F1 for one target (exact search over sorted scores)."""
    o = np.argsort(-p)
    ps, ys = p[o], y[o]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    f1 = 2 * tp / (k + ys.sum())
    i = int(np.argmax(f1))
    return float((ps[i] + ps[min(i + 1, len(ps) - 1)]) / 2)


def tune_thresholds(Pr: np.ndarray, Y: np.ndarray, joint: bool = True) -> np.ndarray:
    """Per-target F1-optimal thresholds, then (optionally) coordinate ascent on the
    official score (macro + micro) / 2, since micro-F1 couples the six targets."""
    th = np.array([best_f1_threshold(Pr[:, j], Y[:, j]) for j in range(Y.shape[1])])
    if not joint:
        return th
    grid = np.linspace(0.02, 0.95, 94)
    best = official_score((Pr >= th).astype(int), Y)["score"]
    for _ in range(3):
        for j in range(Y.shape[1]):
            for g in grid:
                th2 = th.copy(); th2[j] = g
                sc = official_score((Pr >= th2).astype(int), Y)["score"]
                if sc > best + 1e-9:
                    best, th = sc, th2
    return th


def quality_weights(tr: pd.DataFrame, spec: str = "quality") -> dict:
    """Label-quality sample weights per activity: r_k(t) = positive rate of the label week of
    training cutoff t / median over training cutoffs;  w = clip(r ** pow, upper=cap)
    ('quality:pow=..,cap=..'), or 1[r >= thr] with 'quality:hard=thr'.  Down-weights label weeks
    with abnormally low recorded activity (collection drought).  Training labels only."""
    opt = dict(kv.split("=") for kv in spec.split(":")[1].split(",")) if ":" in spec else {}
    prev = tr.groupby("cutoff")[YCOLS].mean()
    r = prev / prev.median()
    if "hard" in opt:
        w = (r >= float(opt["hard"])).astype(float)
    else:
        w = (r ** float(opt.get("pow", 1))).clip(upper=float(opt.get("cap", 1)))
    return {c: tr.cutoff.map(w[f"y_{c}"]).to_numpy() for c in TARGETS}


def outage_label_mask(tr: pd.DataFrame, ev: pd.DataFrame, cut) -> dict:
    """1 / 0 per training row and activity: 0 if the row's label window [t + 4h, t + 4h + 7d)
    overlaps a collection outage of that activity (features.outage_intervals, detected from the
    rows known before `cut`).  Such labels say "no activity" only because nothing was recorded."""
    from common import GAP_SEC
    from features import outage_intervals
    c_s = int(pd.Timestamp(cut).value // 10**9)
    past = ev[(ev.avail.to_numpy() < c_s) & (ev.ts.to_numpy() < c_s)]
    back = (pd.Timestamp(cut) - tr.cutoff.min()).days + 10
    iv = outage_intervals(past, c_s, back_days=back)
    cuts = pd.Series(tr.cutoff.unique())
    a = ((cuts - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).to_numpy() + GAP_SEC   # unit-independent
    out = {}
    for k, c in enumerate(TARGETS):
        hit = np.zeros(len(cuts), bool)
        for s0, e0 in iv[k]:
            hit |= (a < e0) & (a + 7 * DAY > s0)
        out[c] = tr.cutoff.map(dict(zip(cuts, (~hit).astype(float)))).to_numpy()
    return out


def train_models(tr: pd.DataFrame, feats: list[str], params=None, targets=TARGETS, seed=0, weights=None):
    params = dict(params or {})
    backend = params.pop("backend", BACKEND)
    base = XGB_PARAMS if backend == "xgb" else LGB_PARAMS
    params = {**base, **params, "random_state": seed}
    # optional recency weighting: weight = 0.5 ** (age of the cutoff / half-life)
    hl = params.pop("halflife_days", None)
    w = None
    if hl:
        age = (tr.cutoff.max() - tr.cutoff).dt.days.to_numpy()
        w = np.power(0.5, age / hl)
    es = params.pop("early_stop", None)
    if es:   # inner validation = training cutoffs of the last 7 days; inner train ends 8 days before it
        iv = (tr.cutoff > tr.cutoff.max() - pd.Timedelta(days=7)).to_numpy()
        it = (tr.cutoff <= tr.cutoff[iv].min() - pd.Timedelta(days=8)).to_numpy()
    models = {}
    for c in targets:
        sw = w if weights is None else weights[c]
        p = params
        if es and backend == "xgb":   # number of trees from early stopping (logloss), then refit on all rows
            m = xgb.XGBClassifier(**params, early_stopping_rounds=es, eval_metric="logloss")
            m.fit(tr.loc[it, feats], tr.loc[it, f"y_{c}"], sample_weight=None if sw is None else sw[it],
                  eval_set=[(tr.loc[iv, feats], tr.loc[iv, f"y_{c}"])],
                  sample_weight_eval_set=None if sw is None else [sw[iv]], verbose=False)
            p = {**params, "n_estimators": m.best_iteration + 1}
        m = xgb.XGBClassifier(**p) if backend == "xgb" else lgb.LGBMClassifier(**p)
        m.fit(tr[feats], tr[f"y_{c}"], sample_weight=sw)
        models[c] = m
    return models


def predict(models, df, feats) -> np.ndarray:
    return np.column_stack([models[c].predict_proba(df[feats])[:, 1] for c in TARGETS])


def train_window(df: pd.DataFrame, val_cut: pd.Timestamp, start=None, every: int = 1,
                 wednesday_only=False) -> pd.DataFrame:
    """All training samples whose label window ends before val_cut."""
    last = val_cut - pd.Timedelta(days=8)
    m = df.cutoff <= last
    if start is not None:
        m &= df.cutoff >= pd.Timestamp(start)
    if wednesday_only:
        m &= df.cutoff.dt.dayofweek == 2
    elif every > 1:
        d = (last - df.cutoff).dt.days
        m &= (d % every) == 0
    return df[m]


_UID_EV = {}


def censor_labels(tr: pd.DataFrame, ev: pd.DataFrame, cut) -> pd.DataFrame:
    """Recompute the labels of training cutoffs whose late-arrival slack (5 days after the
    label window) extends beyond `cut`, using only rows known before `cut`.  Without this,
    rows that arrived after the validation / prediction cutoff would leak into training labels.
    The dataset stores labels computed with the full release; only cutoffs t with
    t + 4h + 7d + 5d > cut are affected."""
    from common import GAP_SEC, label_matrix
    cut_s = int(pd.Timestamp(cut).value // 10**9)
    tr = tr.copy()
    for t in tr.cutoff.unique():
        ts = int(pd.Timestamp(t).value // 10**9)
        start, end = ts + GAP_SEC, ts + GAP_SEC + 7 * DAY
        if end + 5 * DAY <= cut_s:
            continue
        m = (tr.cutoff == t).to_numpy()
        Y = label_matrix(ev, tr.uid.to_numpy()[m], start, end, slack=max(0, cut_s - end))
        tr.loc[m, YCOLS] = Y
    return tr


def rolling_eval(df, feats, val_cuts, params=None, start=None, every=1, wednesday_only=False,
                 seed=0, log=print, return_preds=False, censor_ev=None):
    """Train one model per validation week, collect probabilities + labels.
    censor_ev: the event log; when given, training labels are censored at each validation cutoff
    (censor_labels), as in pipeline.fit_predict.  Without it, late-arriving rows that were known
    only after the validation cutoff can enter the labels of the last training cutoffs."""
    out = []
    for v in val_cuts:
        v = pd.Timestamp(v)
        tr = train_window(df, v, start, every, wednesday_only)
        if censor_ev is not None:
            tr = censor_labels(tr, censor_ev, v)
        va = df[df.cutoff == v]
        models = train_models(tr, feats, params, seed=seed)
        P = predict(models, va, feats)
        out.append((v, P, va[YCOLS].to_numpy(), va.UserID.to_numpy()))
        log(f"    val {v.date()} train_rows={len(tr)} val_rows={len(va)}")
    return out


def plugin_threshold(p: np.ndarray) -> float:
    """Label-free threshold: pick the top-k that maximises the expected F1 under the
    model's own probabilities, E[F1](top-k) ~= 2 * sum(p_topk) / (k + sum(p)).
    Needs no labels of the target week, so it adapts to the week's predicted base rate."""
    ps = np.sort(p)[::-1]
    k = np.arange(1, len(ps) + 1)
    ef = 2 * np.cumsum(ps) / (k + ps.sum())
    i = int(np.argmax(ef))
    return float((ps[i] + ps[min(i + 1, len(ps) - 1)]) / 2)


def plugin_predict(P: np.ndarray) -> np.ndarray:
    return np.column_stack([P[:, j] >= plugin_threshold(P[:, j]) for j in range(P.shape[1])]).astype(int)


def joint_plugin_predict(P: np.ndarray, iters: int = 3) -> np.ndarray:
    """Label-free decision rule for the official metric: choose the number of predicted
    positives k_j of each target (top-k_j by probability) to maximise the EXPECTED
    (macro-F1 + micro-F1) / 2 under the model's own probabilities, where for top-k
    E[tp] = sum of the k largest p, E[fp] = k - E[tp], E[fn] = sum(p) - E[tp].
    Initialised with the per-target plug-in, then coordinate ascent (micro-F1 couples
    the targets)."""
    n, m = P.shape
    C = [np.concatenate([[0.0], np.cumsum(np.sort(P[:, j])[::-1])]) for j in range(m)]
    tot = [c[-1] for c in C]
    k_all = np.arange(1, n + 1)
    ks = [int(np.argmax(2 * C[j][1:] / (k_all + tot[j]))) + 1 for j in range(m)]

    def exp_score(ks):
        tp = np.array([C[j][ks[j]] for j in range(m)])
        fp = np.array(ks) - tp
        fn = np.array(tot) - tp
        f1 = 2 * tp / (2 * tp + fp + fn)
        return (f1.mean() + 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())) / 2

    grid = np.unique(np.geomspace(1, n, 400).astype(int))
    for _ in range(iters):
        for j in range(m):
            best_s, best_k = exp_score(ks), ks[j]
            for k in grid:
                ks2 = list(ks); ks2[j] = int(k)
                sc = exp_score(ks2)
                if sc > best_s:
                    best_s, best_k = sc, int(k)
            ks[j] = best_k
    out = np.zeros(P.shape, dtype=int)
    for j in range(m):
        out[np.argsort(-P[:, j], kind="stable")[:ks[j]], j] = 1
    return out


# 2026-09-02 week: collection outage (0 pushes recorded 09-02..09-04) -> reported, not averaged
EXCLUDE_WEEKS = {"2026-09-02"}


def summarize(folds, joint=True) -> dict:
    """Per validation week:
      score (honest)  : label-free plug-in thresholds (main metric)
      prev_*          : thresholds tuned on the previous validation week
      oracle_*        : thresholds tuned on the same week (upper bound, not honest)"""
    rows, prev_th = [], None
    for v, P, Y, _ in folds:
        th_or = tune_thresholds(P, Y, joint)
        r = {"val": str(v.date())}
        r.update(official_score(plugin_predict(P), Y))
        r["oracle_score"] = official_score((P >= th_or).astype(int), Y)["score"]
        r["prev_score"] = official_score((P >= prev_th).astype(int), Y)["score"] if prev_th is not None else np.nan
        rows.append(r)
        prev_th = th_or
    R = pd.DataFrame(rows)
    E = R[~R.val.isin(EXCLUDE_WEEKS)]
    return {"table": R, "honest_score": E.score.mean(), "honest_macro": E.macro_f1.mean(),
            "honest_micro": E.micro_f1.mean(), "oracle_score": E.oracle_score.mean(),
            "prev_score": E.prev_score.mean()}


def feature_cols(df) -> list[str]:
    return [c for c in df.columns if c not in META]
