"""Compare decision-threshold strategies on saved rolling-validation predictions."""
import pickle, sys
import numpy as np, pandas as pd
from modeling import official_score, tune_thresholds

def plugin_threshold(p):
    """Label-free: choose k maximising expected F1 assuming p is calibrated:
    E[F1](top-k) ~ 2*sum(p_topk) / (k + sum(p))."""
    ps = np.sort(p)[::-1]; k = np.arange(1, len(ps) + 1)
    ef = 2 * np.cumsum(ps) / (k + ps.sum()); i = int(np.argmax(ef))
    return (ps[i] + ps[min(i + 1, len(ps) - 1)]) / 2

def evaluate(folds, verbose=True):
    res = {}
    prev, hist, prev_rate = None, [], None
    rows = []
    for v, P, Y, _ in folds:
        r = {"val": str(v.date())}
        or_th = tune_thresholds(P, Y)
        r["oracle"] = official_score((P >= or_th).astype(int), Y)["score"]
        r["plugin"] = official_score(np.column_stack([P[:, j] >= plugin_threshold(P[:, j]) for j in range(6)]).astype(int), Y)["score"]
        r["fixed0.3"] = official_score((P >= 0.3).astype(int), Y)["score"]
        if prev is not None:
            r["prev_week"] = official_score((P >= prev).astype(int), Y)["score"]
            r["mean_prev"] = official_score((P >= np.mean(hist, 0)).astype(int), Y)["score"]
            # match previous week's predicted-positive rate per target (rank based)
            pr = np.column_stack([P[:, j] >= np.quantile(P[:, j], 1 - prev_rate[j]) for j in range(6)]).astype(int)
            r["prev_rate"] = official_score(pr, Y)["score"]
        rows.append(r)
        prev = or_th; hist.append(or_th); prev_rate = ((P >= or_th).mean(0))
    R = pd.DataFrame(rows)
    if verbose: print(R.round(4).to_string()); print(R.iloc[1:].mean(numeric_only=True).round(4).to_dict())
    return R

if __name__ == "__main__":
    for f in sys.argv[1:]:
        print("==", f); evaluate(pickle.load(open(f, "rb")))
