"""Pre-declared blends of saved fold probabilities, evaluated on 5 main + 6 extra weeks (+ 09-02).
Members: xgbW5 (weighted XGB x5), seq (GRU hybrid), nnU (unweighted GNN+MLP x2 seeds), v1 ensemble."""
import pickle, sys, json
import numpy as np, pandas as pd
from common import ART, RES
from modeling import joint_plugin_predict, official_score

def L(n): return pickle.load(open(ART / n, "rb"))
mb = L("members_p2b_extra.pkl") + L("members_p2b_backtest.pkl")
mc = L("members_p2c_extra.pkl") + L("members_p2c_backtest.pkl")
M = {"xgbW5": {v: np.mean(m[:5], 0) for v, m, Y, U in mb},
     "nnU": {v: np.mean(m, 0) for v, m, Y, U in mc},
     "v1": {v: P for v, P, Y, U in L("folds_v1_extra.pkl") + L("folds_final_backtest_v1_baseline.pkl")}}
for extra in sys.argv[1:]:                       # extra members: name=fileA,fileB
    n, fs = extra.split("="); M[n] = {v: P for f in fs.split(",") for v, P, Y, U in L(f)}
Y = {v: y for v, m, y, U in mb}
WEEKS = sorted(Y)

def score(blend):
    s = {}
    for v in WEEKS:
        P = sum(w * M[k][v] for k, w in blend.items()) / sum(blend.values())
        s[str(v.date())] = official_score(joint_plugin_predict(P), Y[v])["score"]
    return pd.Series(s)

if __name__ == "__main__":
    ref = score({"v1": 1})
    blends = {"v1": {"v1": 1}, "xgbW5": {"xgbW5": 1}}
    if "seq" in M:
        blends.update({"seq": {"seq": 1}, "xgbW5+seq 0.5/0.5": {"xgbW5": .5, "seq": .5},
                       "xgbW5+seq 0.7/0.3": {"xgbW5": .7, "seq": .3},
                       "xgbW5+seq+nnU 0.5/0.3/0.2": {"xgbW5": .5, "seq": .3, "nnU": .2},
                       "v1+seq 0.7/0.3": {"v1": .7, "seq": .3}})
    if "cnn" in M:   # Phase 2c, pre-declared before evaluation
        blends = {"v1": {"v1": 1}, "cnn": {"cnn": 1}, "tfm": {"tfm": 1},
                  "xgbW5+cnn 0.5/0.5": {"xgbW5": .5, "cnn": .5},
                  "v1+cnn 0.7/0.3": {"v1": .7, "cnn": .3},
                  "v1+cnn+tfm 0.6/0.2/0.2": {"v1": .6, "cnn": .2, "tfm": .2},
                  "xgbW5+cnn+tfm+gru 0.4/0.2/0.2/0.2": {"xgbW5": .4, "cnn": .2, "tfm": .2, "seq": .2},
                  "xgbW5+R11+cnn+tfm+gru equal": {"xgbW5": 1, "R11": 1, "cnn": 1, "tfm": 1, "seq": 1}}
    rows = []
    for n, b in blends.items():
        s = score(b); d = (s - ref) * 100
        main = [w for w in s.index if w >= "2026-09-09"]; extra = [w for w in s.index if w < "2026-09-01"]
        d11 = d[main + extra]
        rows.append({"blend": n, "main5": s[main].mean(), "extra6": s[extra].mean(), "w0902": s["2026-09-02"],
                     "all11": s[main + extra].mean(), "all12": s.mean(),
                     "diff11_vs_v1": d11.mean(), "t11": d11.mean() / (d11.std() / np.sqrt(11) + 1e-12),
                     "wins11": int((d11 > 0).sum()), "per_week": json.dumps(s.round(4).to_dict())})
    R = pd.DataFrame(rows); R.to_csv(RES / "phase2" / ("blends_2c.csv" if "cnn" in M else "blends.csv"), index=False)
    print(R.drop(columns="per_week").round(4).to_string(index=False))
