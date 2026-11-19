"""Average saved fold probabilities of several experiments (aligned by UserID)."""
import pickle, sys
import numpy as np, pandas as pd
from common import ART
from modeling import summarize

def load(n): return pickle.load(open(ART / f"folds_{n}.pkl", "rb"))

def avg(names, weights=None):
    Fs = [load(n) for n in names]; w = weights or [1] * len(names)
    out = []
    for parts in zip(*Fs):
        v, P, Y, u = parts[0]; acc = w[0] * P
        for wi, (v2, P2, Y2, u2) in zip(w[1:], parts[1:]):
            assert v == v2
            acc = acc + wi * P2[pd.Series(np.arange(len(u2)), index=u2).loc[u].to_numpy()]
        out.append((v, acc / sum(w), Y, u))
    return out

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        S = summarize(avg(spec.split("+")))
        print(f"{spec:60s} honest {S['honest_score']:.4f} macro {S['honest_macro']:.4f} micro {S['honest_micro']:.4f} oracle {S['oracle_score']:.4f}")
