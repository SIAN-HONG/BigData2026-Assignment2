"""Hybrid sequence model (Phase 2b), inspired by daily-activity-profile LSTMs for developer
disengagement prediction.

Input per (user, cutoff t):
  * sequence: daily counts of the 13 event kinds (6 own activities + 7 relational kinds) over the
    56 days before t, day 0 = [t-24h, t); only rows with avail < t (point-in-time)
  * static : tabular features T0+T1+R1 (sign-log + standardised)
Model: GRU(13 -> 64) over the reversed-time sequence, concat MLP(static) -> 6 logits.
Loss: BCE with the label-quality weights (optional).  Same folds / training window / censoring.
  CUDA_VISIBLE_DEVICES=3 python src/seqmodel.py --name seq_qw --weights quality [--val extra]
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import ART, DAY, KIND_NAMES
from exp import VAL
from exp2 import VAL_EXTRA, summarize_p2, log_row
from features import feature_groups
from modeling import YCOLS, censor_labels, feature_cols, quality_weights, train_window

NK, L = len(KIND_NAMES), 56


class SeqCache:
    """uint8 daily-count tensors per cutoff (computed once, point-in-time)."""

    def __init__(self, ev):
        self.ev = ev.sort_values("avail", kind="stable")
        self.av = self.ev.avail.to_numpy()
        self.cache = {}

    def get(self, t, uids):
        key = (t, len(uids), int(uids[:5].sum()))
        if key in self.cache:
            return self.cache[key]
        ts = int(pd.Timestamp(t).value // 10**9)
        e = self.ev.iloc[:np.searchsorted(self.av, ts)]
        e = e[(e.ts < ts) & (e.ts > ts - L * DAY)]
        pos = pd.Series(np.arange(len(uids)), index=uids)
        e = e[e.uid.isin(pos.index)]
        X = np.zeros((len(uids), L, NK), np.uint16)
        np.add.at(X, (pos.loc[e.uid].to_numpy(), ((ts - e.ts.to_numpy()) // DAY).astype(int), e.kind.to_numpy()), 1)
        X = np.minimum(X, 255).astype(np.uint8)
        self.cache[key] = X
        return X


class Net(nn.Module):
    """arch: gru (1 layer), gru2 (2 layers), cnn (dilated 1-D conv), tfm (transformer encoder)."""

    def __init__(self, d_static, h=64, arch="gru", use_static=True):
        super().__init__()
        self.arch, self.use_static = arch, use_static
        if arch in ("gru", "gru2"):
            self.enc = nn.GRU(NK, h, num_layers=2 if arch == "gru2" else 1, batch_first=True,
                              dropout=0.1 if arch == "gru2" else 0.0)
        elif arch == "cnn":
            self.enc = nn.Sequential(nn.Conv1d(NK, h, 3, padding=1), nn.GELU(),
                                     nn.Conv1d(h, h, 3, padding=2, dilation=2), nn.GELU(),
                                     nn.Conv1d(h, h, 3, padding=4, dilation=4), nn.GELU())
        elif arch == "tfm":
            self.inp = nn.Linear(NK, h)
            self.pos = nn.Parameter(torch.zeros(1, 512, h))
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 2 * h, 0.1, batch_first=True), 2)
        self.st = nn.Sequential(nn.Linear(d_static, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 64))
        d = h + (64 if use_static else 0) + (h if arch == "cnn" else 0)
        self.head = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 6))

    def forward(self, seq, st):
        x = torch.log1p(seq.flip(1))          # oldest day first
        if self.arch in ("gru", "gru2"):
            _, hN = self.enc(x)
            z = hN[-1]
        elif self.arch == "cnn":
            c = self.enc(x.transpose(1, 2))   # (B, h, L)
            z = torch.cat([c[:, :, -7:].mean(2), c.amax(2)], 1)   # last-week summary + global max
        else:
            y = self.enc(self.inp(x) + self.pos[:, :x.shape[1]])
            z = y[:, -1]
        if self.use_static:
            z = torch.cat([z, self.st(st)], 1)
        return self.head(z)


def run_fold(df, cache, v, feats, every=3, weights=None, epochs=4, seed=0, dev="cuda", bs=2048,
             arch="gru", hidden=64, use_static=True, lr=2e-3, te=None, tr=None):
    """te: target rows at cutoff v (default: the dataset rows of v); tr: censored training rows."""
    torch.manual_seed(seed); np.random.seed(seed)
    if tr is None:
        tr = censor_labels(train_window(df, v, every=every), cache.ev, v)
    if te is None:
        te = df[df.cutoff == v]
    W = quality_weights(tr, weights) if weights else None
    Xs = tr[feats].to_numpy(np.float32)
    Xs = np.sign(Xs) * np.log1p(np.abs(Xs))
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    Xs = (Xs - mu) / sd
    seq = np.concatenate([cache.get(t, tr.uid.to_numpy()[(tr.cutoff == t).to_numpy()])
                          for t in tr.cutoff.unique()])
    order = np.concatenate([np.where((tr.cutoff == t).to_numpy())[0] for t in tr.cutoff.unique()])
    Xs, Y = Xs[order], tr[YCOLS].to_numpy(np.float32)[order]
    Wt = (np.column_stack([W[c] for c in [y[2:] for y in YCOLS]]).astype(np.float32)[order]
          if W is not None else np.ones_like(Y))
    net = Net(Xs.shape[1], hidden, arch, use_static).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(Y); steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    seq_t = torch.from_numpy(seq)               # stays on CPU, batches moved to GPU
    for ep in range(epochs):
        net.train()
        for b in np.array_split(np.random.permutation(n), (n + bs - 1) // bs):
            s = seq_t[b].to(dev).float()
            out = net(s, torch.from_numpy(Xs[b]).to(dev))
            loss = (lossf(out, torch.from_numpy(Y[b]).to(dev)) * torch.from_numpy(Wt[b]).to(dev)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    Xt = te[feats].to_numpy(np.float32)
    Xt = (np.sign(Xt) * np.log1p(np.abs(Xt)) - mu) / sd
    S = torch.from_numpy(cache.get(v, te.uid.to_numpy()))
    Xt = torch.from_numpy(Xt.astype(np.float32))
    with torch.no_grad():   # batched inference (GPU is shared)
        P = np.concatenate([torch.sigmoid(net(S[i:i + bs].to(dev).float(), Xt[i:i + bs].to(dev))).cpu().numpy()
                            for i in range(0, len(Xt), bs)])
    return P, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--groups", default="T0,T1,R1")
    ap.add_argument("--weights", default="")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", default="main")
    ap.add_argument("--arch", default="gru")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--window", type=int, default=56)
    ap.add_argument("--no-static", action="store_true")
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    global L
    L = a.window
    df = pd.read_parquet(ART / "dataset.parquet")
    ev = pd.read_parquet(ART / "events.parquet")
    G = feature_groups(feature_cols(df))
    feats = [f for g in a.groups.split(",") for f in G[g]]
    cache = SeqCache(ev)
    t0 = time.time(); folds = []
    for v in {"extra": VAL_EXTRA, "all": VAL_EXTRA + list(VAL)}.get(a.val, VAL):
        P, te = run_fold(df, cache, v, feats, weights=a.weights or None, epochs=a.epochs, seed=a.seed,
                         arch=a.arch, hidden=a.hidden, use_static=not a.no_static, lr=a.lr)
        folds.append((v, P, te[YCOLS].to_numpy(), te.UserID.to_numpy()))
        print(f"  {v.date()} done {time.time() - t0:.0f}s", flush=True)
    pickle.dump(folds, open(ART / f"folds_p2_{a.name}.pkl", "wb"))
    row = summarize_p2(folds, a.name)
    row.update({"groups": a.groups + f" [{a.arch} h{a.hidden} {a.window}d x 13 kinds{' no-static' if a.no_static else ''}]", "weights": a.weights, "objective": "seq-hybrid",
                "seeds": str(a.seed), "params": f'{{"epochs": {a.epochs}}}', "label_censoring": True,
                "val_weeks": ",".join(str(f[0].date()) for f in folds), "minutes": round((time.time() - t0) / 60, 1)})
    log_row(row)


if __name__ == "__main__":
    main()
