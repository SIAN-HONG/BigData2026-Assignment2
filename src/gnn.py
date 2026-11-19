"""Relational-deep-learning experiment: temporal heterogeneous GNN on user <-> repository
snapshots, evaluated with the same rolling-origin folds and label-free thresholds as the
gradient-boosting models.

Graph at cutoff t (point-in-time, rows with avail < t only):
  * user nodes  : the sample users of that cutoff; input = their tabular features
                  (log1p + standardisation of the T0/T1/R1 groups)
  * repo nodes  : every repository those users touched in [t-28d, t); input = repo stats
  * edges       : user -- repo, edge attribute = log1p(count) of each of 7 interaction
                  types (6 activities + merged_by) in [t-28d, t)  -> typed / weighted edges
Model (2 rounds, user -> repo -> user):
  h_u  = MLP(x_u)
  m_r  = mean_{(u,r)} MLP([h_u, e_ur])        ;  h_r = MLP([m_r, x_r])
  m_u  = mean_{(u,r)} MLP([h_r, e_ur])        ;  z_u = MLP([h_u, m_u])  -> 6 logits
The "mlp" variant drops the message passing (z_u = MLP(h_u)) and is otherwise identical.

  CUDA_VISIBLE_DEVICES=3 python src/gnn.py --variant gnn --name gnn
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import ART, DAY, RES, TARGETS
from exp import VAL
from features import feature_groups
from modeling import YCOLS, feature_cols, summarize

EDGE_KINDS = list(range(7))


def mlp(i, o, h=128, p=0.1, layers=1):
    """`layers` hidden layers of width h (Linear, LayerNorm, GELU, Dropout), then a linear output."""
    mods = []
    for k in range(layers):
        mods += [nn.Linear(i if k == 0 else h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p)]
    return nn.Sequential(*mods, nn.Linear(h, o))


class RelNet(nn.Module):
    def __init__(self, d_u, d_r, d_e, h=128, gnn=True, layers=1):
        super().__init__()
        self.gnn = gnn
        self.enc_u = mlp(d_u, h, h, layers=layers)
        if gnn:
            self.msg_ur = mlp(h + d_e, h, h, layers=layers)
            self.upd_r = mlp(h + d_r, h, h, layers=layers)
            self.msg_ru = mlp(h + d_e, h, h, layers=layers)
            self.head = mlp(2 * h, 6, h, layers=layers)
        else:
            self.head = mlp(h, 6, h, layers=layers)

    @staticmethod
    def _mean(src, idx, n):
        out = torch.zeros(n, src.shape[1], device=src.device).index_add_(0, idx, src)
        cnt = torch.zeros(n, device=src.device).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
        return out / cnt.clamp(min=1).unsqueeze(1)

    def forward(self, g):
        hu = self.enc_u(g["xu"])
        if not self.gnn:
            return self.head(hu)
        eu, er, ea = g["eu"], g["er"], g["ea"]
        mr = self._mean(self.msg_ur(torch.cat([hu[eu], ea], 1)), er, g["xr"].shape[0])
        hr = self.upd_r(torch.cat([mr, g["xr"]], 1))
        mu = self._mean(self.msg_ru(torch.cat([hr[er], ea], 1)), eu, hu.shape[0])
        return self.head(torch.cat([hu, mu], 1))


def build_graphs(df, feats, ev, cuts, rows=None):
    """One graph per cutoff (numpy, moved to GPU later).
    rows: optional {cutoff: DataFrame} overriding df (e.g. the unlabeled final cutoff)."""
    av = ev.avail.to_numpy()
    graphs = {}
    for t in cuts:
        ts = int(pd.Timestamp(t).value // 10**9)
        d = rows[t] if rows is not None and t in rows else df[df.cutoff == t]
        uids = d.uid.to_numpy()
        n = np.searchsorted(av, ts)
        e = ev.iloc[:n]
        e = e[(e.ts >= ts - 28 * DAY) & (e.ts < ts) & (e.kind.isin(EDGE_KINDS)) & (e.repo >= 0)]
        pos = pd.Series(np.arange(len(uids)), index=uids)
        e = e[e.uid.isin(uids)]
        c = e.groupby(["uid", "repo", "kind"]).size().unstack(fill_value=0).reindex(columns=EDGE_KINDS, fill_value=0)
        c = c.reset_index()
        repos, ridx = np.unique(c.repo.to_numpy(), return_inverse=True)
        ea = np.log1p(c[EDGE_KINDS].to_numpy(np.float32))
        # repo inputs: activity of the sample users in the repo + number of such users
        r_act = np.zeros((len(repos), len(EDGE_KINDS)), np.float32)
        np.add.at(r_act, ridx, c[EDGE_KINDS].to_numpy(np.float32))
        r_deg = np.bincount(ridx, minlength=len(repos)).astype(np.float32)[:, None]
        xr = np.log1p(np.concatenate([r_act, r_deg], 1))
        y = d[YCOLS].to_numpy(np.float32) if YCOLS[0] in d else np.zeros((len(d), 6), np.float32)
        graphs[t] = dict(xu=d[feats].to_numpy(np.float32), y=y,
                         eu=pos.loc[c.uid.to_numpy()].to_numpy(), er=ridx, ea=ea, xr=xr,
                         UserID=d.UserID.to_numpy())
    return graphs


def to_dev(g, mu, sd, dev):
    x = (np.sign(g["xu"]) * np.log1p(np.abs(g["xu"])) - mu) / sd
    return dict(xu=torch.tensor(x, device=dev), y=torch.tensor(g["y"], device=dev),
                eu=torch.tensor(g["eu"], device=dev, dtype=torch.long),
                er=torch.tensor(g["er"], device=dev, dtype=torch.long),
                ea=torch.tensor(g["ea"], device=dev), xr=torch.tensor(g["xr"], device=dev))


def run_fold(graphs, train_cuts, val_cut, variant, epochs, dev, seed=0, qweights=False, hidden=128, layers=1):
    """qweights: per-cutoff, per-activity loss weights min(1, prevalence / median prevalence)
    (same label-quality weighting as the tree models)."""
    torch.manual_seed(seed); np.random.seed(seed)
    prev = np.array([graphs[t]["y"].mean(0) for t in train_cuts])
    wq = np.clip(prev / np.maximum(np.median(prev, 0), 1e-9), 0, 1) if qweights else np.ones_like(prev)
    X = np.concatenate([graphs[t]["xu"] for t in train_cuts])
    Xl = np.sign(X) * np.log1p(np.abs(X))
    mu, sd = Xl.mean(0), Xl.std(0) + 1e-6
    G = [to_dev(graphs[t], mu, sd, dev) for t in train_cuts]
    gv = to_dev(graphs[val_cut], mu, sd, dev)
    d_u, d_r, d_e = G[0]["xu"].shape[1], G[0]["xr"].shape[1], G[0]["ea"].shape[1]
    net = RelNet(d_u, d_r, d_e, h=hidden, gnn=(variant == "gnn"), layers=layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(G))
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    W = [torch.tensor(w, device=dev, dtype=torch.float32) for w in wq]
    for ep in range(epochs):
        net.train()
        for i in np.random.permutation(len(G)):
            loss = (lossf(net(G[i]), G[i]["y"]) * W[i]).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        P = torch.sigmoid(net(gv)).cpu().numpy()
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["gnn", "mlp"], default="gnn")
    ap.add_argument("--name", default=None)
    ap.add_argument("--groups", default="T0,T1,R1")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="dataset.parquet")
    ap.add_argument("--hidden", type=int, default=128, help="width of every hidden layer")
    ap.add_argument("--layers", type=int, default=1, help="hidden layers per MLP block")
    ap.add_argument("--val", default="main", help="main (6 Wednesdays 09-02..10-07) or all (+6 earlier weeks)")
    a = ap.parse_args()
    name = a.name or a.variant
    dev = "cuda"
    df = pd.read_parquet(ART / a.data)
    ev = pd.read_parquet(ART / "events.parquet")
    G = feature_groups(feature_cols(df))
    feats = [f for g in a.groups.split(",") for f in G[g]]
    cuts_all = sorted(df.cutoff.unique())
    t0 = time.time()
    folds = []
    need = set()
    plan = []
    from exp2 import VAL_EXTRA
    for v in (VAL_EXTRA + list(VAL) if a.val == "all" else VAL):
        last = v - pd.Timedelta(days=8)
        tc = [c for c in cuts_all if c <= last and ((last - c).days % a.every == 0)]
        plan.append((v, tc)); need |= set(tc) | {v}
    graphs = build_graphs(df, feats, ev, sorted(need))
    print(f"[{name}] graphs built for {len(graphs)} cutoffs in {time.time() - t0:.0f}s", flush=True)
    for v, tc in plan:
        P = run_fold(graphs, tc, v, a.variant, a.epochs, dev, a.seed, hidden=a.hidden, layers=a.layers)
        g = graphs[v]
        folds.append((v, P, g["y"].astype(int), g["UserID"]))
        print(f"    val {v.date()} train_cutoffs={len(tc)}", flush=True)
    S = summarize(folds)
    S["table"].to_csv(RES / f"exp_{name}.csv", index=False)
    pickle.dump(folds, open(ART / f"folds_{name}.pkl", "wb"))
    row = {"name": name, "groups": a.groups.replace(",", "+") + f" [{a.variant}]", "n_feats": len(feats),
           "start": None, "every": a.every, "wed_only": False, "seed": a.seed,
           "params": json.dumps({"epochs": a.epochs, "hidden": a.hidden, "layers": a.layers, "val": a.val}),
           "honest_score": round(S["honest_score"], 5), "honest_macro": round(S["honest_macro"], 5),
           "honest_micro": round(S["honest_micro"], 5), "oracle_score": round(S["oracle_score"], 5),
           "prevweek_th_score": round(S["prev_score"], 5), "minutes": round((time.time() - t0) / 60, 1),
           "note": "torch relational net", "backend": a.variant}
    p = RES / "experiments.tsv"
    pd.DataFrame([row]).to_csv(p, sep="\t", mode="a", header=not p.exists(), index=False)
    print(S["table"].round(4).to_string())
    print(json.dumps(row))


if __name__ == "__main__":
    main()
