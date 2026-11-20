"""Deep user <-> repo GNN designed from the Q2 data analysis (results/data_insights.md), rolling refit on the
last 6 labelled Saturdays.  No MLP member, no XGBoost inside.

  CUDA_VISIBLE_DEVICES=5 ASG2_DB=../data_n19/db ASG2_ART=../artifacts_n19 ../.venv/bin/python hgnn.py --layers 3 4 5

Design choices taken from the Q2 analysis
  * "Received" interactions (in_comment / in_review / in_review_comment) are almost never visible: the tables
    only record what cohort users do (3g).  They are NOT used as edges.  Users are connected through the
    repositories they act in, so 2 hops = user -> repo -> other cohort users in the same repo.
  * The repo neighbourhood is the strongest cross-table signal (3a/3b/3e): repo nodes carry the 7-day and
    28-day activity of every type by all users in the repo, and the number of distinct active users.
  * Own repos vs others' repos changes review behaviour (3f): each edge has an "owner" flag
    (repo owner login == user login).
  * Own objects matter (3c): edges also carry own_pr_merged / own_pr_closed / own_issue_closed and merged_by.
  * Recency matters (Q3): edge counts at 7 d and 28 d and the days since the last action on that edge.

Graph at cutoff t (rows known before t only: avail < t; events in [t - 28 d, t)):
  user nodes  = the sample users of t, input = the 52 S3 features (sign-log, standardised on the training cutoffs)
  repo nodes  = repos the sample users acted in, input 16 = per activity log1p counts 7 d / 28 d (12),
                distinct active users 7 d / 28 d, is_org, log1p(number of sample users linked)
  edges       = user -- repo, input 22 = log1p counts 7 d / 28 d of 10 kinds (6 activities, merged_by,
                own_pr_merged, own_pr_closed, own_issue_closed), days since the last action / 28, owner flag
Model, L >= 3 message-passing layers, hidden 128, residual + LayerNorm:
  h_u = enc(x_u), h_r = enc(x_r), h_e = enc(x_e)
  repeat L times:  h_r <- LN(h_r + MLP([h_r, mean_e MLP([h_u, h_e])]))
                   h_u <- LN(h_u + MLP([h_u, mean_e MLP([h_r, h_e])]))
  logits = MLP([enc(x_u), h_u])  (6 outputs, BCE)
Training as gnn.run_fold: AdamW lr 2e-3, wd 1e-4, cosine, 8 epochs, one cutoff graph per step.
Writes results/hgnn_L<k>.csv / .json and artifacts/hgnn_probs.pkl.
Limitation (same as S3): logins / repo owners / is_org are the final table values, not the values at t.
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

from common import ART, DAY, RES, ROOT, TARGETS, load_tables
from modeling import YCOLS, official_score, train_window
from pipeline import decide, outage_targets, val_weeks

EDGE_KINDS = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12]     # no in_comment / in_review / in_review_comment (7, 8, 9)


def build_graphs(df, feats, ev, cuts, repo_owner_uid, repo_is_org):
    ev = ev.sort_values("avail", kind="stable")
    av = ev.avail.to_numpy()
    graphs = {}
    for t in cuts:
        ts = int(pd.Timestamp(t).value // 10**9)
        d = df[df.cutoff == t]
        uids = d.uid.to_numpy()
        e = ev.iloc[:np.searchsorted(av, ts)]
        e = e[(e.ts >= ts - 28 * DAY) & (e.ts < ts)]
        # repo inputs from all users
        ea_ = e[e.kind < 6]
        r7 = ea_[ea_.ts >= ts - 7 * DAY].groupby(["repo", "kind"]).size().unstack(fill_value=0).reindex(columns=range(6), fill_value=0)
        r28 = ea_.groupby(["repo", "kind"]).size().unstack(fill_value=0).reindex(columns=range(6), fill_value=0)
        u7 = ea_[ea_.ts >= ts - 7 * DAY].groupby("repo").uid.nunique()
        u28 = ea_.groupby("repo").uid.nunique()
        # edges of the sample users
        es = e[e.uid.isin(uids) & e.kind.isin(EDGE_KINDS)]
        c28 = es.groupby(["uid", "repo", "kind"]).size().unstack(fill_value=0).reindex(columns=EDGE_KINDS, fill_value=0)
        c7 = es[es.ts >= ts - 7 * DAY].groupby(["uid", "repo", "kind"]).size().unstack(fill_value=0) \
            .reindex(index=c28.index, columns=EDGE_KINDS, fill_value=0)
        last = es.groupby(["uid", "repo"]).ts.max().reindex(c28.index)
        idx = c28.index.to_frame(index=False)
        repos, ridx = np.unique(idx.repo.to_numpy(), return_inverse=True)
        owner = (repo_owner_uid[idx.repo.to_numpy()] == idx.uid.to_numpy()).astype(np.float32)
        xe = np.concatenate([np.log1p(c7.to_numpy(np.float32)), np.log1p(c28.to_numpy(np.float32)),
                             ((ts - last.to_numpy()) / (28 * DAY)).astype(np.float32)[:, None], owner[:, None]], 1)
        nlink = np.bincount(ridx, minlength=len(repos)).astype(np.float32)
        xr = np.concatenate([np.log1p(r7.reindex(repos, fill_value=0).to_numpy(np.float32)),
                             np.log1p(r28.reindex(repos, fill_value=0).to_numpy(np.float32)),
                             np.log1p(u7.reindex(repos, fill_value=0).to_numpy(np.float32))[:, None],
                             np.log1p(u28.reindex(repos, fill_value=0).to_numpy(np.float32))[:, None],
                             repo_is_org[repos].astype(np.float32)[:, None], np.log1p(nlink)[:, None]], 1)
        pos = pd.Series(np.arange(len(uids)), index=uids)
        graphs[t] = dict(xu=d[feats].to_numpy(np.float32), y=d[YCOLS].to_numpy(np.float32),
                         eu=pos.loc[idx.uid.to_numpy()].to_numpy(), er=ridx, xe=xe, xr=xr, uid=uids)
    return graphs


def block(i, o, h, p=0.1):
    return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


class HGNN(nn.Module):
    def __init__(self, d_u, d_r, d_e, h=128, L=3, p=0.1):
        super().__init__()
        self.L = L
        self.enc_u, self.enc_r, self.enc_e = block(d_u, h, h, p), block(d_r, h, h, p), block(d_e, h, h, p)
        self.msg_ur = nn.ModuleList([block(2 * h, h, h, p) for _ in range(L)])
        self.upd_r = nn.ModuleList([block(2 * h, h, h, p) for _ in range(L)])
        self.msg_ru = nn.ModuleList([block(2 * h, h, h, p) for _ in range(L)])
        self.upd_u = nn.ModuleList([block(2 * h, h, h, p) for _ in range(L)])
        self.ln_r = nn.ModuleList([nn.LayerNorm(h) for _ in range(L)])
        self.ln_u = nn.ModuleList([nn.LayerNorm(h) for _ in range(L)])
        self.head = block(2 * h, 6, h, p)

    @staticmethod
    def _mean(src, idx, n):
        out = torch.zeros(n, src.shape[1], device=src.device).index_add_(0, idx, src)
        cnt = torch.zeros(n, device=src.device).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
        return out / cnt.clamp(min=1).unsqueeze(1)

    def forward(self, g):
        hu0 = self.enc_u(g["xu"])
        hu, hr, he = hu0, self.enc_r(g["xr"]), self.enc_e(g["xe"])
        eu, er = g["eu"], g["er"]
        for l in range(self.L):
            mr = self._mean(self.msg_ur[l](torch.cat([hu[eu], he], 1)), er, hr.shape[0])
            hr = self.ln_r[l](hr + self.upd_r[l](torch.cat([hr, mr], 1)))
            mu = self._mean(self.msg_ru[l](torch.cat([hr[er], he], 1)), eu, hu.shape[0])
            hu = self.ln_u[l](hu + self.upd_u[l](torch.cat([hu, mu], 1)))
        return self.head(torch.cat([hu0, hu], 1))


def to_dev(g, st, dev):
    f = lambda x: np.sign(x) * np.log1p(np.abs(x))
    T = lambda a, dt=torch.float32: torch.tensor(a, device=dev, dtype=dt)
    return dict(xu=T((f(g["xu"]) - st["mu"]) / st["sd"]), xr=T((g["xr"] - st["rmu"]) / st["rsd"]),
                xe=T((g["xe"] - st["emu"]) / st["esd"]), y=T(g["y"]),
                eu=T(g["eu"], torch.long), er=T(g["er"], torch.long))


def run_fold(graphs, train_cuts, val_cut, L, epochs=8, seed=0, h=128, dev="cuda", lr=2e-3, p=0.1, wd=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cat = lambda k: np.concatenate([graphs[t][k] for t in train_cuts])
    xu = np.sign(cat("xu")) * np.log1p(np.abs(cat("xu")))
    st = dict(mu=xu.mean(0), sd=xu.std(0) + 1e-6, rmu=cat("xr").mean(0), rsd=cat("xr").std(0) + 1e-6,
              emu=cat("xe").mean(0), esd=cat("xe").std(0) + 1e-6)
    G = [to_dev(graphs[t], st, dev) for t in train_cuts]
    gv = to_dev(graphs[val_cut], st, dev)
    net = HGNN(G[0]["xu"].shape[1], G[0]["xr"].shape[1], G[0]["xe"].shape[1], h=h, L=L, p=p).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(G))
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        net.train()
        for i in np.random.permutation(len(G)):
            loss = lossf(net(G[i]), G[i]["y"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(gv)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, nargs="+", default=[8])
    ap.add_argument("--hidden", type=int, nargs="+", default=[128])
    ap.add_argument("--lr", type=float, nargs="+", default=[2e-3])
    ap.add_argument("--dropout", type=float, nargs="+", default=[0.1])
    ap.add_argument("--wd", type=float, nargs="+", default=[1e-4])
    ap.add_argument("--tag", default="", help="suffix of the result files, e.g. _s3 for a 3-seed run")
    a = ap.parse_args()
    assert min(a.layers) >= 3
    cfg = json.load(open(ROOT / "configs/final_config_s3.json"))
    feats = cfg["xgb_features"]
    df = pd.read_parquet(ART / cfg["dataset"])
    ev = pd.read_parquet(ART / "events.parquet")
    T = load_tables()
    login2uid = pd.Series(T["Users"].user_id.to_numpy(), index=T["Users"].login.to_numpy())
    login2uid = login2uid[~login2uid.index.duplicated()]
    repo_owner_uid = login2uid.reindex(T["Repositories"].owner_login.to_numpy()).fillna(-1).to_numpy().astype(np.int64)
    repo_is_org = T["Repositories"].org_id.notna().to_numpy()
    weeks = val_weeks("last6", df)
    drop = outage_targets(ev, weeks)
    t0 = time.time()
    graphs = build_graphs(df, feats, ev, sorted(df.cutoff.unique()), repo_owner_uid, repo_is_org)
    ne = [len(g["eu"]) for g in graphs.values()]
    print(f"graphs: {len(graphs)} cutoffs, edges {min(ne)}..{max(ne)}, edge dim {graphs[weeks[0]]['xe'].shape[1]}, "
          f"repo dim {graphs[weeks[0]]['xr'].shape[1]} [{(time.time() - t0) / 60:.1f} min]", flush=True)
    store, summ = {}, []
    import itertools
    for L, ep, h, lr, p, wd in itertools.product(a.layers, a.epochs, a.hidden, a.lr, a.dropout, a.wd):
        key = dict(layers=L, epochs=ep, hidden=h, lr=lr, dropout=p, wd=wd)
        rows = []
        for v in weeks:
            tr_cuts = sorted(train_window(df, v, every=1).cutoff.unique())
            assert tr_cuts[-1] + pd.Timedelta(hours=4, days=12) <= v
            P = np.mean([run_fold(graphs, tr_cuts, v, L, ep, sd, h=h, lr=lr, p=p, wd=wd) for sd in a.seeds], 0)
            Y = graphs[v]["y"]
            wk = str(v.date())
            keep = [j for j, c in enumerate(TARGETS) if c not in drop[wk]]
            pred = decide(P, cfg)
            s = official_score(pred, Y)
            rows.append({**key, "val": wk, "score": s["score"], "macro_f1": s["macro_f1"], "micro_f1": s["micro_f1"],
                         "outage_aware_score": official_score(pred[:, keep], Y[:, keep])["score"], "dropped": ",".join(drop[wk])})
            store[(L, ep, h, lr, p, wd, wk)] = P.astype(np.float32)
        R = pd.DataFrame(rows)
        summ.append({**key, "score": 100 * R.score.mean(), "outage_aware_score": 100 * R.outage_aware_score.mean(),
                     "weeks": " ".join(f"{100 * x:.2f}" for x in R.score)})
        print(f"{key}: mean {100 * R.score.mean():.2f} / outage-aware {100 * R.outage_aware_score.mean():.2f}"
              f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        pd.DataFrame(summ).sort_values("score", ascending=False).to_csv(RES / f"hgnn{a.tag}_summary.csv", index=False)
    pickle.dump({"probs": store, "uids": {str(w.date()): graphs[w]["uid"] for w in weeks}},
                open(ART / f"hgnn_probs{a.tag}.pkl", "wb"))


if __name__ == "__main__":
    main()
