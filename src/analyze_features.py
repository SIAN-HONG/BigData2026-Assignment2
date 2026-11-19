"""Feature analysis (Phase 3 planning): lagged cross-activity correlations and repo-centric
candidate features, all point-in-time (rows with avail < t only).

Samples: the dataset rows (users, labels) at the Wednesday cutoffs 2026-03-04 .. 10-07, 12:00 UTC.
Outputs: results/analysis/*.csv and figures/analysis_*.png

  A  data overview and repo coverage (how much of a repo's PRs / issues we can see)
  B  lag correlations: 1[user did kind k in week -L] vs label y_j of the target week, raw and
     partial (within-cutoff demeaned, controls = own wk1/wk2 of j, log activity, active days)
  C  repo-centric / pending-work candidate features: univariate AUC, partial correlation
     beyond the controls + the existing repo features, top-decile lift

  python src/analyze_features.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import ART, DAY, FIG, KIND_NAMES, RES, TARGETS, load_tables, ts2s

OUT = RES / "analysis"
OUT.mkdir(exist_ok=True)
CUTS = pd.date_range("2026-03-04 12:00", "2026-10-07 12:00", freq="7D")
NK, NL = len(KIND_NAMES), 8
BIG = np.iinfo(np.int64).max
EXIST = ["wk1_", "wk2_", "adays28_any", "c28d_", "review_demand_open_pr", "repo_wmean_r_open_pr",
         "repo_max_r_open_pr", "own_open_pr90", "own_open_issue90", "n_repos28"]


def load_samples():
    cols = ["cutoff", "uid", "UserID"] + [f"y_{c}" for c in TARGETS]
    cols += [f"wk1_{c}" for c in TARGETS] + [f"wk2_{c}" for c in TARGETS] + [f"c28d_{c}" for c in TARGETS]
    cols += ["adays28_any", "review_demand_open_pr", "repo_wmean_r_open_pr", "repo_max_r_open_pr",
             "own_open_pr90", "own_open_issue90", "n_repos28"]
    df = pd.read_parquet(ART / "dataset_p2.parquet", columns=cols, filters=[("cutoff", "in", list(CUTS))])
    return df.sort_values("cutoff", kind="stable").reset_index(drop=True)   # row order = CUTS order


# ---------------------------------------------------------------------------------------------
# A. overview / coverage
# ---------------------------------------------------------------------------------------------
def overview(T, ev):
    u = T["Users"]
    cohort = u.is_cohort.to_numpy()
    rows = []
    for n, d in T.items():
        rows.append({"table": n, "rows": len(d)})
    pr, iss = T["PullRequests"], T["Issues"]
    a = pr.author_id.to_numpy(dtype=float, na_value=np.nan)
    known = ~np.isnan(a)
    pr_cohort = cohort[a[known].astype(int)].mean()
    m = (iss.is_pull_request == False).to_numpy()
    ia = iss.author_id.to_numpy(dtype=float, na_value=np.nan)[m]
    iss_cohort = cohort[ia[~np.isnan(ia)].astype(int)].mean()
    cov = {"PR rows with known author": known.mean(), "PRs authored by cohort users": pr_cohort,
           "issues (non-PR) authored by cohort users": iss_cohort,
           "PRs with merged_at": pr.merged_at.notna().mean(), "PRs with closed_at": pr.closed_at.notna().mean(),
           "issues with closed_at": iss.closed_at[m].notna().mean(),
           "issues with assignee": iss.assignee_id[m].notna().mean(),
           "PR first_seen - created (median days)": ((pr.first_seen_at - pr.created_at).dt.total_seconds() / DAY).median()}
    # how a cohort user's PR / issue is "seen": PRs in repos touched by cohort users
    pd.DataFrame(rows).to_csv(OUT / "A_tables.csv", index=False)
    pd.Series(cov).to_csv(OUT / "A_coverage.csv", header=["value"])
    return cov


# ---------------------------------------------------------------------------------------------
# B. lag correlations
# ---------------------------------------------------------------------------------------------
def lag_indicators(ev, t, uids):
    """X[n, NK, NL]: did the user do kind k in week L (L=1..8 -> [t-7L d, t-7(L-1) d))."""
    n_av = np.searchsorted(ev.avail.to_numpy(), t, side="left")
    e = ev.iloc[:n_av]
    e = e[(e.ts < t) & (e.ts > t - NL * 7 * DAY)]
    pos = pd.Series(np.arange(len(uids)), index=uids)
    e = e[e.uid.isin(pos.index)]
    X = np.zeros((len(uids), NK, NL), np.int8)
    X[pos.loc[e.uid].to_numpy(), e.kind.to_numpy().astype(int), ((t - e.ts.to_numpy()) // (7 * DAY)).astype(int)] = 1
    return X


def demean_by(a, g):
    a = np.asarray(a, dtype=np.float64)
    out = a.copy()
    for v in np.unique(g):
        m = g == v
        out[m] -= a[m].mean(0)
    return out


def corr_cols(A, b):
    """Pearson correlation of every column of A with vector b (both centred inside)."""
    A = A - A.mean(0)
    b = b - b.mean()
    den = np.sqrt((A ** 2).sum(0) * (b ** 2).sum()) + 1e-12
    return (A * b[:, None]).sum(0) / den


def residualize(A, Z):
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(Z1, A, rcond=None)
    return A - Z1 @ beta


def controls(df, j):
    c = TARGETS[j]
    return np.column_stack([df[f"wk1_{c}"], df[f"wk2_{c}"], np.log1p(df[f"c28d_{c}"]),
                            np.log1p(df[[f"c28d_{x}" for x in TARGETS]].sum(1)), df["adays28_any"]])


def lag_analysis(df, ev):
    Xs = []
    for t in CUTS:
        d = df[df.cutoff == t]
        Xs.append(lag_indicators(ev, int(t.value // 10**9), d.uid.to_numpy()))
    X = np.concatenate(Xs).reshape(len(df), NK * NL).astype(np.float64)
    g = df.cutoff.to_numpy()
    Xd = demean_by(X, g)
    rows = []
    for j, c in enumerate(TARGETS):
        y = df[f"y_{c}"].to_numpy(np.float64)
        yd = demean_by(y, g)
        raw = corr_cols(Xd, yd)
        Z = demean_by(controls(df, j), g)
        part = corr_cols(residualize(Xd, Z), residualize(yd[:, None], Z)[:, 0])
        base = y.mean()
        for k in range(NK):
            for L in range(NL):
                i = k * NL + L
                x = X[:, i] > 0
                rows.append({"target": c, "feature_kind": KIND_NAMES[k], "lag_week": L + 1,
                             "raw_corr": raw[i], "partial_corr": part[i], "share_x": x.mean(),
                             "p_y_given_x": y[x].mean() if x.any() else np.nan,
                             "lift": (y[x].mean() / base) if x.any() else np.nan})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "B_lag_corr.csv", index=False)
    return R


# ---------------------------------------------------------------------------------------------
# C. repo-centric candidate features
# ---------------------------------------------------------------------------------------------
class RepoState:
    def __init__(self, T):
        u = T["Users"]
        self.cohort = u.is_cohort.to_numpy()
        pr = T["PullRequests"]
        self.pr = pd.DataFrame({
            "pid": np.arange(len(pr)), "repo": pr.repo_id.fillna(-1).astype("int64").to_numpy(),
            "author": pr.author_id.fillna(-1).astype("int64").to_numpy(),
            "fs": ts2s(pr.first_seen_at), "closed": ts2s(pr.closed_at),
            "merged": ts2s(pr.merged_at), "merger": pr.merged_by_id.fillna(-1).astype("int64").to_numpy()})
        iss = T["Issues"]
        isp = iss.is_pull_request.fillna(False).to_numpy()
        self.iss_all = pd.DataFrame({
            "iid": np.arange(len(iss)), "repo": iss.repo_id.fillna(-1).astype("int64").to_numpy(),
            "author": iss.author_id.fillna(-1).astype("int64").to_numpy(), "is_pr": isp,
            "fs": ts2s(iss.first_seen_at), "closed": ts2s(iss.closed_at),
            "assignee": iss.assignee_id.fillna(-1).astype("int64").to_numpy(),
            "pr": iss.pull_request_id.fillna(-1).astype("int64").to_numpy()})
        rv = T["PRReviews"]
        self.rv = pd.DataFrame({"pid": rv.pr_id.fillna(-1).astype("int64").to_numpy(),
                                "uid": rv.reviewer_id.fillna(-1).astype("int64").to_numpy(),
                                "ts": ts2s(rv.submitted_at), "avail": ts2s(rv.first_seen_at)})
        rc = T["PRReviewComments"]
        self.rc = pd.DataFrame({"pid": rc.pr_id.fillna(-1).astype("int64").to_numpy(),
                                "uid": rc.author_id.fillna(-1).astype("int64").to_numpy(),
                                "ts": ts2s(rc.created_at)})
        ic = T["IssueComments"]
        self.ic = pd.DataFrame({"iid": ic.issue_id.fillna(-1).astype("int64").to_numpy(),
                                "uid": ic.author_id.fillna(-1).astype("int64").to_numpy(),
                                "ts": ts2s(ic.created_at)})

    def features(self, ev, t, uids):
        F = pd.DataFrame(index=uids)
        us = pd.Index(uids)
        n_av = np.searchsorted(ev.avail.to_numpy(), t, side="left")
        past = ev.iloc[:n_av]
        past = past[past.ts < t]
        own = past[(past.kind < 6) & (past.ts >= t - 28 * DAY) & (past.repo >= 0) & past.uid.isin(us)]
        ur = own.groupby(["uid", "repo"]).size().rename("n").reset_index()
        ur["w"] = ur.n / ur.groupby("uid").n.transform("sum")
        repos = np.unique(ur.repo.to_numpy())
        # ---- repo state at t (rows known before t, first seen within 90 days) ----------------
        pr = self.pr[(self.pr.fs < t) & (self.pr.fs >= t - 90 * DAY)]
        open_pr = pr[pr.closed >= t]
        rv_known = self.rv[(self.rv.avail < t) & (self.rv.ts < t)]
        reviewed = np.isin(open_pr.pid.to_numpy(), rv_known.pid.to_numpy())
        iss = self.iss_all[(self.iss_all.fs < t) & (self.iss_all.fs >= t - 90 * DAY)]
        oi = iss[(~iss.is_pr) & (iss.closed >= t)]
        ic7 = self.ic[(self.ic.ts < t) & (self.ic.ts >= t - 7 * DAY)]
        ic3 = ic7[ic7.ts >= t - 3 * DAY]
        oi_hot = oi[np.isin(oi.iid.to_numpy(), ic3.iid.to_numpy())]
        act3 = np.union1d(self.rc[(self.rc.ts < t) & (self.rc.ts >= t - 3 * DAY)].pid.to_numpy(),
                          rv_known[rv_known.ts >= t - 3 * DAY].pid.to_numpy())
        pr_conv = self.iss_all[self.iss_all.is_pr & (self.iss_all.pr >= 0)]
        hot_conv_pr = pr_conv.pr.to_numpy()[np.isin(pr_conv.iid.to_numpy(), ic3.iid.to_numpy())]
        open_pr_hot = open_pr[np.isin(open_pr.pid.to_numpy(), np.union1d(act3, hot_conv_pr))]
        R = pd.DataFrame(index=repos)
        R["open_pr"] = open_pr.groupby("repo").size()
        R["open_pr_unreviewed"] = open_pr[~reviewed].groupby("repo").size()
        R["open_pr_noncohort_author"] = open_pr[(open_pr.author >= 0) & ~self.cohort[np.maximum(open_pr.author, 0)]].groupby("repo").size()
        R["open_pr_stale7"] = open_pr[open_pr.fs < t - 7 * DAY].groupby("repo").size()
        R["open_pr_hot3"] = open_pr_hot.groupby("repo").size()
        R["new_pr7"] = pr[pr.fs >= t - 7 * DAY].groupby("repo").size()
        R["open_issue"] = oi.groupby("repo").size()
        R["open_issue_hot3"] = oi_hot.groupby("repo").size()
        R["new_issue7"] = iss[(~iss.is_pr) & (iss.fs >= t - 7 * DAY)].groupby("repo").size()
        R["merged7"] = pr[(pr.merged < t) & (pr.merged >= t - 7 * DAY)].groupby("repo").size()
        R = R.reindex(repos).fillna(0)
        J = ur.join(R, on="repo")
        for c in R.columns:
            F[f"rp_wmean_{c}"] = (J[c] * J.w).groupby(J.uid).sum().reindex(uids).fillna(0)
            F[f"rp_sum_{c}"] = J.groupby("uid")[c].sum().reindex(uids).fillna(0)
        F["rp_open_pr_cohort_share"] = ((F.rp_sum_open_pr - F.rp_sum_open_pr_noncohort_author) /
                                        F.rp_sum_open_pr.where(F.rp_sum_open_pr > 0)).fillna(-1)
        # ---- user-specific pending work -------------------------------------------------------
        # maintainer backlog: open PRs by others in repos where the user merged a PR (merge known < t, 90 d)
        mr = self.pr[(self.pr.merged < t) & (self.pr.merged >= t - 90 * DAY) & np.isin(self.pr.merger, uids)]
        mrep = mr[["merger", "repo"]].drop_duplicates().rename(columns={"merger": "uid"})
        op_r = open_pr.groupby("repo").size().rename("k")
        F["u_maint_backlog_open_pr"] = mrep.join(op_r, on="repo").fillna(0).groupby("uid").k.sum().reindex(uids).fillna(0)
        F["u_maint_backlog_unreviewed"] = mrep.join(open_pr[~reviewed].groupby("repo").size().rename("k"), on="repo") \
            .fillna(0).groupby("uid").k.sum().reindex(uids).fillna(0)
        # open PRs authored by people whose PRs the user reviewed in the last 90 days
        rv90 = rv_known[(rv_known.ts >= t - 90 * DAY) & np.isin(rv_known.uid, uids)]
        rel = rv90.merge(self.pr[["pid", "author"]], on="pid")
        rel = rel[(rel.author >= 0) & (rel.author != rel.uid)][["uid", "author"]].drop_duplicates()
        opa = open_pr[open_pr.author >= 0].groupby("author").size().rename("k")
        F["u_open_pr_by_reviewed_authors"] = rel.join(opa, on="author").fillna(0).groupby("uid").k.sum().reindex(uids).fillna(0)
        rel_new = open_pr[(open_pr.fs >= t - 7 * DAY) & (open_pr.author >= 0)].groupby("author").size().rename("k")
        F["u_new_pr7_by_reviewed_authors"] = rel.join(rel_new, on="author").fillna(0).groupby("uid").k.sum().reindex(uids).fillna(0)
        # own open PRs that received review / review comment / conversation comment from others recently
        mine = open_pr[np.isin(open_pr.author, uids)][["pid", "author"]]
        inc = pd.concat([rv_known[["pid", "uid", "ts"]],
                         self.rc[self.rc.ts < t][["pid", "uid", "ts"]],
                         self.ic[self.ic.ts < t].merge(pr_conv[["iid", "pr"]], on="iid")
                         .rename(columns={"pr": "pid"})[["pid", "uid", "ts"]]])
        j = mine.merge(inc, on="pid")
        j = j[j.uid != j.author]
        for w in (3, 7):
            F[f"u_own_pr_feedback{w}d"] = j[j.ts >= t - w * DAY].groupby("author").pid.nunique().reindex(uids).fillna(0)
        # own open issues with comments by others in the last 7 days
        myi = oi[np.isin(oi.author, uids)][["iid", "author"]].merge(ic7, on="iid")
        F["u_own_issue_feedback7d"] = myi[myi.uid != myi.author].groupby("author").iid.nunique().reindex(uids).fillna(0)
        # open issues assigned to the user (assignee known at the first event)
        F["u_assigned_open_issue"] = oi[np.isin(oi.assignee, uids)].groupby("assignee").size().reindex(uids).fillna(0)
        # threads waiting for the user's reply: the user commented in the last 30 d and someone else
        # commented after the user's last comment
        icp = self.ic[(self.ic.ts < t) & (self.ic.ts >= t - 30 * DAY)]
        mylast = icp[np.isin(icp.uid, uids)].groupby(["uid", "iid"]).ts.max().rename("my").reset_index()
        lastc = icp.sort_values("ts").groupby("iid").tail(1)[["iid", "uid", "ts"]].rename(columns={"uid": "last_uid", "ts": "last_ts"})
        w = mylast.merge(lastc, on="iid")
        w = w[(w.last_uid != w.uid) & (w.last_ts > w.my)]
        F["u_threads_awaiting_reply"] = w.groupby("uid").size().reindex(uids).fillna(0)
        return F


def auc(x, y):
    """Rank AUC of score x for binary y (ties averaged)."""
    r = pd.Series(x).rank().to_numpy()
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def repo_analysis(df, ev, T):
    if (OUT / "C_features.parquet").exists():
        F = pd.read_parquet(OUT / "C_features.parquet")
    else:
        st = RepoState(T)
        Fs = []
        for t in CUTS:
            d = df[df.cutoff == t]
            Fs.append(st.features(ev, int(t.value // 10**9), d.uid.to_numpy()).reset_index(drop=True))
            print("  C", t.date(), flush=True)
        F = pd.concat(Fs, ignore_index=True)
        F.to_parquet(OUT / "C_features.parquet")
    g = df.cutoff.to_numpy()
    Fd = demean_by(np.log1p(F.clip(lower=0).to_numpy()), g)
    exist = demean_by(np.log1p(df[["review_demand_open_pr", "repo_wmean_r_open_pr", "repo_max_r_open_pr",
                                   "own_open_pr90", "own_open_issue90", "n_repos28"]].clip(lower=0).to_numpy(np.float64)), g)
    rows = []
    for jj, c in enumerate(TARGETS):
        y = df[f"y_{c}"].to_numpy()
        yd = demean_by(y, g)
        Z = np.column_stack([demean_by(controls(df, jj), g), exist])
        part = corr_cols(residualize(Fd, Z), residualize(yd[:, None], Z)[:, 0])
        for i, f in enumerate(F.columns):
            x = F[f].to_numpy()
            nz = x > 0
            q = np.quantile(x[nz], 0.9) if nz.sum() > 50 else np.inf
            top = x >= q
            rows.append({"target": c, "feature": f, "share_nonzero": nz.mean(), "auc": auc(x, y),
                         "partial_corr": part[i],
                         "p_y_nonzero": y[nz].mean() if nz.any() else np.nan,
                         "p_y_zero": y[~nz].mean() if (~nz).any() else np.nan,
                         "lift_top10pct_of_nonzero": (y[top].mean() / y.mean()) if top.any() else np.nan})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "C_repo_features.csv", index=False)
    # existing features, same metrics, for reference
    ref = []
    for jj, c in enumerate(TARGETS):
        y = df[f"y_{c}"].to_numpy()
        for f in ["review_demand_open_pr", "repo_wmean_r_open_pr", "repo_max_r_open_pr", "own_open_pr90",
                  "own_open_issue90", f"wk1_{c}", "adays28_any"]:
            ref.append({"target": c, "feature": f, "auc": auc(df[f].to_numpy(), y)})
    pd.DataFrame(ref).to_csv(OUT / "C_existing_reference.csv", index=False)
    return R


def figures(B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for col, name in (("raw_corr", "raw"), ("partial_corr", "partial")):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
        vmax = np.nanmax(np.abs(B[col])) if name == "partial" else 0.5
        for ax, c in zip(axes.flat, TARGETS):
            M = B[B.target == c].pivot(index="feature_kind", columns="lag_week", values=col).reindex(KIND_NAMES)
            im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(f"target: {c}")
            ax.set_xticks(range(NL)); ax.set_xticklabels(range(1, NL + 1))
            ax.set_yticks(range(NK)); ax.set_yticklabels(KIND_NAMES, fontsize=8)
            ax.set_xlabel("weeks before cutoff")
            for (i, k), v in np.ndenumerate(M.to_numpy()):
                ax.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=axes, shrink=0.6)
        fig.suptitle(f"{name} correlation: did kind k in week -L  vs  label of the target week "
                     + ("(within-cutoff demeaned)" if name == "raw" else
                        "(after controlling own wk1/wk2/c28 of the target, total activity, active days)"))
        fig.savefig(FIG / f"analysis_lag_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def main():
    T = load_tables()
    ev = pd.read_parquet(ART / "events.parquet")
    df = load_samples()
    print("samples", df.shape, df.cutoff.nunique(), "cutoffs", flush=True)
    print(overview(T, ev), flush=True)
    B = pd.read_csv(OUT / "B_lag_corr.csv") if (OUT / "B_lag_corr.csv").exists() else lag_analysis(df, ev)
    figures(B)
    print("B done", flush=True)
    repo_analysis(df, ev, T)
    print("C done", flush=True)


if __name__ == "__main__":
    main()
