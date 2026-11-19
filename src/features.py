"""Point-in-time feature construction for one cutoff t.

Only event rows with avail < t are used; state of PRs / issues is reconstructed at t from
first_seen_at and the "known only later" columns (closed_at / merged_at used only if < t).

Feature groups (used for the ablations in Q2 / Q3):
  T0  basic temporal  : per-activity counts over 7 / 28 / 90 days
  T1  rich temporal   : more windows, recency, weekly-lag indicators, active days, decayed
                        counts, trend ratios, tenure, cohort-wide activity level
  R1  user-centric relational : incoming interactions on the user's PRs / issues, maintainer
                        role (merged_by), point-in-time open PR / issue state
  R2  repo / collaborator relational : aggregates over the repositories the user works in
                        (2-hop user->repo->users), open-PR review demand in those repos,
                        org / ownership info
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DAY, KIND_NAMES, OUTAGE_REL, TARGETS, outage_flags

N_KIND = len(KIND_NAMES)
ACT = list(range(6))
REL_KINDS = list(range(6, N_KIND))


def _pivot_count(sub: pd.DataFrame, uids: np.ndarray, kinds, prefix: str) -> pd.DataFrame:
    c = sub.groupby(["uid", "kind"]).size().unstack(fill_value=0)
    c = c.reindex(index=uids, columns=kinds, fill_value=0)
    c.columns = [f"{prefix}_{KIND_NAMES[k]}" for k in kinds]
    return c


class FeatureBuilder:
    def __init__(self, T: dict, ev: pd.DataFrame, impute=False):
        # False, or the method used to fill detected collection outages before computing features
        # (impute_outages): "copy" (same span whole weeks earlier), "week" (mean of the previous 7 days),
        # "prevday" (the previous day repeated); True means "copy"
        self.impute = impute
        self.ev = ev
        self.ev_avail = ev.avail.to_numpy()
        users = T["Users"]
        repos = T["Repositories"]
        self.login = users.login.to_numpy()
        self.repo_owner = repos.owner_login.to_numpy()
        self.repo_is_org = repos.org_id.notna().to_numpy()
        pr = T["PullRequests"]
        from common import ts2s
        self.pr = pd.DataFrame({
            "repo": pr.repo_id.fillna(-1).astype("int64").to_numpy(),
            "author": pr.author_id.fillna(-1).astype("int64").to_numpy(),
            "fs": ts2s(pr.first_seen_at), "closed": ts2s(pr.closed_at),
            "draft": pr.draft.fillna(False).to_numpy(), "cross": pr.is_cross_repo.fillna(False).to_numpy(),
            "merged": ts2s(pr.merged_at)})
        self.cohort = users.is_cohort.to_numpy()
        # --- Phase 2 (groups T2 / R3) inputs -------------------------------------------
        repos_gid = pd.Series(np.arange(len(repos)), index=repos.github_repo_id.to_numpy())
        repos_gid = repos_gid[~repos_gid.index.duplicated()]
        head_row = repos_gid.reindex(pr.head_repo_id.to_numpy()).to_numpy()
        cross = pr.is_cross_repo.fillna(False).to_numpy()
        base = pr.repo_id.fillna(-1).astype("int64").to_numpy()
        head_repo = np.where(cross, np.nan_to_num(head_row, nan=-1), base).astype("int64")
        hr = pr.head_ref.fillna("").to_numpy()
        self.pr_branch = pd.DataFrame({"repo": head_repo, "branch": hr, "fs": ts2s(pr.first_seen_at)})
        self.pr_branch = self.pr_branch[(self.pr_branch.repo >= 0) & (self.pr_branch.branch != "")]
        pu = T["Pushes"]
        self.push = pd.DataFrame({"uid": pu.user_id.fillna(-1).astype("int64").to_numpy(),
                                  "repo": pu.repo_id.fillna(-1).astype("int64").to_numpy(),
                                  "branch": pu.branch.fillna("").to_numpy(), "ts": ts2s(pu.created_at)})
        ic = T["IssueComments"]
        self.ic = pd.DataFrame({"uid": ic.author_id.fillna(-1).astype("int64").to_numpy(),
                                "issue": ic.issue_id.fillna(-1).astype("int64").to_numpy(),
                                "ts": ts2s(ic.created_at)})
        iss_all = T["Issues"]
        self.iss_all = pd.DataFrame({"author": iss_all.author_id.fillna(-1).astype("int64").to_numpy(),
                                     "fs": ts2s(iss_all.first_seen_at),
                                     "pr": iss_all.pull_request_id.fillna(-1).astype("int64").to_numpy()})
        rv, rc = T["PRReviews"], T["PRReviewComments"]
        pr_fs = ts2s(pr.first_seen_at)
        rv_pid = rv.pr_id.fillna(-1).astype("int64").to_numpy()
        rc_pid = rc.pr_id.fillna(-1).astype("int64").to_numpy()
        # PR-thread actions: reviews (avail = first_seen) and line comments (avail = created)
        self.pr_act = pd.concat([
            pd.DataFrame({"uid": rv.reviewer_id.fillna(-1).astype("int64").to_numpy(), "pr": rv_pid,
                          "ts": ts2s(rv.submitted_at), "avail": ts2s(rv.first_seen_at)}),
            pd.DataFrame({"uid": rc.author_id.fillna(-1).astype("int64").to_numpy(), "pr": rc_pid,
                          "ts": ts2s(rc.created_at), "avail": ts2s(rc.created_at)})], ignore_index=True)
        self.pr_act = self.pr_act[self.pr_act.pr >= 0]
        self.pr_meta = pd.DataFrame({"author": pr.author_id.fillna(-1).astype("int64").to_numpy(), "fs": pr_fs})

        iss = T["Issues"]
        m = (iss.is_pull_request == False).fillna(False).to_numpy()
        self.iss = pd.DataFrame({
            "repo": iss.repo_id.fillna(-1).astype("int64").to_numpy()[m],
            "author": iss.author_id.fillna(-1).astype("int64").to_numpy()[m],
            "fs": ts2s(iss.first_seen_at)[m], "closed": ts2s(iss.closed_at)[m], "iid": np.arange(len(iss))[m]})

    # ------------------------------------------------------------------------------
    def build(self, t: int, uids: np.ndarray, groups=("T0", "T1", "R1", "R2", "TG", "T2", "R3")) -> pd.DataFrame:
        n_av = np.searchsorted(self.ev_avail, t, side="left")   # ev sorted by avail
        past = self.ev.iloc[:n_av]
        past = past[past.ts < t]
        if self.impute:
            past = impute_outages(past, t, "copy" if self.impute is True else self.impute)
        ts = past.ts.to_numpy()
        feats = []
        if "T0" in groups:
            for w in (7, 28, 90):
                feats.append(_pivot_count(past[ts >= t - w * DAY], uids, ACT, f"c{w}d"))
        if "T1" in groups:
            feats.append(self._temporal_rich(past, t, uids))
        if "R1" in groups:
            feats.append(self._rel_user(past, t, uids))
        if "R2" in groups:
            feats.append(self._rel_repo(past, t, uids))
        if "TG" in groups:
            feats.append(self._global_level(past, t, uids))
        if "T2" in groups:
            feats.append(self._temporal_v2(past, t, uids))
        if "R3" in groups:
            feats.append(self._rel_threads_branches(t, uids))
        if "O" in groups:
            feats.append(self._outage(past, t, uids))
        if "R4" in groups:
            feats.append(self._repo_traffic(past, t, uids))
        X = pd.concat(feats, axis=1)
        X.index.name = "uid"
        return X.astype("float32")

    # ------------------------------------------------------------------------------
    def _temporal_rich(self, past, t, uids):
        ts = past.ts.to_numpy()
        f = []
        for w in (1, 3, 14, 56):
            f.append(_pivot_count(past[ts >= t - w * DAY], uids, ACT, f"c{w}d"))
        f.append(_pivot_count(past, uids, ACT, "call"))
        own = past[past.kind < 6]
        # recency: days since last / first action of each kind
        g = own.groupby(["uid", "kind"]).ts
        last = g.max().unstack().reindex(index=uids, columns=ACT)
        first = g.min().unstack().reindex(index=uids, columns=ACT)
        rec = ((t - last) / DAY).fillna(999)
        rec.columns = [f"rec_{KIND_NAMES[k]}" for k in ACT]
        ten = ((t - first) / DAY).fillna(-1)
        ten.columns = [f"tenure_{KIND_NAMES[k]}" for k in ACT]
        f += [rec, ten]
        any_last = own.groupby("uid").ts.max().reindex(uids)
        any_first = own.groupby("uid").ts.min().reindex(uids)
        f.append(pd.DataFrame({"rec_any": ((t - any_last) / DAY).fillna(999),
                               "tenure_any": ((t - any_first) / DAY).fillna(-1)}, index=uids))
        # weekly-lag indicators: did the user do activity k in week -L (L = 1..8)?
        # these mirror the label definition one / two / ... weeks back
        o12 = own[own.ts >= t - 12 * 7 * DAY]
        wk = ((t - o12.ts.to_numpy()) // (7 * DAY)).astype(int)
        lag = pd.DataFrame({"uid": o12.uid.to_numpy(), "kind": o12.kind.to_numpy(), "wk": wk})
        lag = lag.drop_duplicates()
        cnt_w = lag.groupby(["uid", "kind"]).size().unstack().reindex(index=uids, columns=ACT).fillna(0)
        cnt_w.columns = [f"weeks12_{KIND_NAMES[k]}" for k in ACT]
        f.append(cnt_w)
        for L in range(8):
            x = lag[lag.wk == L].assign(v=1).pivot_table(index="uid", columns="kind", values="v", aggfunc="max")
            x = x.reindex(index=uids, columns=ACT).fillna(0)
            x.columns = [f"wk{L+1}_{KIND_NAMES[k]}" for k in ACT]
            f.append(x)
        # weeks since last "active week" streak: consecutive active weeks ending at lag 1
        # active days in the last 28 days, any own activity and per kind
        o28 = own[own.ts >= t - 28 * DAY]
        d = pd.DataFrame({"uid": o28.uid.to_numpy(), "kind": o28.kind.to_numpy(),
                          "d": (t - o28.ts.to_numpy()) // DAY}).drop_duplicates()
        ad = d.groupby(["uid", "kind"]).size().unstack().reindex(index=uids, columns=ACT).fillna(0)
        ad.columns = [f"adays28_{KIND_NAMES[k]}" for k in ACT]
        f.append(ad)
        f.append(d.drop_duplicates(["uid", "d"]).groupby("uid").size().reindex(uids).fillna(0)
                 .rename("adays28_any").to_frame())
        # exponentially decayed counts (half-life 3 / 14 days)
        for hl in (3, 14):
            o = own[own.ts >= t - 8 * hl * DAY]
            w = np.exp2(-(t - o.ts.to_numpy()) / (hl * DAY))
            e = pd.DataFrame({"uid": o.uid.to_numpy(), "kind": o.kind.to_numpy(), "w": w}) \
                .groupby(["uid", "kind"]).w.sum().unstack().reindex(index=uids, columns=ACT).fillna(0)
            e.columns = [f"ewm{hl}_{KIND_NAMES[k]}" for k in ACT]
            f.append(e)
        # day-of-week profile of the user's own activity in the last 90 days (UTC)
        o90 = own[own.ts >= t - 90 * DAY]
        dow = ((o90.ts.to_numpy() // DAY) + 3) % 7   # 0 = Monday
        wd = pd.DataFrame({"uid": o90.uid.to_numpy(), "we": dow >= 5}).groupby("uid").we.mean()
        f.append(wd.reindex(uids).fillna(-1).rename("weekend_share90").to_frame())
        X = pd.concat(f, axis=1)
        # trend ratios (last week vs average week of the last 4 / 12)
        for k in ACT:
            n = KIND_NAMES[k]
            c7 = X[f"wk1_{n}"]
            X[f"trend4_{n}"] = X[[f"wk{L}_{n}" for L in (1, 2, 3, 4)]].mean(1)
            X[f"trend_ratio_{n}"] = (c7 + 1) / (X[f"trend4_{n}"] + 1)
        # cohort-wide level of activity in the last 7 days (captures seasonality/holidays)
        o7 = own[own.ts >= t - 7 * DAY]
        glob = o7.groupby("kind").uid.nunique().reindex(ACT).fillna(0).to_numpy()
        for k in ACT:
            X[f"glob7_{KIND_NAMES[k]}"] = glob[k]
        return X

    def _global_level(self, past, t, uids):
        """Group TG: data-collection / activity level of the whole dataset right before t
        (events of every kind in the last 1 / 3 / 7 / 28 days, and short / long ratios).
        Identical for all users of a cutoff; lets the model adapt its base rate to the
        strongly non-stationary collection volume (e.g. days with no recorded pushes)."""
        own = past[past.kind < 6]
        X = pd.DataFrame(index=uids)
        cnt = {}
        for w in (1, 3, 7, 28):
            c = own[own.ts >= t - w * DAY].groupby("kind").size().reindex(ACT).fillna(0).to_numpy() / w
            cnt[w] = c
            for k in ACT:
                X[f"globx{w}d_{KIND_NAMES[k]}"] = np.log1p(c[k])
        for k in ACT:
            X[f"globx_ratio1_28_{KIND_NAMES[k]}"] = (cnt[1][k] + 1) / (cnt[28][k] + 1)
            X[f"globx_ratio7_28_{KIND_NAMES[k]}"] = (cnt[7][k] + 1) / (cnt[28][k] + 1)
        return X

    # ------------------------------------------------------------------------------
    def _rel_user(self, past, t, uids):
        ts = past.ts.to_numpy()
        f = []
        rel = past[past.kind >= 6]
        rts = rel.ts.to_numpy()
        for w in (7, 28, 90):
            f.append(_pivot_count(rel[rts >= t - w * DAY], uids, REL_KINDS, f"c{w}d"))
        last = rel.groupby(["uid", "kind"]).ts.max().unstack().reindex(index=uids, columns=REL_KINDS)
        last = ((t - last) / DAY).fillna(999)
        last.columns = [f"rec_{KIND_NAMES[k]}" for k in REL_KINDS]
        f.append(last)
        # point-in-time state of the user's PRs and issues
        pr = self.pr
        known = (pr.fs.to_numpy() < t) & (pr.fs.to_numpy() >= t - 90 * DAY)
        open_ = known & (pr.closed.to_numpy() >= t)        # closed_at unknown or after t
        o = pr[open_]
        f.append(o.groupby("author").size().reindex(uids).fillna(0).rename("own_open_pr90").to_frame())
        f.append(o[o.draft].groupby("author").size().reindex(uids).fillna(0).rename("own_open_draft_pr90").to_frame())
        iss = self.iss
        known = (iss.fs.to_numpy() < t) & (iss.fs.to_numpy() >= t - 90 * DAY)
        o = iss[known & (iss.closed.to_numpy() >= t)]
        f.append(o.groupby("author").size().reindex(uids).fillna(0).rename("own_open_issue90").to_frame())
        X = pd.concat(f, axis=1)
        # fraction of the user's PRs that were merged (point in time)
        X["merge_rate90"] = (X["c90d_own_pr_merged"] + 0.5) / (X["c90d_own_pr_merged"] + X["c90d_own_pr_closed"] + 1)
        return X

    # ------------------------------------------------------------------------------
    def _rel_repo(self, past, t, uids):
        ts = past.ts.to_numpy()
        act = past[past.kind < 6]
        a28 = act[act.ts.to_numpy() >= t - 28 * DAY]
        a28 = a28[a28.repo >= 0]
        a7 = a28[a28.ts.to_numpy() >= t - 7 * DAY]
        # repo-level statistics over ALL users observed in the data (last 7 / 28 days)
        r7 = a7.groupby(["repo", "kind"]).size().unstack(fill_value=0).reindex(columns=ACT, fill_value=0)
        r7.columns = [f"r7_{KIND_NAMES[k]}" for k in ACT]
        r7["r7_users"] = a7.groupby("repo").uid.nunique()
        r28_users = a28.groupby("repo").uid.nunique().rename("r28_users")
        # open PRs (point in time) per repo, authored by anyone: review demand
        pr = self.pr
        known = (pr.fs.to_numpy() < t) & (pr.fs.to_numpy() >= t - 90 * DAY)
        op = pr[known & (pr.closed.to_numpy() >= t)]
        r_open = op.groupby("repo").size().rename("r_open_pr")
        R = pd.concat([r7, r28_users, r_open], axis=1).fillna(0)
        R["r_is_org"] = self.repo_is_org[R.index.to_numpy()].astype(float)
        # user -> repo weights (share of the user's own activity in the last 28 days)
        ur = a28[a28.uid.isin(uids)].groupby(["uid", "repo"]).size().rename("n").reset_index()
        ur["w"] = ur.n / ur.groupby("uid").n.transform("sum")
        ur["own"] = (self.login[ur.uid.to_numpy()] == self.repo_owner[ur.repo.to_numpy()]).astype(float)
        J = ur.join(R, on="repo")
        cols = list(R.columns)
        wmean = J[cols].mul(J.w, axis=0).groupby(J.uid).sum()
        wmean.columns = [f"repo_wmean_{c}" for c in cols]
        mx = J.groupby("uid")[cols].max()
        mx.columns = [f"repo_max_{c}" for c in cols]
        g = J.groupby("uid")
        basic = pd.DataFrame({
            "n_repos28": g.size(),
            "own_repo_share28": (J.own * J.w).groupby(J.uid).sum(),
            "top_repo_share28": g.w.max(),
        })
        # 2-hop collaborators: other users active in the same repos in the last 28 days,
        # and how active those collaborators were in the last 7 days per activity type
        rep_users = a28[["repo", "uid"]].drop_duplicates()
        pairs = ur[["uid", "repo"]].merge(rep_users.rename(columns={"uid": "co"}), on="repo")
        pairs = pairs[pairs.uid != pairs.co][["uid", "co"]].drop_duplicates()
        co_act = a7.groupby(["uid", "kind"]).size().unstack(fill_value=0).reindex(columns=ACT, fill_value=0)
        co_act = (co_act > 0).astype(float)
        co_act.columns = [f"co7_frac_{KIND_NAMES[k]}" for k in ACT]
        P = pairs.join(co_act, on="co").fillna(0)
        co = P.groupby("uid")[list(co_act.columns)].mean()
        co["n_collab28"] = pairs.groupby("uid").size()
        # incoming review demand: open PRs authored by others in repos where the user
        # reviewed / merged during the last 90 days
        rv = past[(past.kind.isin([3, 4, 6])) & (ts >= t - 90 * DAY) & (past.repo.to_numpy() >= 0)]
        rr = rv[["uid", "repo"]].drop_duplicates()
        rr = rr[rr.uid.isin(uids)].join(r_open, on="repo").fillna(0)
        dem = rr.groupby("uid").r_open_pr.sum().rename("review_demand_open_pr")
        X = pd.concat([basic, wmean, mx, co, dem], axis=1).reindex(uids)
        X[["n_repos28", "n_collab28", "review_demand_open_pr"]] = \
            X[["n_repos28", "n_collab28", "review_demand_open_pr"]].fillna(0)
        return X.fillna(-1)


# ======================= Phase 2 feature groups ============================================
DEFAULT_BRANCHES = {"main", "master", "dev", "develop", "development", "gh-pages", "staging", "trunk"}


def _t2(self, past, t, uids):
    """T2 temporal v2: inter-active-day gaps / burstiness, very recent activity, active-week
    streaks, user-specific conditional weekly rates, push intensity."""
    own = past[(past.kind < 6) & (past.ts >= t - 90 * DAY)]
    X = pd.DataFrame(index=uids)
    d = pd.DataFrame({"uid": own.uid.to_numpy(), "d": (t - own.ts.to_numpy()) // DAY}).drop_duplicates()
    d = d[d.uid.isin(uids)].sort_values(["uid", "d"])
    d["gap"] = d.groupby("uid").d.diff()
    g = d.groupby("uid").gap
    X["t2_adays90"] = d.groupby("uid").size().reindex(uids).fillna(0)
    mu, sd = g.mean().reindex(uids), g.std().reindex(uids)
    X["t2_gap_mean"] = mu.fillna(-1)
    X["t2_gap_std"] = sd.fillna(-1)
    X["t2_gap_max"] = g.max().reindex(uids).fillna(-1)
    X["t2_burstiness"] = ((sd - mu) / (sd + mu)).reindex(uids).fillna(-2)
    second = d.groupby("uid").d.nth(1)
    second = pd.Series(second.to_numpy(), index=d.loc[second.index, "uid"].to_numpy()) if len(second) else pd.Series(dtype=float)
    X["t2_rec_2nd_day"] = second.reindex(uids).fillna(999)
    ts = own.ts.to_numpy()
    for h in (6, 12):
        X[f"t2_c{h}h_any"] = own[ts >= t - h * 3600].groupby("uid").size().reindex(uids).fillna(0)
    # weekly structure over 12 weeks (lag 0 = the week before t)
    o12 = own[ts >= t - 84 * DAY]
    wk = pd.DataFrame({"uid": o12.uid.to_numpy(), "kind": o12.kind.to_numpy(),
                       "wk": (t - o12.ts.to_numpy()) // (7 * DAY)}).drop_duplicates()
    wk = wk[wk.wk < 12]
    wany = wk[["uid", "wk"]].drop_duplicates()
    n_any = wany.groupby("uid").size().reindex(uids).fillna(0)
    X["t2_weeks12_any"] = n_any
    act = np.zeros((len(uids), 12), bool)
    pos = pd.Series(np.arange(len(uids)), index=uids)
    wa = wany[wany.uid.isin(uids)]
    act[pos.loc[wa.uid].to_numpy(), wa.wk.to_numpy().astype(int)] = True
    X["t2_streak_any"] = np.where(act.all(1), 12, np.argmin(act, axis=1))
    for k in range(6):
        n = KIND_NAMES[k]
        wk_k = wk[(wk.kind == k) & wk.uid.isin(uids)]
        nk = wk_k.groupby("uid").size().reindex(uids).fillna(0)
        X[f"t2_cond_{n}"] = nk / n_any.clip(lower=1)          # P(kind k | active week), user-specific
        a = np.zeros((len(uids), 12), bool)
        a[pos.loc[wk_k.uid].to_numpy(), wk_k.wk.to_numpy().astype(int)] = True
        X[f"t2_streak_{n}"] = np.where(a.all(1), 12, np.argmin(a, axis=1))
    p28 = own[(own.kind == 5) & (ts >= t - 28 * DAY)]
    pdays = pd.DataFrame({"uid": p28.uid.to_numpy(), "d": (t - p28.ts.to_numpy()) // DAY})
    X["t2_push_per_day28"] = (p28.groupby("uid").size() / pdays.drop_duplicates().groupby("uid").size()).reindex(uids).fillna(0)
    p7 = own[(own.kind == 5) & (ts >= t - 7 * DAY)]
    X["t2_push_repos7"] = p7.groupby("uid").repo.nunique().reindex(uids).fillna(0)
    return X


def _r3(self, t, uids):
    """R3 relational v2: (a) activity of OTHER users in the issue / PR threads the user took part
    in during the last 30 days, (b) feature-branch pushes and whether a PR exists for the branch."""
    X = pd.DataFrame(index=uids)
    us = set(uids.tolist())
    # ---- (a) issue / PR-conversation threads ------------------------------------------------
    ic = self.ic[(self.ic.ts < t) & (self.ic.ts >= t - 30 * DAY) & (self.ic.issue >= 0)]
    part = ic[ic.uid.isin(us)][["uid", "issue"]]
    ia = self.iss_all
    own_iss = np.where((ia.author.to_numpy() >= 0) & (ia.fs.to_numpy() < t) & (ia.fs.to_numpy() >= t - 30 * DAY))[0]
    part = pd.concat([part, pd.DataFrame({"uid": ia.author.to_numpy()[own_iss], "issue": own_iss})])
    part = part[part.uid.isin(us)].drop_duplicates()
    X["r3_threads30"] = part.groupby("uid").size().reindex(uids).fillna(0)
    for w in (1, 3, 7):
        oc = ic[ic.ts >= t - w * DAY][["uid", "issue"]].rename(columns={"uid": "other"})
        j = part.merge(oc, on="issue")
        j = j[j.uid != j.other]
        X[f"r3_thread_others_c{w}d"] = j.groupby("uid").size().reindex(uids).fillna(0)
        if w == 7:
            X["r3_thread_active7"] = j.groupby("uid").issue.nunique().reindex(uids).fillna(0)
    # ---- PR review threads (reviews + line comments; PR author known from the PR row) ---------
    pa = self.pr_act[(self.pr_act.avail < t) & (self.pr_act.ts < t) & (self.pr_act.ts >= t - 30 * DAY)]
    pm = self.pr_meta
    own_pr = np.where((pm.author.to_numpy() >= 0) & (pm.fs.to_numpy() < t) & (pm.fs.to_numpy() >= t - 30 * DAY))[0]
    ppart = pd.concat([pa[pa.uid.isin(us)][["uid", "pr"]],
                       pd.DataFrame({"uid": pm.author.to_numpy()[own_pr], "pr": own_pr})])
    ppart = ppart[ppart.uid.isin(us)].drop_duplicates()
    X["r3_prthreads30"] = ppart.groupby("uid").size().reindex(uids).fillna(0)
    for w in (3, 7):
        oa = pa[pa.ts >= t - w * DAY][["uid", "pr"]].rename(columns={"uid": "other"})
        j = ppart.merge(oa, on="pr")
        j = j[j.uid != j.other]
        X[f"r3_prthread_others_c{w}d"] = j.groupby("uid").size().reindex(uids).fillna(0)
    # ---- (b) branches ------------------------------------------------------------------------
    pu = self.push[(self.push.ts < t) & (self.push.ts >= t - 28 * DAY) & self.push.uid.isin(us)]
    nd = ~pu.branch.isin(DEFAULT_BRANCHES)
    X["r3_default_push_share28"] = (~nd).groupby(pu.uid).mean().reindex(uids).fillna(-1)
    known = self.pr_branch[self.pr_branch.fs < t]
    keys = set(zip(known.repo.to_numpy(), known.branch.to_numpy()))
    for w in (3, 7):
        b = pu[nd & (pu.ts >= t - w * DAY)][["uid", "repo", "branch"]].drop_duplicates()
        has = np.fromiter((k in keys for k in zip(b.repo.to_numpy(), b.branch.to_numpy())), bool, len(b))
        X[f"r3_feat_branches{w}d"] = b.groupby("uid").size().reindex(uids).fillna(0)
        X[f"r3_branches_no_pr{w}d"] = b[~has].groupby("uid").size().reindex(uids).fillna(0)
        X[f"r3_branches_with_pr{w}d"] = b[has].groupby("uid").size().reindex(uids).fillna(0)
    return X




def outage_intervals(past, t, back_days=100):
    """Collection outages of each activity kind, detected point-in-time from rows known before t.
    A complete UTC day is an outage of kind k if its global count is < OUTAGE_REL x the median of
    the last 28 earlier non-outage days (common.outage_flags); the partial day ending at t is judged by its hourly rate
    (only if >= 6 h old).  Returns {kind: [(start, end), ...]} in seconds, clipped to < t.
    Outage days are never back-filled by later releases, so features that count them as
    "no activity" are biased (e.g. 10-15..10-17: no pushes at all)."""
    own = past[(past.kind < 6) & (past.ts >= t - (back_days + 29) * DAY)]
    d0 = (t // DAY - back_days - 29) * DAY
    day = (own.ts.to_numpy() - d0) // DAY
    n_day = (t - d0) // DAY + 1
    cnt = np.zeros((6, n_day))
    np.add.at(cnt, (own.kind.to_numpy(), day), 1)
    part_h = (t - (d0 + (n_day - 1) * DAY)) / 3600       # hours of the partial last day
    out = {}
    for k in ACT:
        flags, _ = outage_flags(cnt[k, :n_day - 1])            # complete days
        iv = [(d0 + i * DAY, d0 + (i + 1) * DAY) for i in np.flatnonzero(flags)]
        r = outage_flags(cnt[k])[1][-1]                       # reference for the partial day
        if part_h >= 6 and r > 0 and cnt[k, -1] < OUTAGE_REL * r * part_h / 24:
            iv.append((d0 + (n_day - 1) * DAY, t))
        out[k] = iv
    return out


def _fill_prevday(own, a, b):
    """Repeat the events of the day before the hole [a, b) on every day of the hole."""
    src = own[(own.ts >= a - DAY) & (own.ts < a)]
    parts = []
    for j in range(int(-(-(b - a) // DAY))):
        f = src.assign(ts=src.ts + (j + 1) * DAY)
        parts.append(f[f.ts < b])
    return parts


def _fill_week(own, a, b):
    """Spread each user's mean daily count of the 7 days before the hole [a, b) over the hole days.
    A user with n events in those 7 days gets floor((j+1) n/7 + f) - floor(j n/7 + f) events on hole
    day j (so n/7 per day on average; f in [0, 1) is a fixed per-user phase, so that users with few
    events do not all land on the same day), taken in time order from the user's own events of that
    week (cycled), each keeping its repo and time of day."""
    src = own[(own.ts >= a - 7 * DAY) & (own.ts < a)].sort_values(["uid", "ts"], kind="stable")
    if not len(src):
        return []
    n = src.groupby("uid").ts.transform("size").to_numpy()
    r = src.groupby("uid").cumcount().to_numpy()
    tod = src.ts.to_numpy() % DAY
    f0 = (src.uid.to_numpy() * 0.6180339887) % 1.0       # deterministic per-user phase
    parts = []
    for j in range(int(-(-(b - a) // DAY))):
        lo = np.floor(j * n / 7 + f0).astype(int)
        hi = np.floor((j + 1) * n / 7 + f0).astype(int)
        m = ((r - lo) % n) < (hi - lo)
        f = src[m].assign(ts=a + j * DAY + tod[m])
        parts.append(f[f.ts < b])
    return parts


def impute_outages(past, t, method="copy"):
    """Fill every collection outage of an activity kind (outage_intervals, point-in-time); only rows
    known before t are used and the filled rows get avail = ts.  Methods:
      copy    - that kind's events of the same span whole weeks earlier, shifted forward (a hole of
                length L is filled from [a - s, b - s), s = L rounded up to whole weeks)
      week    - each user's mean daily count of the 7 days before the hole (_fill_week)
      prevday - the events of the day before the hole, repeated on each hole day (_fill_prevday)"""
    iv = outage_intervals(past, t)
    fill = []
    for k, ivs in iv.items():
        holes = []
        for a, b in ivs:                       # merge adjacent outage days
            if holes and a <= holes[-1][1]:
                holes[-1][1] = b
            else:
                holes.append([a, b])
        own = past[past.kind == k]
        for a, b in holes:                     # in time order: a later hole may copy an earlier fill
            if method == "copy":
                sh = -(-(b - a) // (7 * DAY)) * 7 * DAY
                src = own[(own.ts >= a - sh) & (own.ts < b - sh)]
                parts = [src.assign(ts=src.ts + sh)] if len(src) else []
            elif method == "week":
                parts = _fill_week(own, a, b)
            elif method == "prevday":
                parts = _fill_prevday(own, a, b)
            else:
                raise ValueError(f"unknown imputation method {method}")
            parts = [f.assign(avail=f.ts) for f in parts if len(f)]
            if parts:
                fill.extend(parts)
                own = pd.concat([own, *parts], ignore_index=True)
    return pd.concat([past, *fill], ignore_index=True) if fill else past


def _overlap(iv, a, b):
    """Seconds of the intervals iv inside [a, b); a may be an array."""
    a = np.asarray(a, dtype="float64")
    o = np.zeros_like(a)
    for s0, e0 in iv:
        o += np.clip(np.minimum(e0, b) - np.maximum(s0, a), 0, None)
    return o


def _decay_mass(a, b, t, hl):
    """Integral of 2^(-(t-s)/hl) ds over [a, b)."""
    h = hl * DAY
    return h / np.log(2) * (np.exp2(-(t - b) / h) - np.exp2(-(t - a) / h))


def _outage(self, past, t, uids):
    """Group O: outage-corrected versions of the per-activity temporal features.
    Counts / active days are divided by the observed fraction of their window (NaN if < 25 %
    of it was observed), decayed counts by the observed decay mass, recency skips outage time,
    weekly indicators are NaN for weeks with < 50 % observed time.  Column o_<name> corrects
    <name>; o_obs<w>_<kind> is the observed fraction (same for all users of a cutoff)."""
    iv = outage_intervals(past, t)
    own = past[(past.kind < 6) & (past.ts >= t - 90 * DAY)]
    X = {}
    for k in ACT:
        n = KIND_NAMES[k]
        ok = own[own.kind.to_numpy() == k]
        ts = ok.ts.to_numpy()
        for w in (1, 3, 7, 14, 28, 56, 90):
            frac = float(1 - _overlap(iv[k], t - w * DAY, t) / (w * DAY))
            if w in (1, 7, 28):
                X[f"o_obs{w}_{n}"] = frac
            c = ok[ts >= t - w * DAY].groupby("uid").size().reindex(uids).fillna(0)
            X[f"o_c{w}d_{n}"] = c / frac if frac >= 0.25 else np.nan
        frac28 = float(1 - _overlap(iv[k], t - 28 * DAY, t) / (28 * DAY))
        o28 = ok[ts >= t - 28 * DAY]
        ad = pd.Series((t - o28.ts.to_numpy()) // DAY, index=o28.uid.to_numpy())
        ad = ad.groupby(level=0).nunique().reindex(uids).fillna(0)
        X[f"o_adays28_{n}"] = ad / frac28 if frac28 >= 0.25 else np.nan
        for hl in (3, 14):
            a = t - 8 * hl * DAY
            tot = _decay_mass(a, t, t, hl)
            lost = sum(_decay_mass(max(s0, a), min(e0, t), t, hl) for s0, e0 in iv[k] if e0 > a)
            o = ok[ts >= a]
            e = pd.Series(np.exp2(-(t - o.ts.to_numpy()) / (hl * DAY)), index=o.uid.to_numpy())
            X[f"o_ewm{hl}_{n}"] = e.groupby(level=0).sum().reindex(uids).fillna(0) / max(1 - lost / tot, 0.25)
        last = ok.groupby("uid").ts.max().reindex(uids)
        m = last.notna().to_numpy()
        rec = np.full(len(uids), 999.0)
        lv = last.to_numpy()[m]
        rec[m] = (t - lv - _overlap(iv[k], lv, t)) / DAY
        X[f"o_rec_{n}"] = rec
        wk = (t - ts) // (7 * DAY)
        for L in range(8):
            fr = float(1 - _overlap(iv[k], t - (L + 1) * 7 * DAY, t - L * 7 * DAY) / (7 * DAY))
            hit = pd.Series(1.0, index=np.unique(ok.uid.to_numpy()[wk == L])).reindex(uids).fillna(0)
            X[f"o_wk{L + 1}_{n}"] = hit if fr >= 0.5 else np.nan
    return pd.DataFrame(X, index=uids)


FeatureBuilder._temporal_v2 = _t2
FeatureBuilder._rel_threads_branches = _r3
FeatureBuilder._outage = _outage


def _r4(self, past, t, uids):
    """Group R4 repo traffic (Phase 3, from results/analysis/ANALYSIS.md): PR / issue flow in the
    repos the user worked in during the last 28 days, summed and activity-weighted over those
    repos.  PRs / issues first seen within 90 days, state reconstructed at t:
    open PRs (and the share not authored by known non-cohort users), PRs opened / merged in the last 7 days, open PRs
    older than 7 days, open PRs with review activity or conversation comments in the last 3 days,
    issues opened in the last 7 days, open issues commented on in the last 3 days, and open PRs
    by authors whose PRs the user reviewed in the last 90 days."""
    own = past[(past.kind < 6) & (past.ts >= t - 28 * DAY) & (past.repo >= 0)]
    own = own[own.uid.isin(uids)]
    ur = own.groupby(["uid", "repo"]).size().rename("n").reset_index()
    ur["w"] = ur.n / ur.groupby("uid").n.transform("sum")
    pr = self.pr
    pid = np.arange(len(pr))
    fs, closed, merged = pr.fs.to_numpy(), pr.closed.to_numpy(), pr.merged.to_numpy()
    known = (fs < t) & (fs >= t - 90 * DAY)
    is_open = known & (closed >= t)
    # PRs with review / line-comment activity or a conversation comment in the last 3 days
    pa = self.pr_act[(self.pr_act.avail < t) & (self.pr_act.ts < t) & (self.pr_act.ts >= t - 3 * DAY)]
    ic3 = self.ic[(self.ic.ts < t) & (self.ic.ts >= t - 3 * DAY) & (self.ic.issue >= 0)]
    conv_pr = self.iss_all.pr.to_numpy()[ic3.issue.to_numpy()]
    hot = np.zeros(len(pr), bool)
    hot[pa.pr.to_numpy()] = True
    hot[conv_pr[conv_pr >= 0]] = True
    a = pr.author.to_numpy()
    by_noncohort = (a >= 0) & ~self.cohort[np.maximum(a, 0)]   # unknown authors count as cohort
    P = pd.DataFrame({"repo": pr.repo.to_numpy(), "open_pr": is_open, "open_pr_noncohort": is_open & by_noncohort,
                      "new_pr7": known & (fs >= t - 7 * DAY), "merged7": (fs < t) & (merged < t) & (merged >= t - 7 * DAY),
                      "open_pr_stale7": is_open & (fs < t - 7 * DAY), "open_pr_hot3": is_open & hot})
    P = P[(P.repo >= 0) & (P.open_pr | P.new_pr7 | P.merged7)]
    iss = self.iss
    ifs = iss.fs.to_numpy()
    iknown = (ifs < t) & (ifs >= t - 90 * DAY)
    iopen = iknown & (iss.closed.to_numpy() >= t)
    iid_hot = np.unique(ic3.issue.to_numpy())
    I = pd.DataFrame({"repo": iss.repo.to_numpy(), "new_issue7": iknown & (ifs >= t - 7 * DAY),
                      "open_issue_hot3": iopen & np.isin(iss.iid.to_numpy(), iid_hot)})
    I = I[(I.repo >= 0) & (I.new_issue7 | I.open_issue_hot3)]
    R = pd.concat([P.groupby("repo").sum(), I.groupby("repo").sum()], axis=1).fillna(0)
    J = ur.join(R, on="repo").fillna(0)
    X = {}
    for c in ("open_pr", "new_pr7", "merged7", "open_pr_stale7", "open_pr_hot3", "new_issue7", "open_issue_hot3"):
        X[f"r4_sum_{c}"] = J.groupby("uid")[c].sum().reindex(uids).fillna(0)
        if c in ("new_pr7", "merged7", "open_pr_hot3"):
            X[f"r4_wmean_{c}"] = (J[c] * J.w).groupby(J.uid).sum().reindex(uids).fillna(0)
    sop = J.groupby("uid").open_pr.sum().reindex(uids)
    X["r4_open_pr_cohort_share"] = (1 - J.groupby("uid").open_pr_noncohort.sum().reindex(uids) / sop.where(sop > 0)).fillna(-1)
    # open PRs by authors whose PRs the user reviewed in the last 90 days
    rv = self.pr_act[(self.pr_act.avail < t) & (self.pr_act.ts < t) & (self.pr_act.ts >= t - 90 * DAY)]
    rv = rv[rv.uid.isin(uids)]
    # the PR author is known only once the PR row is (first_seen_at < t)
    rel = pd.DataFrame({"uid": rv.uid.to_numpy(), "author": np.where(fs[rv.pr.to_numpy()] < t, a[rv.pr.to_numpy()], -1)})
    rel = rel[(rel.author >= 0) & (rel.author != rel.uid)].drop_duplicates()
    opa = pd.Series(a[is_open & (a >= 0)]).value_counts().rename("k")
    X["r4_open_pr_by_reviewed_authors"] = rel.join(opa, on="author").fillna(0).groupby("uid").k.sum().reindex(uids).fillna(0)
    return pd.DataFrame(X, index=uids)


FeatureBuilder._repo_traffic = _r4


def add_regime_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Group N: each key activity feature divided by its mean over all sample users of the same
    cutoff (point-in-time: uses only features of other users at the same t).  Makes the
    features comparable across collection regimes."""
    base = [f"{p}_{c}" for p in ("c7d", "c28d", "ewm3", "ewm14", "wk1", "adays28") for c in TARGETS]
    base = [c for c in base if c in df]
    m = df.groupby("cutoff")[base].transform("mean")
    N = (df[base] / m.where(m > 0)).fillna(0).astype("float32")
    N.columns = [f"nrm_{c}" for c in base]
    return pd.concat([df, N], axis=1)


def feature_groups(columns) -> dict[str, list[str]]:
    """Map feature names back to their groups (for ablations)."""
    T0 = [c for c in columns if any(c.startswith(f"c{w}d_") for w in (7, 28, 90))
          and c.split("_", 1)[1] in TARGETS]
    R1_prefix = tuple(f"_{KIND_NAMES[k]}" for k in REL_KINDS)
    R1 = [c for c in columns if c.endswith(R1_prefix) or c.startswith(("own_open", "merge_rate"))]
    R2 = [c for c in columns if c.startswith(("repo_", "n_repos", "own_repo", "top_repo", "co7_",
                                              "n_collab", "review_demand"))]
    TG = [c for c in columns if c.startswith("globx")]
    T2 = [c for c in columns if c.startswith("t2_")]
    R3 = [c for c in columns if c.startswith("r3_")]
    N = [c for c in columns if c.startswith("nrm_")]
    O = [c for c in columns if c.startswith("o_")]
    R4 = [c for c in columns if c.startswith("r4_")]
    T1 = [c for c in columns if c not in set(T0) | set(R1) | set(R2) | set(TG) | set(T2) | set(R3) | set(N) | set(O) | set(R4)]
    return {"T0": T0, "T1": T1, "R1": R1, "R2": R2, "TG": TG, "T2": T2, "R3": R3, "N": N, "O": O, "R4": R4}
