"""Shared constants, data loading and the unified point-in-time event log.

Every time in this project is UTC (the parquet files are UTC without tz).
Taiwan time = UTC+8; the official target window 2026-11-22 00:00 -> 11-29 00:00 Taiwan
is therefore 2026-11-21 16:00 -> 2026-11-28 16:00 UTC (course update of 11-07).

The tables are built by the course's assignment2/scripts/preprocess.py from GH Archive hour
files that we download ourselves (download_gharchive.sh); default location data/db.

Point-in-time rule used everywhere: an event row may be used at cutoff t only if its
*availability time* (`avail`) is < t.  `avail` is the row-time column of the source table
(first_seen_at for PRs / issues / reviews, created_at for comments / pushes) or, for the
"known only later" columns (merged_at, closed_at, dismissed_at), the value itself.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.environ.get("ASG2_DB", ROOT / "data" / "db"))
ART = ROOT / "artifacts"
RES = ROOT / "results"
FIG = ROOT / "figures"
for _d in (ART, RES, FIG):
    _d.mkdir(exist_ok=True)

TARGETS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]

# Event kinds.  0..5 are the six target activities (performed BY the user).
K_OPEN_PR, K_OPEN_ISSUE, K_ISSUE_COMMENT, K_PR_REVIEW, K_PR_REVIEW_COMMENT, K_PUSH = range(6)
# Relational / "incoming" kinds: things that happen TO the user's objects or extra roles.
K_MERGED_BY = 6          # user merged somebody's PR (maintainer role)        avail = merged_at
K_IN_COMMENT = 7         # someone else commented on the user's issue/PR      avail = comment time
K_IN_REVIEW = 8          # someone else reviewed the user's PR               avail = review first_seen
K_IN_REVIEW_COMMENT = 9  # someone else line-commented on the user's PR      avail = comment time
K_OWN_PR_MERGED = 10     # user's PR got merged                               avail = merged_at
K_OWN_PR_CLOSED = 11     # user's PR closed without merge                     avail = closed_at
K_OWN_ISSUE_CLOSED = 12  # user's issue got closed                           avail = closed_at
KIND_NAMES = TARGETS + ["merged_by", "in_comment", "in_review", "in_review_comment",
                        "own_pr_merged", "own_pr_closed", "own_issue_closed"]

DAY = 86400
# Our prediction cutoff is 4 hours before the target window starts: GH Archive hours up to
# 11-21 12:00 UTC (20:00 Taiwan) are used; the window starts 11-21 16:00 UTC.  The 4 h leave
# time to download, preprocess and predict before the 11-21 23:55 Taiwan deadline.
GAP_SEC = 4 * 3600
FINAL_CUTOFF = pd.Timestamp("2026-11-21 12:00:00")   # last data timestamp used for the submission
FINAL_WINDOW_START = pd.Timestamp("2026-11-21 16:00:00")


def ts2s(x) -> np.ndarray:
    """datetime64 Series -> int64 seconds since epoch (NaT -> INT64 max)."""
    v = pd.to_datetime(x).to_numpy("datetime64[s]").astype("int64")
    nat = pd.isna(x).to_numpy() if hasattr(x, "to_numpy") else np.isnat(x)
    v = v.copy()
    v[nat] = np.iinfo(np.int64).max
    return v


def s(ts) -> int:
    return int(pd.Timestamp(ts).value // 10**9)


def load_tables(db: Path = DB) -> dict[str, pd.DataFrame]:
    names = ["Users", "Repositories", "PullRequests", "Issues", "IssueComments",
             "PRReviews", "PRReviewComments", "Pushes"]
    return {n: pd.read_parquet(db / f"{n}.parquet") for n in names}


def _ev(uid, repo, kind, ts, avail) -> pd.DataFrame:
    uid = pd.array(uid, dtype="Int64")
    df = pd.DataFrame({"uid": uid, "repo": pd.array(repo, dtype="Int64"),
                       "kind": np.int8(kind), "ts": ts, "avail": avail})
    df = df[df.uid.notna() & (df.ts < np.iinfo(np.int64).max)]
    df["uid"] = df.uid.astype("int64")
    df["repo"] = df.repo.fillna(-1).astype("int64")
    return df


def build_events(T: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Unified event log over ALL users (cohort and non-cohort).

    columns: uid, repo, kind, ts (action time, s), avail (time the row is known, s).
    """
    pr, iss = T["PullRequests"], T["Issues"]
    ic, rv, rc, pu = T["IssueComments"], T["PRReviews"], T["PRReviewComments"], T["Pushes"]
    pr_fs = ts2s(pr.first_seen_at)
    out = []
    # --- six target activities ------------------------------------------------------
    m = pr.created_at.notna().to_numpy()
    out.append(_ev(pr.author_id[m], pr.repo_id[m], K_OPEN_PR, ts2s(pr.created_at[m]), pr_fs[m]))
    m = (iss.is_pull_request == False).fillna(False).to_numpy()
    out.append(_ev(iss.author_id[m], iss.repo_id[m], K_OPEN_ISSUE, ts2s(iss.created_at[m]),
                   ts2s(iss.first_seen_at[m])))
    t = ts2s(ic.created_at)
    out.append(_ev(ic.author_id, ic.repo_id, K_ISSUE_COMMENT, t, t))
    out.append(_ev(rv.reviewer_id, rv.repo_id, K_PR_REVIEW, ts2s(rv.submitted_at), ts2s(rv.first_seen_at)))
    t = ts2s(rc.created_at)
    out.append(_ev(rc.author_id, rc.repo_id, K_PR_REVIEW_COMMENT, t, t))
    t = ts2s(pu.created_at)
    out.append(_ev(pu.user_id, pu.repo_id, K_PUSH, t, t))
    # --- maintainer role: merged_by is "known only later" -> avail = merged_at --------
    mt = ts2s(pr.merged_at)
    m = pr.merged_by_id.notna().to_numpy() & (mt < np.iinfo(np.int64).max)
    out.append(_ev(pr.merged_by_id[m], pr.repo_id[m], K_MERGED_BY, mt[m], np.maximum(mt[m], pr_fs[m])))
    # --- outcomes of the user's own PRs / issues (known only later) ------------------
    ct = ts2s(pr.closed_at)
    m = pr.author_id.notna().to_numpy() & (mt < np.iinfo(np.int64).max)
    out.append(_ev(pr.author_id[m], pr.repo_id[m], K_OWN_PR_MERGED, mt[m], np.maximum(mt[m], pr_fs[m])))
    m = pr.author_id.notna().to_numpy() & (ct < np.iinfo(np.int64).max) & (mt == np.iinfo(np.int64).max)
    out.append(_ev(pr.author_id[m], pr.repo_id[m], K_OWN_PR_CLOSED, ct[m], np.maximum(ct[m], pr_fs[m])))
    it = ts2s(iss.closed_at)
    m = (iss.is_pull_request == False).fillna(False).to_numpy() & (it < np.iinfo(np.int64).max)
    i_fs = ts2s(iss.first_seen_at)
    out.append(_ev(iss.author_id[m], iss.repo_id[m], K_OWN_ISSUE_CLOSED, it[m], np.maximum(it[m], i_fs[m])))
    # --- incoming interactions on the user's objects (cross-table joins) -------------
    iss_author = iss.author_id.to_numpy()
    a = iss_author[ic.issue_id.fillna(0).astype(int).to_numpy()]
    a = pd.array(a, dtype="Int64")
    keep = ic.issue_id.notna().to_numpy() & pd.notna(a) & (np.asarray(a, dtype=float) != ic.author_id.to_numpy(dtype=float, na_value=np.nan))
    t = ts2s(ic.created_at)
    out.append(_ev(a[keep], ic.repo_id[keep], K_IN_COMMENT, t[keep], t[keep]))
    pr_author = pr.author_id.to_numpy()
    for tab, who, kind in [(rv, "reviewer_id", K_IN_REVIEW), (rc, "author_id", K_IN_REVIEW_COMMENT)]:
        pid = tab.pr_id.fillna(0).astype(int).to_numpy()
        a = pd.array(pr_author[pid], dtype="Int64")
        keep = tab.pr_id.notna().to_numpy() & pd.notna(a) & (np.asarray(a, dtype=float) != tab[who].to_numpy(dtype=float, na_value=np.nan))
        tt = ts2s(tab.submitted_at if kind == K_IN_REVIEW else tab.created_at)
        av = ts2s(tab.first_seen_at if kind == K_IN_REVIEW else tab.created_at)
        # the PR author may only become known when the PR row appears
        av = np.maximum(av, pr_fs[pid])
        out.append(_ev(a[keep], tab.repo_id[keep], kind, tt[keep], av[keep]))
    ev = pd.concat(out, ignore_index=True)
    ev["kind"] = ev.kind.astype("int8")
    return ev.sort_values("avail", kind="stable").reset_index(drop=True)


LABEL_SLACK = 5 * DAY  # official truth of the 9/27 example used the release ~5 days after the window


def label_matrix(ev: pd.DataFrame, uids: np.ndarray, start: int, end: int,
                 slack: int = LABEL_SLACK) -> np.ndarray:
    """Binary labels (n_users x 6) for the window [start, end).

    An activity counts if its action time is in the window and its row was known within
    `slack` after the window end.  With slack = 5 days this reproduces the official
    sample_truth.csv of the 9/27 example exactly (0 mismatches, 4785 users x 6 targets).
    """
    m = (ev.ts.to_numpy() >= start) & (ev.ts.to_numpy() < end) & (ev.avail.to_numpy() < end + slack) \
        & (ev.kind.to_numpy() < 6)
    sub = ev[m]
    Y = np.zeros((len(uids), 6), dtype=np.int8)
    pos = pd.Series(np.arange(len(uids)), index=uids)
    for k in range(6):
        u = sub.uid[sub.kind == k].unique()
        idx = pos.reindex(u).dropna().astype(int).to_numpy()
        Y[idx, k] = 1
    return Y


SEL_LAG = 17 * DAY   # test list of 11-07: selected at 11-04 12:00 UTC for the 11-21 12:00 cutoff


def eligible_users(ev: pd.DataFrame, cohort: np.ndarray, t: int, min_records: int = 3,
                   days: int = 90) -> np.ndarray:
    """Official selection rule for a prediction cutoff t: cohort users with >= 3 activity records
    (action time) in the 90 days before t - 17 days.  Reproduces the 4,465 users of the 11-07
    test_input.csv exactly (0 missing, 0 extra).  Only rows known before t are counted."""
    sel = t - SEL_LAG
    ts, a = ev.ts.to_numpy(), ev.avail.to_numpy()
    m = (ev.kind.to_numpy() < 6) & (ts < sel) & (ts >= sel - days * DAY) & (a < t)
    cnt = ev.uid[m].value_counts()
    cnt = cnt.reindex(cohort).fillna(0)
    return cnt.index[cnt >= min_records].to_numpy()


OUTAGE_REL = 0.1   # a day is an outage of an activity if its count < 10 % of the reference


def outage_flags(cnt, rel: float = OUTAGE_REL, win: int = 28, span: int = 56, min_ref: int = 7):
    """Outage days of one daily count series (oldest first).  Day i is an outage if
    cnt[i] < rel x ref[i], ref[i] = median of the most recent `win` earlier NON-outage days within
    the last `span` days (needs >= min_ref of them, else the day is not judged).  Excluding earlier
    outage days keeps a long hole detected; the span lets a level change of ~7 weeks become the new
    normal.  Returns (flags, ref); ref is NaN where the day could not be judged."""
    cnt = np.asarray(cnt, dtype="float64")
    if os.environ.get("ASG2_OUTAGE_RULE") == "rolling":   # rule before addendum 6, only for comparison
        r = pd.Series(cnt).rolling(win, min_periods=min_ref).median().shift(1).to_numpy()
        return np.nan_to_num(cnt < rel * r, nan=False).astype(bool) & (r > 0), r
    out = np.zeros(len(cnt), dtype=bool)
    ref = np.full(len(cnt), np.nan)
    gi: list[int] = []        # indices of the non-outage days so far
    lo = 0                    # first entry of gi inside the span
    for i, x in enumerate(cnt):
        while lo < len(gi) and gi[lo] < i - span:
            lo += 1
        g = cnt[gi[max(lo, len(gi) - win):]]
        if len(g) >= min_ref:
            ref[i] = np.median(g)
            if ref[i] > 0 and x < rel * ref[i]:
                out[i] = True
                continue
        gi.append(i)
    return out, ref
