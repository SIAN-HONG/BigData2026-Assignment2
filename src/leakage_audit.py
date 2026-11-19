"""Leakage audit: features computed at cutoff t from the FULL database must be identical to
features computed from a database physically truncated at t, i.e. what a release made at
time t would have contained:
  * every table filtered by its row-time column < t
  * "known only later" columns nulled when their value is >= t
    (merged_at + merged_by_id, PR closed_at, Issues.closed_at + state_reason, dismissed_at)
  * FKs into removed rows set to null (rows are not renumbered, so ids stay comparable)
Run: python src/leakage_audit.py [--impute[=week|prevday]] [cutoff ...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import DAY, build_events, eligible_users, load_tables, s
from features import FeatureBuilder

# audit the outage-imputed features (features.impute_outages): --impute (copy) or --impute=week|prevday
IMPUTE = next((a.split("=", 1)[1] if "=" in a else "copy" for a in sys.argv if a.startswith("--impute")), False)

GROUPS = ("T0", "T1", "R1", "R2", "TG", "T2", "R3", "O", "R4")

ROWTIME = {"Users": "first_seen_at", "Repositories": "first_seen_at", "PullRequests": "first_seen_at",
           "Issues": "first_seen_at", "IssueComments": "created_at", "PRReviews": "first_seen_at",
           "PRReviewComments": "created_at", "Pushes": "created_at"}


def truncate(T, t):
    tt = pd.Timestamp(t, unit="s")
    R = {}
    for n, d in T.items():
        d = d.copy()
        drop = d[ROWTIME[n]] >= tt
        # keep row positions (ids are row numbers); blank out rows not yet known
        d.loc[drop, d.columns.difference([ROWTIME[n]])] = pd.NA
        d.loc[drop, ROWTIME[n]] = pd.NaT
        R[n] = d
    pr = R["PullRequests"]
    late = pr.merged_at >= tt
    pr.loc[late, ["merged_at", "merged_by_id"]] = [pd.NaT, pd.NA]
    pr.loc[pr.closed_at >= tt, "closed_at"] = pd.NaT
    iss = R["Issues"]
    late = iss.closed_at >= tt
    iss.loc[late, "closed_at"] = pd.NaT
    iss.loc[late, "state_reason"] = pd.NA
    rv = R["PRReviews"]
    rv.loc[rv.dismissed_at >= tt, "dismissed_at"] = pd.NaT
    return R


def audit(T, ev, t):
    u = T["Users"]
    uids = eligible_users(ev, u.user_id[u.is_cohort].to_numpy(), t)
    full = FeatureBuilder(T, ev, impute=IMPUTE).build(t, uids, groups=GROUPS)
    Tt = truncate(T, t)
    evt = build_events(Tt)
    assert (evt.avail < t).all(), "truncated event log contains future rows"
    trunc = FeatureBuilder(Tt, evt, impute=IMPUTE).build(t, uids, groups=GROUPS)
    diff = (full - trunc).abs()
    bad = diff.max()[diff.max() > 1e-4]
    nan_bad = (full.isna() != trunc.isna()).sum()       # group O uses NaN for unobserved windows
    bad = pd.concat([bad, nan_bad[nan_bad > 0].astype(float)])
    print(f"cutoff {pd.Timestamp(t, unit='s')}: {len(uids)} users x {full.shape[1]} features, "
          f"columns differing: {len(bad)}")
    if len(bad):
        print(bad.sort_values(ascending=False).head(20))
    return len(bad)


if __name__ == "__main__":
    T = load_tables()
    ev = build_events(T)
    cuts = [a for a in sys.argv[1:] if not a.startswith("--")] or ["2026-05-16 12:00", "2026-09-05 12:00", "2026-11-07 12:00"]   # Saturdays; 09-05 and 11-07 follow outages
    nbad = sum(audit(T, ev, s(c)) for c in cuts)
    print("LEAKAGE AUDIT", "PASSED" if nbad == 0 else "FAILED")
