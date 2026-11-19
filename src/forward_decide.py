"""Apply the pre-registered forward-check rule of docs/FORWARD_CHECK.md (with addenda 3 and 4).

  python src/forward_decide.py --list-weeks --data-end "2026-11-10 12:00"   # fresh weeks to backtest
  python src/forward_decide.py --fwd ../fwd/results --data-end "2026-11-10 12:00"

Old weeks come from artifacts/folds_reh_*_imp.pkl (backtests on the 10-28 release, outage imputation),
fresh weeks from the fwd/ backtests (fwd_v1_imp, fwd_p3_imp, fwd_p4_imp).  Writes results/forward_check.{json,md}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pickle

import numpy as np
import pandas as pd

from common import ART, DAY, GAP_SEC, RES, ROOT, TARGETS, build_events, load_tables, outage_flags
from modeling import EXCLUDE_WEEKS, joint_plugin_predict, official_score

FRESH = ["2026-10-14", "2026-10-21", "2026-10-28"]
# Addendum 4 (2026-11-01): all versions with outage imputation (dataset_imp); old weeks from the
# backtests on the 10-28 release (reh/, folds copied to artifacts/folds_reh_*.pkl).
OLD_BASE = ["reh_v1_imp"]
FWD_BASE = "fwd_v1_imp"
BASE_CONFIG = "configs/final_config_v1_baseline_imp.json"
SLACK = 5 * DAY


def status(week, data_end):
    """'complete' | 'provisional' (label window over, slack not) | 'pending'."""
    v = pd.Timestamp(f"{week} 12:00")
    end = v + pd.Timedelta(seconds=GAP_SEC + 7 * DAY)
    if data_end >= end + pd.Timedelta(seconds=SLACK):
        return "complete"
    return "provisional" if data_end >= end else "pending"


# outage day: count < 10 % of the median of the last 28 earlier non-outage days (common.outage_flags)
DROP = {}          # week -> activities left out of the score (filled in main)


def outage_targets(weeks, data_end):
    """Activities with an outage day overlapping the label window [V+4h, V+4h+7d) of each week."""
    ev = build_events(load_tables())
    ev = ev[ev.kind < 6]
    day = pd.to_datetime(ev.ts, unit="s").dt.floor("D")
    c = ev.groupby([day, ev.kind]).size().unstack(fill_value=0)
    c = c.reindex(pd.date_range(c.index.min(), data_end.floor("D") - pd.Timedelta(days=1)), fill_value=0)
    c.columns = TARGETS
    out = c.apply(lambda col: pd.Series(outage_flags(col.to_numpy())[0], index=col.index))   # common.outage_flags
    res = {}
    for w in weeks:
        s = pd.Timestamp(f"{w} 12:00") + pd.Timedelta(seconds=GAP_SEC)
        days = out.index[(out.index >= s.floor("D")) & (out.index < s + pd.Timedelta(days=7))]
        res[w] = [k for k in TARGETS if out.loc[days, k].any()]
    return res


def scores(files, root):
    """Week scores (points) recomputed from the fold pickles, leaving out the DROP activities."""
    art = ART if Path(root) == RES else Path(root).parent / "artifacts"
    out = {}
    for f in files:
        path = art / f"folds_{f}.pkl"
        if not path.exists():
            raise FileNotFoundError(path)
        for v, P, Y, _ in pickle.load(open(path, "rb")):
            w = str(v.date())
            keep = [j for j, k in enumerate(TARGETS) if k not in DROP.get(w, [])]
            d = joint_plugin_predict(P)
            out[w] = official_score(d[:, keep], Y[:, keep])["score"] * 100
    return pd.Series(out)


CANDIDATES = {"p3": (["reh_p3_imp"], "fwd_p3_imp", "configs/final_config_p3_imp.json"),
              "p4": (["reh_p4_imp"], "fwd_p4_imp", "configs/final_config_p4_imp.json")}


def judge(name, old_files, fwd_name, st, fb, a):
    """Pre-registered rule for one candidate against v1 -> dict with the per-week table."""
    try:
        old = (scores(old_files, RES) - scores(OLD_BASE, RES)).drop(list(EXCLUDE_WEEKS) + FRESH, errors="ignore").dropna()
    except FileNotFoundError:
        old = pd.Series(dtype=float)
    try:
        fc = scores([fwd_name], a.fwd)
    except FileNotFoundError:
        fc = pd.Series(dtype=float)
    F = pd.DataFrame([{"week": w, "status": st[w], "v1": fb.get(w, np.nan), name: fc.get(w, np.nan)} for w in FRESH])
    F["delta"] = F[name] - F.v1
    comp = F[F.status == "complete"]
    fresh = comp.delta.dropna()
    allw = pd.concat([old, fresh])
    if not len(comp) or comp.delta.isna().any() or not len(old):
        ok = False
        why = ("no complete fresh week" if not len(comp) else
               "a config failed on a fresh week" if comp.delta.isna().any() else "no old-week backtest")
    else:
        c1, c2 = fresh.mean() >= 0, allw.mean() > 0
        ok = bool(c1 and c2)
        why = (f"(1) fresh mean Δ {fresh.mean():+.2f} {'≥' if c1 else '<'} 0; "
               f"(2) all-week mean Δ {allw.mean():+.2f} {'>' if c2 else '≤'} 0")
    t = allw.mean() / (allw.std(ddof=1) / np.sqrt(len(allw))) if len(allw) > 1 else np.nan
    return {"name": name, "passes": ok, "reason": why, "table": F, "fresh_mean": fresh.mean() if len(fresh) else np.nan,
            "old_mean": old.mean() if len(old) else np.nan, "all_mean": allw.mean() if len(allw) else np.nan,
            "n_all": len(allw), "t_all": t, "wins_all": int((allw > 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-end", required=True, help="last data timestamp of the release (UTC)")
    ap.add_argument("--fwd", default=str(RES.parents[0] / "fwd" / "results"))
    ap.add_argument("--list-weeks", action="store_true")
    a = ap.parse_args()
    data_end = pd.Timestamp(a.data_end)
    st = {w: status(w, data_end) for w in FRESH}
    if a.list_weeks:   # weeks with a finished label window (complete or provisional)
        print(",".join(w for w in FRESH if st[w] != "pending"))
        return
    old_weeks = sorted({str(v.date()) for f in OLD_BASE for v, *_ in pickle.load(open(ART / f"folds_{f}.pkl", "rb"))})
    DROP.update(outage_targets(old_weeks + FRESH, data_end))
    try:
        fb = scores([FWD_BASE], a.fwd)
    except FileNotFoundError:
        fb = pd.Series(dtype=float)
    J = [judge(n, o, f, st, fb, a) for n, (o, f, _) in CANDIDATES.items()]
    passing = [j for j in J if j["passes"]]
    best = max(passing, key=lambda j: j["fresh_mean"]) if passing else None
    choice = best["name"] if best else "v1"
    config = CANDIDATES[choice][2] if best else BASE_CONFIG
    md = [f"# Forward check result (release data up to {data_end} UTC)", "",
          "Rule: docs/FORWARD_CHECK.md (pre-registered 2026-10-21; addenda: p3, p4, 3 = outage-aware scoring, "
          "4 = all versions with outage imputation).", "",
          "Activities left out of a week's score (outage in its label window): "
          + ("; ".join(f"{w}: {', '.join(DROP[w])}" for w in sorted(DROP) if DROP[w]) or "none") + ".", ""]
    for j in J:
        n = j["name"]
        md += [f"## {n} vs v1", "", f"| week | status | v1 | {n} | Δ |", "|---|---|---|---|---|"]
        md += [f"| {r.week} | {r.status} | {r.v1:.2f} | {getattr(r, n):.2f} | {r.delta:+.2f} |" for r in j["table"].itertuples()]
        md += ["", f"Old weeks: mean Δ {j['old_mean']:+.2f}. All counted weeks ({j['n_all']}): mean Δ {j['all_mean']:+.2f}, "
               f"t = {j['t_all']:.2f}, {j['wins_all']} of {j['n_all']} better (reported, not used).",
               f"**{'passes' if j['passes'] else 'does not pass'}** — {j['reason']}.", ""]
    md += [f"**Decision: {choice}**" + (" (larger fresh mean Δ among passing candidates)" if len(passing) > 1 else ""), "",
           f"Submit with: `./run_final.sh <studentID> <gpu> {config}`"]
    out = {"data_end": str(data_end), "choice": choice, "config": config, "dropped": DROP,
           "candidates": [{k: (v.round(3).to_dict("records") if isinstance(v, pd.DataFrame) else v) for k, v in j.items()} for j in J]}
    json.dump(out, open(RES / "forward_check.json", "w"), indent=2, default=float)
    (RES / "forward_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main_new():
    """Addendum 5 (2026-11-07): the same rule on the tables built from GH Archive by preprocess.py.
    One backtest per version (backtest_new.sh: new_v1_imp, new_p3_imp, new_p4_imp) holds both the
    development weeks and the fresh weeks; a fresh week counts once its label window has ended."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", action="store_true")
    ap.add_argument("--data-end", required=True)
    ap.add_argument("--fresh", default="2026-10-14,2026-10-21,2026-10-28")
    ap.add_argument("--prefix", default="new", help="backtest names <prefix>_{v1,p3,p4}_imp (a6 = addendum 6)")
    ap.add_argument("--weeks", default="", help="--formal: comma list of all non-fresh weeks every candidate must have")
    ap.add_argument("--formal", action="store_true",
                    help="addendum 7: stop unless every fresh week is mature and scored for every candidate")
    a = ap.parse_args()
    data_end = pd.Timestamp(a.data_end)
    fresh = a.fresh.split(",")
    weeks = sorted({str(v.date()) for v, *_ in pickle.load(open(ART / f"folds_{a.prefix}_v1_imp.pkl", "rb"))})
    DROP.update(outage_targets(weeks, data_end))
    sc = pd.DataFrame({n: scores([f"{a.prefix}_{n}_imp"], RES) for n in ["v1", "p3", "p4"]})
    done = [w for w in fresh if w in sc.index and
            data_end >= pd.Timestamp(f"{w} 12:00") + pd.Timedelta(seconds=GAP_SEC + 7 * DAY)]
    # labels are final only after the 5-day slack of build_dataset (late-arriving events) has passed
    immature = [w for w in done if data_end < pd.Timestamp(f"{w} 12:00") + pd.Timedelta(seconds=GAP_SEC + 12 * DAY)]
    if a.formal and not a.weeks:
        raise SystemExit("ERROR (--formal): --weeks <all non-fresh weeks> is required")
    if a.formal:
        bad = [w for w in fresh if w not in done or w in immature or not np.isfinite(sc.loc[w]).all()]
        if bad:
            raise SystemExit(f"ERROR (--formal): fresh weeks not mature or not scored for all candidates: {bad}")
        # every candidate: the same weeks, all expected weeks present, every score finite
        wk = {n: sorted({str(v.date()) for v, *_ in pickle.load(open(ART / f"folds_{a.prefix}_{n}_imp.pkl", "rb"))})
              for n in ["v1", "p3", "p4"]}
        expected = sorted(set(a.weeks.split(",")) | set(fresh))
        if len({tuple(x) for x in wk.values()}) != 1 or wk["v1"] != expected:
            raise SystemExit(f"ERROR (--formal): candidates do not cover the same / the expected weeks: {wk}")
        if not np.isfinite(sc.to_numpy()).all():
            raise SystemExit("ERROR (--formal): a candidate has a missing or non-finite week score")
        # partial guard against stale folds of an earlier run (mtime only; not a proof of provenance)
        ds = (ART / "dataset_imp.parquet").stat().st_mtime
        stale = [n for n in wk if (ART / f"folds_{a.prefix}_{n}_imp.pkl").stat().st_mtime < ds]
        if stale:
            raise SystemExit(f"ERROR (--formal): folds older than artifacts/dataset_imp.parquet: {stale}")
    old = [w for w in sc.index if w not in fresh and w not in EXCLUDE_WEEKS]
    md = [f"# Forward check on the GH Archive tables (data up to {data_end} UTC)", "",
          f"Rule: docs/FORWARD_CHECK.md, addendum 5{', with the outage-detection correction of addendum 6' if a.prefix == 'a6' else ''} (same conditions: fresh-week mean Δ ≥ 0 and all-week mean Δ > 0, else v1).", "",
          "Activities left out of a week's score (outage in its label window): "
          + ("; ".join(f"{w}: {', '.join(DROP[w])}" for w in sorted(DROP) if DROP[w]) or "none") + ".", "",
          "| week | kind | v1 | p3 | p4 |", "|---|---|---|---|---|"]
    for w in sc.index:
        kind = "fresh" if w in done else "fresh (not finished)" if w in fresh else "reported only" if w in EXCLUDE_WEEKS else "development"
        md.append(f"| {w} | {kind} | {sc.v1[w]:.2f} | {sc.p3[w]:.2f} | {sc.p4[w]:.2f} |")
    md.append(f"| mean of {len(old)} dev + {len(done)} fresh | | {sc.v1[old + done].mean():.2f} | {sc.p3[old + done].mean():.2f} | {sc.p4[old + done].mean():.2f} |")
    res, passing = {}, []
    for n in ["p3", "p4"]:
        d = sc[n] - sc.v1
        fm, am = d[done].mean(), d[old + done].mean()
        ok = bool(len(done) and fm >= 0 and am > 0)
        t = am / (d[old + done].std(ddof=1) / np.sqrt(len(old + done)))
        res[n] = {"passes": ok, "fresh_mean_delta": fm, "all_mean_delta": am, "t": t}
        md += ["", f"**{n}: {'passes' if ok else 'does not pass'}** — fresh mean Δ {fm:+.2f}, all-week mean Δ {am:+.2f} "
               f"(t = {t:.2f}, {int((d[old + done] > 0).sum())} of {len(old + done)} weeks better)."]
        if ok:
            passing.append((fm, n))
    choice = max(passing)[1] if passing else "v1"
    config = {"v1": BASE_CONFIG, "p3": CANDIDATES["p3"][2], "p4": CANDIDATES["p4"][2]}[choice]
    if immature:
        md += ["", f"**PROVISIONAL** — the labels of {', '.join(immature)} are not final yet (5-day slack after the "
               "label window); this is not a decision."]
    md += ["", f"**{'Provisional result' if immature else 'Decision'}: {choice}**", "",
           f"{'Would submit' if immature else 'Submit'} with: `./run_final.sh <studentID> <gpu> {config}`"]
    json.dump({"data_end": str(data_end), "choice": choice, "provisional_weeks": immature, "config": config,
               "formal": bool(a.formal), "prefix": a.prefix,
               "dropped": DROP, "fresh": done,
               "scores": sc.round(3).to_dict(), "candidates": res},
              open(RES / f"forward_check_{a.prefix}.json", "w"), indent=2, default=float)
    (RES / f"forward_check_{a.prefix}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    import sys
    main_new() if "--new" in sys.argv else main()
