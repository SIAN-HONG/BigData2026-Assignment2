"""Record what a run used, so a rerun can be compared with it.

  python src/manifest.py --raw ~/gharchive --start 2025-12-01 --cutoff 2026-11-21T12:00 \
      --config configs/final_config_s3.json --out results/manifest.json [--hash-hours]
  python src/manifest.py --compare results/manifest.json other/manifest.json

Writes: the hour-file bound (--cutoff) and the prediction cutoff, the effective config, ASG2_* settings,
course repository commit and file hashes (preprocess.py, evaluate.py, cohort.txt, test_input.csv), our code
hashes, the GH Archive hour files of [start, cutoff) with sizes (and sha256 with --hash-hours), the 8 tables
read (ASG2_DB or data/db), events / training dataset / final features, this run's output files (from
results/final_prediction_info.json), and package / CUDA / GPU versions.  A missing input is an error.
--compare compares every field of both files; created_utc and versions are shown but never fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def files(paths) -> dict:
    return {str(p.relative_to(ROOT)): {"size": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(paths) if p.is_file()}


def run(cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # tool not installed
        return f"unavailable ({type(e).__name__})"


def versions() -> dict:
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ["numpy", "pandas", "pyarrow", "scipy", "sklearn", "xgboost", "torch"]:
        try:
            v[m] = __import__(m).__version__
        except Exception:
            v[m] = None
    try:
        import torch
        v["torch_cuda"] = torch.version.cuda
        v["cudnn"] = torch.backends.cudnn.version()
        v["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    v["nvidia_driver"] = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"])
    return v


HASH_CACHE = ROOT / "results" / "gharchive_sha256_cache.json"   # name -> [size, mtime_ns, sha256]


def hours(raw: Path, start: str, cutoff: str, do_hash: bool, threads: int = 32) -> dict:
    t, end = datetime.fromisoformat(start), datetime.fromisoformat(cutoff)
    names = []
    while t < end:
        names.append(f"{t:%Y-%m-%d}-{t.hour}.json.gz")
        t += timedelta(hours=1)
    miss = [n for n in names if not (raw / n).is_file()]

    cache = json.load(open(HASH_CACHE)) if do_hash and HASH_CACHE.exists() else {}

    def one(n):
        p = raw / n
        st = p.stat()
        d = {"size": st.st_size}
        if do_hash:  # hashing ~230 GB takes ~30 min on this NFS; reuse hashes of unchanged files
            c = cache.get(n)
            d["sha256"] = c[2] if c and c[:2] == [st.st_size, st.st_mtime_ns] else sha256(p)
            cache[n] = [st.st_size, st.st_mtime_ns, d["sha256"]]
        return n, d

    with ThreadPoolExecutor(threads) as ex:
        got = dict(ex.map(one, [n for n in names if n not in set(miss)]))
    if do_hash:
        HASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(cache, open(HASH_CACHE, "w"))
    return {"start": start, "end_exclusive": cutoff, "n_expected": len(names), "missing": miss,
            "total_bytes": sum(d["size"] for d in got.values()), "files": got}


def need(paths) -> list:
    miss = [str(p) for p in paths if not Path(p).is_file()]
    if miss:
        raise SystemExit(f"ERROR: manifest inputs missing: {miss}")
    return list(paths)


def build(a) -> dict:
    course = ROOT / "assignment2"
    db = Path(os.environ.get("ASG2_DB", ROOT / "data" / "db"))   # the tables common.load_tables reads
    info_p = ROOT / "results/final_prediction_info.json"
    info = json.load(open(need([info_p])[0]))
    cfg = info["cfg"]                                             # effective config used by predict.py
    if a.config:
        need([ROOT / a.config])
    ds = ROOT / "artifacts" / cfg["dataset"]
    m = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "raw_end_exclusive": a.cutoff,                           # right bound of the hour files used
         "prediction_cutoff": info["cutoff"],                      # cutoff of the features predict.py used
         "config_path": a.config, "config_effective": cfg,
         "env": {k: os.environ.get(k) for k in ["ASG2_DB", "ASG2_ART", "ASG2_OUTAGE_RULE"]},
         "course_repo": {"commit": run(["git", "-c", "safe.directory=*", "-C", str(course), "rev-parse", "HEAD"]),
                         "files": files(need([course / "scripts/preprocess.py", course / "scripts/evaluate.py",
                                              course / "scripts/cohort.txt",
                                              course / "submission/test_input.csv"]))},
         "code": files(list((ROOT / "src").glob("*.py")) + list((ROOT / "configs").glob("*.json"))
                       + [ROOT / "run_final.sh", ROOT / "download_gharchive.sh", ROOT / "requirements.txt",
                          ROOT / "final_config.json"]),
         "versions": versions(),
         "tables": {p.name: {"size": p.stat().st_size, "sha256": sha256(p)} for p in
                    sorted(need([db / f"{t}.parquet" for t in
                                 ["Users", "Repositories", "PullRequests", "Issues", "IssueComments",
                                  "PRReviews", "PRReviewComments", "Pushes"]]))},
         "artifacts": files(need([ROOT / "artifacts/events.parquet", ds,
                                  ROOT / "artifacts" / cfg["dataset"].replace("dataset", "final_features")])),
         "outputs": files(need([ROOT / info["output"], ROOT / "results/final_probabilities.csv", info_p])),
         "artifacts_used": {"dataset": f"artifacts/{cfg['dataset']}",
                            "final_features": f"artifacts/{cfg['dataset'].replace('dataset', 'final_features')}"}}
    # the files must come from one run: the features predict.py used are the ones on disk
    import pandas as pd
    fcut = pd.read_parquet(ROOT / m["artifacts_used"]["final_features"], columns=["cutoff"]).cutoff.iloc[0]
    if pd.Timestamp(fcut) != pd.Timestamp(info["cutoff"]):
        raise SystemExit(f"ERROR: final features are at {fcut} but the prediction info says {info['cutoff']} "
                         "(outputs and inputs come from different runs)")
    if a.raw:
        m["gharchive"] = hours(Path(a.raw).expanduser(), a.start, a.cutoff, a.hash_hours, a.threads)
        if m["gharchive"]["missing"]:
            raise SystemExit(f"ERROR: {len(m['gharchive']['missing'])} hour files missing, "
                             f"e.g. {m['gharchive']['missing'][:3]}")
    return m


def flat(x, pre="") -> dict:
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(flat(v, f"{pre}.{k}" if pre else str(k)))
        return out if out else {pre: {}}
    return {pre: x}


INFO_ONLY = ("created_utc", "versions")   # shown, but not a failure (other machines differ)


def compare(p1: str, p2: str) -> int:
    A, B = flat(json.load(open(p1))), flat(json.load(open(p2)))
    keys = sorted(set(A) | set(B))
    diff = [k for k in keys if A.get(k, "<absent>") != B.get(k, "<absent>")]
    secs = sorted({k.split(".")[0] for k in keys})
    bad = 0
    for sec in secs:
        d = [k for k in diff if k.split(".")[0] == sec]
        if not d:
            print(f"{sec:20s} same")
            continue
        info = sec in INFO_ONLY
        bad += not info
        print(f"{sec:20s} {len(d)} differ{' (info only)' if info else ''}: "
              + "; ".join(f"{k}: {A.get(k, '<absent>')!s:.40} | {B.get(k, '<absent>')!s:.40}" for k in d[:4])
              + (" ..." if len(d) > 4 else ""))
    print("MANIFESTS MATCH" if bad == 0 else f"MANIFESTS DIFFER in {bad} section(s)")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="GH Archive hour-file directory")
    ap.add_argument("--start", default="2025-12-01")
    ap.add_argument("--cutoff", default="2026-11-21T12:00", help="right bound (exclusive) of the hour files used")
    ap.add_argument("--config", default=None)
    ap.add_argument("--hash-hours", action="store_true", help="sha256 of every hour file (~200 GB to read)")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "results/manifest.json"))
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    m = build(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(a.out, "w"), indent=1)
    g = m.get("gharchive", {})
    print(f"manifest -> {a.out}; course commit {m['course_repo']['commit'][:9]}; prediction cutoff "
          f"{m['prediction_cutoff']}; "
          f"hours {g.get('n_expected')} expected, {len(g.get('missing', []))} missing")


if __name__ == "__main__":
    main()
