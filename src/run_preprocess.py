"""Run the course's assignment2/scripts/preprocess.py unchanged, only with more worker processes.

    python src/run_preprocess.py --raw RAW_DATA_FOLDER --out data [--workers 32]

The official script reads the hour files with a fixed pool of 4 processes (WORKERS = 4).  The pool
uses imap, which returns the files in order, so the number of workers changes only the run time,
not the tables.  Everything else (cohort filter, tables, anonymization, output) is the official
code.  `python assignment2/scripts/preprocess.py --raw ... --out ...` gives the same files.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "assignment2" / "scripts" / "preprocess.py"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args, rest = ap.parse_known_args()
    spec = importlib.util.spec_from_file_location("preprocess", SCRIPT)
    pre = importlib.util.module_from_spec(spec)
    sys.modules["preprocess"] = pre          # so the worker processes can find extract_hour
    spec.loader.exec_module(pre)
    pre.WORKERS = args.workers
    sys.argv = [str(SCRIPT)] + rest
    pre.main()
