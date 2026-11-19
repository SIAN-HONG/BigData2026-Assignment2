#!/usr/bin/env bash
# Download GH Archive hour files (https://www.gharchive.org) for a UTC period.
# usage: ./download_gharchive.sh RAW_DIR [START=2025-12-01] [END=now-2h]   (END exclusive, UTC)
# Files already present with the right size are skipped, so it can be rerun to add new hours.
set -euo pipefail
RAW=${1:?RAW_DIR}; START=${2:-2025-12-01}; END=${3:-$(date -u -d '-2 hour' +%Y-%m-%dT%H:00)}
mkdir -p "$RAW"
python3 - "$START" "$END" <<'PY' | xargs -P 16 -I{} sh -c 'f="$1"; [ -s "$2/$f" ] || wget -q -t 5 -T 60 -O "$2/$f.part" "https://data.gharchive.org/$f" && { [ -s "$2/$f" ] || mv "$2/$f.part" "$2/$f"; } || { rm -f "$2/$f.part"; echo "missing $f"; }' _ {} "$RAW"
import sys
from datetime import datetime, timedelta
t, end = datetime.fromisoformat(sys.argv[1]), datetime.fromisoformat(sys.argv[2])
while t < end:
    print(f"{t:%Y-%m-%d}-{t.hour}.json.gz"); t += timedelta(hours=1)
PY
echo "files: $(ls "$RAW" | grep -c 'json.gz$')"
# every hour in [START, END) must be present and non-empty, otherwise stop (a missing hour would make
# our tables differ from a rerun that has it; GH Archive answers 404 for hours not yet published)
python3 - "$RAW" "$START" "$END" <<'PY'
import os, sys
from datetime import datetime, timedelta
raw, t, end = sys.argv[1], datetime.fromisoformat(sys.argv[2]), datetime.fromisoformat(sys.argv[3])
miss = []
while t < end:
    f = f"{t:%Y-%m-%d}-{t.hour}.json.gz"
    if not (os.path.isfile(os.path.join(raw, f)) and os.path.getsize(os.path.join(raw, f)) > 0):
        miss.append(f)
    t += timedelta(hours=1)
if miss:
    print(f"ERROR: {len(miss)} hour file(s) missing, e.g. {miss[:5]} - wait until GH Archive publishes them "
          "and rerun, or use an earlier END", file=sys.stderr)
    sys.exit(1)
print("all hours present")
PY
