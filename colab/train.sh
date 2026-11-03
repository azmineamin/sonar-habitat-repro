#!/usr/bin/env bash
# Notebook cell 6: train + test with live output, also saved to Drive.
# Re-running resumes from the last checkpoint and skips finished runs.
set -euo pipefail
: "${DRIVE_ROOT:?run cell 0 (settings) first}"
: "${UATD_DATA_ROOT:?run cell 4 (dataset) first}"
VENV=${VENV:-/content/venv310}
RSOD=${RSOD:-/content/RSOD}

[ -x "$VENV/bin/python" ] || { echo "ERROR: environment missing - run cell 3 first"; exit 1; }
[ -f "$UATD_DATA_ROOT/annotations/instances_test.json" ] || { echo "ERROR: dataset missing at $UATD_DATA_ROOT - run cell 4"; exit 1; }
ls "$RSOD"/projects/RSOD/configs/repro/*_uatd_*pct_seed*.py >/dev/null 2>&1 || { echo "ERROR: configs missing - run cell 5"; exit 1; }

export PATH="$VENV/bin:$PATH" PYTHONUNBUFFERED=1
export WORK_ROOT="$DRIVE_ROOT/work_dirs"
mkdir -p "$WORK_ROOT"
echo "python: $(command -v python)  | data: $UATD_DATA_ROOT  | runs: $WORK_ROOT"
echo "settings: PCTS='${PCTS:-}' SEEDS='${SEEDS:-}' METHODS='${METHODS:-}' ITERS='${ITERS:-}' CKPT_INTERVAL='${CKPT_INTERVAL:-}'"
cd "$RSOD"
bash /content/kit/scripts/run_uatd.sh 2>&1 | tee -a "$WORK_ROOT/console.log"
