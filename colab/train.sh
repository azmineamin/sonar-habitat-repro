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
# Colab sets MPLBACKEND to its notebook backend, which the 3.10 venv does not have.
export MPLBACKEND=Agg
# Colab's own Python (e.g. wandb.login() in the Forge cell) starts a wandb-core service and
# exports WANDB_SERVICE; the 3.10 venv runs a different wandb version and must use its own.
unset WANDB_SERVICE
if [ -n "${WANDB_PROJECT:-}" ]; then
  # Accept either the WANDB_API_KEY secret or an earlier `wandb.login()` (stored in ~/.netrc).
  if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    echo "ERROR: not logged in to W&B/Forge - add the Colab secret WANDB_API_KEY, or run wandb.login() in a cell"; exit 1
  fi
  "$VENV/bin/python" -c "import wandb" 2>/dev/null || uv pip install -q --python "$VENV/bin/python" wandb
  # Log only the reproduction results: no automatic GPU memory / power / CPU charts.
  export WANDB_X_DISABLE_STATS=true
  echo "logging to CoreWeave Forge (W&B): project=$WANDB_PROJECT entity=${WANDB_ENTITY:-<your default>} (system metrics off)"
else
  echo "W&B / Forge logging off (run the Forge cell to turn it on)"
fi
export WORK_ROOT="$DRIVE_ROOT/work_dirs"
mkdir -p "$WORK_ROOT"
echo "python: $(command -v python)  | data: $UATD_DATA_ROOT  | runs: $WORK_ROOT"
echo "settings: PCTS='${PCTS:-}' SEEDS='${SEEDS:-}' METHODS='${METHODS:-}' ITERS='${ITERS:-}' CKPT_INTERVAL='${CKPT_INTERVAL:-}'"
cd "$RSOD"
bash /content/kit/scripts/run_uatd.sh 2>&1 | tee -a "$WORK_ROOT/console.log"
