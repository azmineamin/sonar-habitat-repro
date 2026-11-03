#!/usr/bin/env bash
# Notebook cell 8: compare our test mAP with the RSOD paper (printed + results.csv on Drive).
# If the Forge cell ran (WANDB_PROJECT set), also upload the tables and build/refresh the
# Forge report "RSOD reproduction on UATD" that shows only the reproduction results.
set -euo pipefail
: "${DRIVE_ROOT:?run cell 0 (settings) first}"
VENV=${VENV:-/content/venv310}
PY=python3; [ -x "$VENV/bin/python" ] && PY=$VENV/bin/python
# Colab's own Python (e.g. wandb.login() in the Forge cell) starts a wandb-core service and
# exports WANDB_SERVICE; the 3.10 venv runs a different wandb version and must use its own.
unset WANDB_SERVICE
args=(--work-root "$DRIVE_ROOT/work_dirs" --out "$DRIVE_ROOT/results.csv")
if [ -n "${WANDB_PROJECT:-}" ]; then
  export WANDB_X_DISABLE_STATS=true
  "$PY" -c "import wandb, wandb_workspaces" 2>/dev/null || uv pip install -q --python "$PY" wandb wandb-workspaces
  args+=(--wandb)
fi
if [ -n "${WANDB_PROJECT:-}" ]; then
  # Runs finished before Forge logging was switched on: upload their saved curves + test mAP once.
  "$PY" /content/kit/tools/backfill_forge.py --work-root "$DRIVE_ROOT/work_dirs"
fi
"$PY" /content/kit/tools/collect_results.py "${args[@]}"
if [ -n "${WANDB_PROJECT:-}" ]; then
  "$PY" /content/kit/tools/forge_report.py
fi
