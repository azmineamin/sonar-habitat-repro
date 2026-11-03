#!/usr/bin/env bash
# Notebook cell 8: compare our test mAP with the RSOD paper; also logs the table to
# W&B / CoreWeave Forge when WANDB_PROJECT is set (see the Forge cell).
set -euo pipefail
: "${DRIVE_ROOT:?run cell 0 (settings) first}"
VENV=${VENV:-/content/venv310}
PY=python3; [ -x "$VENV/bin/python" ] && PY=$VENV/bin/python
args=(--work-root "$DRIVE_ROOT/work_dirs" --out "$DRIVE_ROOT/results.csv")
if [ -n "${WANDB_PROJECT:-}" ]; then
  "$PY" -c "import wandb" 2>/dev/null || uv pip install -q --python "$PY" wandb
  args+=(--wandb)
fi
"$PY" /content/kit/tools/collect_results.py "${args[@]}"
