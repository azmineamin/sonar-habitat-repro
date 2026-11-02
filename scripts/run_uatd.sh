#!/usr/bin/env bash
# Reproduce RSOD Table 1/2 (UATD rows): supervised baseline + RSOD at 1/5/10 % labels.
# Run from the RSOD repo root inside its environment (Linux, WSL2 or Colab, NVIDIA GPU).
#
#   export UATD_DATA_ROOT=/data/uatd_coco
#   bash /path/to/sonar-habitat-repro/scripts/run_uatd.sh            # all settings, seeds 1-3
#   PCTS="5" SEEDS="1" bash .../run_uatd.sh                            # one setting first
#
# Resume-safe: re-running the same command continues an interrupted run from its
# last checkpoint (configs set resume=True) and skips runs marked DONE.
#   WORK_ROOT=/content/drive/MyDrive/sonar-habitat/work_dirs   keep checkpoints on Drive (Colab)
#   CKPT_INTERVAL=2000      save more often than the paper's 20k (does not change results)
#   ITERS=40000             shortened schedule for debugging (NOT comparable to the paper)
set -euo pipefail
: "${UATD_DATA_ROOT:?set UATD_DATA_ROOT to the converted COCO folder}"
PCTS=${PCTS:-"1 5 10"}
SEEDS=${SEEDS:-"1 2 3"}
METHODS=${METHODS:-"sup rsod"}
ITERS=${ITERS:-}
CKPT_INTERVAL=${CKPT_INTERVAL:-}
WORK_ROOT=${WORK_ROOT:-work_dirs}
CFG=projects/RSOD/configs/repro
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

for pct in $PCTS; do
  for seed in $SEEDS; do
    for m in $METHODS; do
      cfg=$CFG/${m}_uatd_${pct}pct_seed${seed}.py
      name=${m}_uatd_${pct}pct_seed${seed}
      opts=()
      if [ -n "$ITERS" ]; then opts+=(train_cfg.max_iters=$ITERS); name=${name}_it${ITERS}; fi
      if [ -n "$CKPT_INTERVAL" ]; then opts+=(default_hooks.checkpoint.interval=$CKPT_INTERVAL); fi
      extra=(); [ ${#opts[@]} -gt 0 ] && extra=(--cfg-options "${opts[@]}")
      wd=$WORK_ROOT/$name
      if [ -f "$wd/DONE" ]; then echo "=== $name already done, skipping"; continue; fi
      echo "=== train $name  ($(date))"
      python tools/train.py "$cfg" --work-dir "$wd" "${extra[@]}"
      ckpt=$(ls -t "$wd"/best_*.pth 2>/dev/null | head -1 || true)
      if [ -z "$ckpt" ]; then ckpt=$(cat "$wd/last_checkpoint"); fi
      echo "=== test $name with $(basename "$ckpt")"
      python tools/test.py "$cfg" "$ckpt" --work-dir "$wd" "${extra[@]}"
      date > "$wd/DONE"
    done
  done
done
echo "Done. Compare with the paper:  python <kit>/tools/collect_results.py --work-root $WORK_ROOT"
