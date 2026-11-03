#!/usr/bin/env bash
# Notebook cell 5: generate RSOD + supervised-baseline configs for the chosen splits.
# Reads PCTS / SEEDS (cell 0) and UATD_DATA_ROOT (cell 4) from the environment.
set -euo pipefail
: "${UATD_DATA_ROOT:?run cell 4 (dataset) first}"
PCTS=${PCTS:-5}
SEEDS=${SEEDS:-1}
RSOD=${RSOD:-/content/RSOD}
# shellcheck disable=SC2086  # PCTS/SEEDS are space-separated lists on purpose
python "${KIT:-/content/kit}/tools/gen_rsod_configs.py" --rsod-repo "$RSOD" --dataset uatd \
    --coco-train "$UATD_DATA_ROOT/annotations/instances_train.json" \
    --percents $PCTS --seeds $SEEDS
ls "$RSOD/projects/RSOD/configs/repro"
