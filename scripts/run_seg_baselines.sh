#!/usr/bin/env bash
# Supervised habitat-segmentation baselines on BenthiCat at 1/5/10/100 % labelled waterfalls.
#   BENTHICAT_ROOT=/data/benthicat_seg TRAIN_IMG=train/images TRAIN_MASK=train/masks \
#   TEST_IMG=test/images TEST_MASK=test/masks bash scripts/run_seg_baselines.sh
# (Run seg/benthicat_inspect.py first to find the right sub-folders.)
set -euo pipefail
: "${BENTHICAT_ROOT:?}"; : "${TRAIN_IMG:?}"; : "${TRAIN_MASK:?}"; : "${TEST_IMG:?}"; : "${TEST_MASK:?}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
FRACS=${FRACS:-"0.01 0.05 0.10 1.0"}
SEEDS=${SEEDS:-"1 2 3"}
ITERS=${ITERS:-20000}
for f in $FRACS; do
  for s in $SEEDS; do
    python "$HERE/seg/train_seg_baseline.py" --root "$BENTHICAT_ROOT" \
      --train-img "$TRAIN_IMG" --train-mask "$TRAIN_MASK" --test-img "$TEST_IMG" --test-mask "$TEST_MASK" \
      --label-frac "$f" --seed "$s" --iters "$ITERS" --out "runs/seg_sup_${f}_seed${s}"
  done
done
