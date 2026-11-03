#!/usr/bin/env bash
# Build a Python 3.10 environment with RSOD's exact versions on Colab:
# torch 2.0.0 + CUDA 11.8, mmcv 2.1.0, mmengine 0.10.6, RSOD's mmdet fork (editable).
# Called by notebook cell 3. Kept in the repo so fixes arrive with `git pull`.
set -euo pipefail
VENV=${VENV:-/content/venv310}
RSOD=${RSOD:-/content/RSOD}

pip -q install uv
uv python install -q 3.10
[ -x "$VENV/bin/python" ] || uv venv -q "$VENV" --python 3.10 --seed
P=$VENV/bin/python

echo "[1/6] torch 2.0.0 + cu118"
uv pip install -q --python $P torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
echo "[2/6] setuptools < 70 (torch 2.0's cpp_extension imports pkg_resources.packaging)"
uv pip install -q --python $P "setuptools==69.5.1" wheel
echo "[3/6] numpy 1.26.4, mmengine 0.10.6"
uv pip install -q --python $P "numpy==1.26.4" mmengine==0.10.6 cython
echo "[4/6] mmcv 2.1.0 (prebuilt CUDA ops)"
uv pip install -q --python $P mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
echo "[5/6] RSOD runtime requirements"
uv pip install -q --python $P -r "$RSOD/requirements/runtime.txt"
echo "[6/6] RSOD (editable) + wandb (for CoreWeave Forge logging)"
uv pip install -q --python $P wandb
uv pip install -q --python $P --no-build-isolation -e "$RSOD"
# Some installs above can upgrade setuptools again; re-assert the pin.
uv pip install -q --python $P "setuptools==69.5.1"

MPLBACKEND=Agg $P - <<'EOF'
import torch, mmcv, mmengine, mmdet, setuptools
import mmdet.utils  # imports matplotlib too, as tools/train.py does
from mmcv.ops import RoIAlign, nms  # fails here if mmcv was not built with CUDA ops
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(),
      '| mmcv', mmcv.__version__, '| mmengine', mmengine.__version__, '| mmdet', mmdet.__version__,
      '| setuptools', setuptools.__version__)
x = torch.rand(10, 4, device='cuda') * 50; x[:, 2:] += x[:, :2]
print('CUDA nms OK:', nms(x, torch.rand(10, device='cuda'), 0.5)[0].shape)
EOF
echo "Environment ready: $VENV"
