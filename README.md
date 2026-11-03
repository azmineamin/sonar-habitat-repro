# Sonar habitat mapping: reproduction kit

Reproduction of the baselines our project builds on:

1. **RSOD** (Li et al., *AAAI 2026*, [paper](https://doi.org/10.1609/aaai.v40i8.37529), [code](https://github.com/LICZ9/RSOD)):
   reliability-guided semi-supervised sonar detection. We reproduce its UATD rows (and FSOD if you can get the data).
2. **Supervised habitat segmentation on BenthiCat**: the few-label lower bound our
   pixel-level reliability method must beat.

## Target numbers (from the RSOD paper)

COCO bbox mAP (%), test set. `targets/paper_targets.csv` holds these; `tools/collect_results.py` compares against them.

| Method | UATD 1% | UATD 5% | UATD 10% | FSOD 5% | FSOD 10% | FSOD 20% |
|---|---|---|---|---|---|---|
| Faster R-CNN, labeled only | 13.7 | 22.9 | 26.2 | 44.9 | 50.6 | 59.6 |
| RSOD | 23.2 | 34.7 | 34.8 | 67.4 | 72.6 | 75.5 |

Small-object mAP on UATD: supervised 2.9 / 7.1 / 7.3, RSOD 10.1 / 12.6 / 12.8.

## Running on Google Colab (recommended for us)

Open `colab/RSOD_reproduction_colab.ipynb` in Colab (File → Open notebook → GitHub tab → this repo), choose a GPU runtime, set `GITHUB_REPO` in cell 0, and run all cells.

- It builds a separate Python 3.10 environment with RSOD's exact versions (torch 2.0.0 + CUDA 11.8, mmcv 2.1.0, mmengine 0.10.6). Colab's own Python is too new for these mmcv wheels.
- It downloads UATD from figshare once and stores it on Google Drive as **lossless grayscale PNG** (UATD's three BMP channels are identical, and the converter checks this per image).
- Checkpoints and logs go to Drive. After a disconnect, run all cells again: training resumes from the last checkpoint, and finished runs are skipped.
- Drive space: each RSOD checkpoint holds teacher + student weights, so keep a few GB free per run.

### Viewing results in CoreWeave Forge (W&B Models)

Logging is optional and off by default. To turn it on:
1. Sign up or log in at https://id.coreweave.com/signup and copy your API key (User settings → API keys).
2. In Colab, add a secret named `WANDB_API_KEY` with that key (key icon in the left bar; allow notebook access).
3. Run cell **5b** before training. Set `FORGE_PROJECT` there (default `sonar-habitat-repro`) and optionally `FORGE_ENTITY` (our team: `walidur-rahman01-north-south-university`). Instead of the secret you can also run `wandb.login()` in a cell.

Then:
- **Each training run** (cell 6) appears as its own run: loss, learning rate and validation mAP curves, plus the full config.
  Runs are grouped by setting (`rsod_uatd_5pct`, `sup_uatd_5pct`, …) and tagged with method, label % and seed.
  A run resumed after a disconnect continues the same Forge run, because its id is fixed.
- **Each test** appears as a `<run>-test` run (job type `eval`) with the test-set COCO metrics.
- **Cell 8** uploads a `results-summary` run with the table "ours vs. paper" (mean ± std over seeds),
  a per-run table, and an "ours vs. paper" bar chart for full-schedule runs.

Project URL: `https://forge.coreweave.com/wandb/<entity>/<project>`.
Under the hood this is the standard `wandb` library (MMEngine's `WandbVisBackend`); Forge serves the W&B app,
and the `api.wandb.ai` endpoint keeps working, so nothing else needs configuring.

## What you need (own machine)

- An **NVIDIA GPU** (the authors used one RTX 4090; each run is 180k iterations).
- **Linux or WSL2 on Windows.** RSOD's runtime config uses `mp_start_method='fork'`, which native Windows does not support.
- RSOD's environment: Python 3.9, PyTorch 2.0.0 + CUDA 11.8, mmengine 0.10.6, mmcv 2.1.0 (see the RSOD README).

## Steps

```bash
# 0. Get the code
git clone https://github.com/LICZ9/RSOD.git
# follow RSOD's README to create the conda env (torch 2.0.0/cu118, mmcv 2.1.0, mmengine 0.10.6)
pip install -e RSOD   # or run everything from inside RSOD/ with PYTHONPATH=.

# 1. Download UATD from figshare: https://doi.org/10.6084/m9.figshare.21331143
#    unzip UATD_Training.zip, UATD_Test_1.zip, UATD_Test_2.zip into /data/UATD

# 2. Convert to COCO and make label splits (seeds 1-3)
python tools/uatd_to_coco.py --uatd-root /data/UATD --out /data/uatd_coco
python tools/make_ssod_splits.py /data/uatd_coco/annotations/instances_train.json --percents 1 5 10 --seeds 1 2 3

# 3. Generate configs inside the RSOD repo
python tools/gen_rsod_configs.py --rsod-repo ../RSOD --dataset uatd \
    --coco-train /data/uatd_coco/annotations/instances_train.json --percents 1 5 10 --seeds 1 2 3

# 4. Train + test (from the RSOD repo root). Start with ONE setting to check speed.
cd ../RSOD
export UATD_DATA_ROOT=/data/uatd_coco
PCTS="5" SEEDS="1" bash ../sonar-habitat-repro/scripts/run_uatd.sh
# then everything:
bash ../sonar-habitat-repro/scripts/run_uatd.sh

# 5. Compare with the paper
python ../sonar-habitat-repro/tools/collect_results.py --work-root work_dirs
```

The full UATD table is 3 label ratios × 3 seeds × 2 methods = 18 runs of 180k iterations.
If GPU time is short, run **5% labels, seed 1, both methods** first; that is RSOD's headline result (22.9 → 34.7).
`ITERS=40000` gives a shorter schedule for debugging, but its numbers are not comparable to the paper.

**FSOD** (the authors' dataset) is on Baidu Netdisk (link and code `0516` in the RSOD README).
After downloading, make splits with `--percents 5 10 20` and generate configs with `--dataset fsod`.

## Habitat segmentation baseline (BenthiCat)

```bash
# Download "BenthiCat - SSS Seafloor Segmentation" from https://dataverse.harvard.edu/dataverse/benthicat
python seg/benthicat_inspect.py --root /data/benthicat_seg                     # shows the folder layout
python seg/benthicat_inspect.py --root /data/benthicat_seg --img-dir <...> --mask-dir <...>  # checks pairs + classes
BENTHICAT_ROOT=/data/benthicat_seg TRAIN_IMG=<...> TRAIN_MASK=<...> TEST_IMG=<...> TEST_MASK=<...> \
    bash scripts/run_seg_baselines.sh                                             # 1/5/10/100 %, seeds 1-3
```

Needs only `torch`, `torchvision`, `numpy` and `opencv-python`. It uses a U-Net with an ImageNet ResNet-34 encoder,
384×384 tiles at stride 192, labels sampled per waterfall (not per tile, to avoid leakage), and
the official sector-disjoint test split. Results go to `runs/*/metrics.json` (mIoU and per-class IoU).

## Deviations and assumptions to record in the paper

| Item | Paper | Released code / our choice |
|---|---|---|
| Learning rate | 0.001 | 0.01 in the released config, kept as released |
| mAP definition | "standard mAP" | COCO mAP@[.5:.95] (released config uses `CocoMetric`); mAP@.5 also logged |
| UATD val/test | 7,600 / 800 / 800 | Assumed Test_1 = val, Test_2 = test (`--val-split/--test-split` to change) |
| Label-split seeds | not given | seeds 1, 2, 3; report mean ± std |
| Supervised baseline schedule | not given | same 180k schedule, pipeline and backbone as RSOD |
| Checkpoint for testing | not given | best on val (`save_best`), then evaluated once on test |

## Status of this kit

Tested here on synthetic data (no GPU or internet in the build machine):
- `uatd_to_coco.py`, `make_ssod_splits.py`: converted and split a fake UATD tree; every box's class was checked against its XML.
- `gen_rsod_configs.py`: all 18 generated configs, plus the original FSOD config, parse with mmengine 0.10.6; dataset paths, splits, seeds and class counts were checked.
- `collect_results.py`: parsed fake mmengine test logs and compared them with the targets.
- BenthiCat helpers: tiling matches BenthiCat's own `image2patches.py`; colour and id masks both decode.

**Not run yet:** actual training (`tools/train.py`, `seg/train_seg_baseline.py` only compile-checked).
The BenthiCat folder layout and mask encoding are assumed from the dataset paper; run the inspect tool first.
