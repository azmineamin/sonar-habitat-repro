# CLAUDE.md: project context for Claude Code

CSE499A: self-supervised sonar image understanding for underwater habitat mapping (Nov 2026 to May 2027).
Target venue: Q1/Q2 journal or A/A* conference.

## Direction
Reliability-guided semi-/self-supervised **habitat segmentation on side-scan sonar** (BenthiCat).
It extends RSOD's reliability idea from boxes to per-pixel, adds SSL pretraining on unlabelled sonar,
and tests on **unseen survey sectors**. Baselines to beat: supervised U-Net at 1/5/10/100 % labels (3 seeds).

## Rules
- References must be 2025/2026 papers (IEEE, Nature, Springer, Elsevier, MDPI, Wiley, T&F, ACM, Hindawi, IGI, OUP, or A/A* conferences).
- The student writes all academic text (literature review, report) in their own words. Claude may fix facts, check numbers and shorten, not draft prose.
- Git commits end with the attribution lines given in the session reminder.

## Repo layout
- `colab/`: Colab notebooks and helpers.
  - `RSOD_reproduction_colab.ipynb` with `setup_env.sh`, `prepare_uatd.py`, `train.sh`, `gen_configs.sh`, `results.sh`
  - `BenthiCat_pretrain_inspect.ipynb`: list, download and inspect a few SSS Pre-training sectors
- `tools/`: `uatd_to_coco.py`, `make_ssod_splits.py`, `gen_rsod_configs.py`, `collect_results.py`, `forge_report.py`, `backfill_forge.py`
- `scripts/`: `run_uatd.sh` (resume-safe), `run_seg_baselines.sh`
- `seg/`: `benthicat_common.py`, `benthicat_inspect.py`, `pretrain_data.py` (SSL tile loader, waterfall-level split; tested with `tests/test_pretrain_data.py`), `train_ssl.py` (SimCLR ResNet-34 pretraining, writes `encoder.pt`; smoke-tested on CPU with synthetic tiles only, never on real data or GPU), `train_seg_baseline.py` (`--init-encoder encoder.pt` loads it; U-Net ResNet-34, 384x384, stride 192).
  **Written from paper assumptions. The real BenthiCat file layout and mask encoding are unverified. Never run on real data. Fix after inspecting real tiles.**
- `targets/paper_targets.csv`: RSOD paper numbers. `README.md`: deviations from the paper.

## Done
RSOD reproduction on UATD works end to end on Colab (T4), short 500-iteration test only. Numbers are not comparable with the paper (180k iterations).
Speed on a T4: about 1.3 s/iter supervised, about 3.4 s/iter RSOD.

## Datasets
| Dataset | Status |
|---|---|
| UATD | Downloaded and converted. 9,200 FLS images, 10 classes, 7,600/800/800 split (figshare 21331143). |
| BenthiCat Raw | Public, doi 10.7910/DVN/UMABNT. GIS habitat interpretations (last resort for labels). |
| BenthiCat SSS-CAM | Public, doi 10.7910/DVN/VZD5S6. Sonar tiles paired with camera photos. |
| BenthiCat SSS Pre-training | Public, doi 10.7910/DVN/2VCN7Y. About 1M unlabelled 384x384 tiles. One split 7z archive set per sector (N01..N16, S01..S03 seen so far; 59 GB in 32 archive files; N11, N13, N14 not listed). |
| BenthiCat SSS Seafloor Segmentation | **Not released** (paper under review). About 36k masked tiles, 12 classes. Sector-disjoint split: train N01, N02, N04, N07, N08, S01, S02, S03; test N03, N05, N06, N09, N10, N13, N15, N16. |
| SeafloorAI (NeurIPS 2024) | Labelled fallback. |

## Verified on real SSS Pre-training tiles (N02, 2,522 tiles; N08 has 17,128 and N03 10,698)
- Every tile is a `.npy`, float32, 384x384, single channel, values in [0, 1] (log-scaled already; ~0-4 % exact zeros). No masks, no images, no NaNs.
- Filename: `<sector>_<survey>_<yymmddhhmmss>_xtf-CH12_batch<b>_ch<0|1>_r<row>_c<col>.npy` (ch0/ch1 = sonar side, r/c = tile grid position in the waterfall). Split SSL train/val by waterfall or sector, never by random tile, because neighbouring tiles overlap in content.
- `seg/benthicat_common.py` `load_array` and `to_unit_float` already handle this format. Segmentation mask encoding is still unverified (no labelled data released).
- Local download from Dataverse works with a browser User-Agent (no 403); it is slow (~50-500 KB/s). 7-Zip is at `C:\Program Files-Zipz.exe`; system Python has no numpy (use `uv run --with numpy`).

## Next steps
1. Done: tile format inspected (see above). Cell 5 of `colab/BenthiCat_pretrain_inspect.ipynb` repeats that check on Colab.
2. Done: `seg/pretrain_data.py` loader. Done: `seg/train_ssl.py` (SimCLR). Next: run it on Colab (GPU) on N02/N03/N08, then compare against ImageNet init once labels exist. Fix `seg/` mask handling if labels arrive.
3. If no segmentation labels arrive, evaluate on SeafloorAI.
4. Supervised baseline at 1/5/10/100 % labels, 3 seeds.
5. SSL pretraining, then pixel-level reliability pseudo-labelling (consistency over flipped/scaled views), ablations, unseen-sector tests.

## Environment notes and pitfalls
- Colab: Python 3.10 venv (torch 2.0.0+cu118, mmcv 2.1.0, mmengine 0.10.6, setuptools pinned to 69.5.1). Use `!bash script.sh`, not `%%bash`. Do not mix `$VAR` and `{var}` in one `!` line.
- Training needs `MPLBACKEND=Agg` and `unset WANDB_SERVICE`.
- Store UATD on Drive as one tar (`uatd_coco.tar` plus `.ok`); loose files on Drive sync unreliably after disconnects.
- Re-running training resumes from the last checkpoint; finished runs have a `DONE` marker.
