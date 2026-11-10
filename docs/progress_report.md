# Progress report: SSL pretraining on BenthiCat side-scan sonar

Status: working project log, 11 Nov 2026. Facts and numbers only; this is not text for submission.
Repo: `azmineamin/sonar-habitat-repro` (commit `27dfe11` at time of writing).

## 1. Project in one paragraph

CSE499A: reliability-guided semi-/self-supervised **habitat segmentation on side-scan sonar (SSS)**, evaluated on **unseen survey sectors**.
The idea extends RSOD's reliability scoring [2] from bounding boxes to per-pixel predictions, adds self-supervised pretraining on the unlabelled BenthiCat tiles [1], and compares against a supervised U-Net at 1/5/10/100 % labels (3 seeds).

## 2. What we did so far

| # | Step | Result |
|---|---|---|
| 1 | Reproduced RSOD on UATD (Colab, T4) with the repo's notebook and scripts | Pipeline works end to end. Only a **500-iteration test** was run (paper: 180k iterations), so numbers are not comparable |
| 2 | Listed the BenthiCat "SSS Pre-training" dataset via the Dataverse API (`doi:10.7910/DVN/2VCN7Y`) | 32 archive files, 59.03 GB, sectors N01-N10, N12, N15, N16, S01-S03 |
| 3 | Downloaded and extracted three sectors | N02 2,522 + N03 10,698 + N08 17,128 = **30,348 tiles** |
| 4 | Inspected the real tile format (N02, checked locally) | `.npy`, float32, 384x384, 1 channel, values in [0, 1] (already log-scaled), no masks, no NaNs. File name encodes sector, survey, timestamp, sonar side (ch0/ch1), tile row and column |
| 5 | Wrote the SSL tile loader (`seg/pretrain_data.py`) and tests | Train/validation split **by waterfall** (a waterfall is the long sonar strip the tiles are cut from), not by tile, because neighbouring tiles overlap |
| 6 | Wrote SimCLR pretraining (`seg/train_ssl.py`) with a ResNet-34 encoder | GPU augmentation (crops, flips, gain/bias, speckle and Gaussian noise, blur; no rotations), checkpoint resume, `--eval-only` |
| 7 | Added `--init-encoder` to the U-Net baseline (`seg/train_seg_baseline.py`) | Verified: the saved encoder loads into the U-Net with no unexpected or missing keys (decoder and head start untrained) |
| 8 | Ran SSL pretraining on Colab (T4) | See section 4 |

### How the SSL run was set up
- Data: sectors N02, N03, N08. **N03 held out entirely for validation** (unseen-sector check), plus 5 % of the remaining waterfalls. Result: 18,792 train / 11,556 validation tiles.
- Model: SimCLR, ResNet-34 (ImageNet init), projection head 512-512-128, NT-Xent loss (temperature 0.2), AdamW, one-cycle learning rate (peak 3e-4), mixed precision.
- Settings: 224 px crops, batch 128, **8,000 iterations**, about 89 minutes (0.66 s/iteration). Seed 1, one run.
- Output: `MyDrive/benthicat/ssl_r34_v224/encoder.pt`.

### Problems met and how they were handled
- **Dataverse 403 from Colab** in the first notebook version: browser headers and a fallback endpoint were added (earlier commit). Local download also works with a browser User-Agent.
- **Learning-rate schedule crash** on very short runs (division by zero): warm-up is now at least 3 steps; runs under 6 iterations are refused.
- **Run on CPU by mistake:** the script now stops unless a GPU is found.
- **Too slow:** 384 px took 1.75 s/iteration (about 10 h for 20,000 iterations), so training moved to 224 px crops and 8,000 iterations.
- **Misleading validation:** the first validation loader was not shuffled, so batches held near-identical neighbouring tiles and the score was artificially low (0.30-0.49). It now shuffles; the numbers in section 4 come from the fixed version.

## 3. How this fits the project

| Project component | Status | Evidence |
|---|---|---|
| Reliability idea from RSOD [2] | Baseline reproduced in short form only | Step 1 |
| SSL pretraining on unlabelled sonar [1] | **Done once**, proxy metric only | Steps 2-8 |
| Test on unseen sectors | Partly: N03 held out for the pretext task. The segmentation split (train N01, N02, N04, N07, N08, S01-S03; test N03, N05, N06, N09, N10, N13, N15, N16) is planned | `CLAUDE.md` |
| Supervised U-Net baseline, 1/5/10/100 % labels | Code exists, **never run on real data** | `seg/` was written from paper assumptions |
| Segmentation labels | The BenthiCat segmentation set is **not released** (paper under review). Fallbacks: SSS-CAM tiles or SeafloorAI | `CLAUDE.md` |
| Pixel-level reliability pseudo-labelling, ablations | Not started | |

The encoder is only useful if it beats ImageNet initialisation on a labelled segmentation task. That comparison has not been done, so no claim about usefulness is made yet.

## 4. Results and comparison with the papers

### Table 1: RSOD reproduction vs the paper [2]
COCO bbox mAP (%), UATD test set. Paper values are from RSOD Tables 1 and 2 (also in `targets/paper_targets.csv`).

| Method | Labels | Paper mAP | Ours |
|---|---|---|---|
| Faster R-CNN, labelled only | 1 % | 13.7 | not recorded here |
| Faster R-CNN, labelled only | 5 % | 22.9 | not recorded here |
| Faster R-CNN, labelled only | 10 % | 26.2 | not recorded here |
| RSOD | 1 % | 23.2 | not recorded here |
| RSOD | 5 % | 34.7 | not recorded here |
| RSOD | 10 % | 34.8 | not recorded here |

Our only run was a 500-iteration test, against 180k iterations in the paper, so it cannot be compared with these values even where a number exists. The numbers live in the Colab/Drive results and were not copied into the repository. A full-schedule run (`tools/collect_results.py` prints the ours-vs-paper table) is still needed.

### Table 2: SSL pretraining on BenthiCat vs the paper [1]
The BenthiCat paper (arXiv v2) describes the dataset and its split sizes (about 950,000 unlabelled tiles for self-supervised pretraining, about 36,000 annotated tiles with a roughly 30,000 / 7,000 train/test split). **It reports no segmentation or SSL results**, so there is no paper number to compare against.

| Quantity | Paper [1] | Ours |
|---|---|---|
| Pretrain tiles used | about 950,000 available | 30,348 (3 sectors) |
| Retrieval top-1, training batches | not reported | about 0.97 (last logged: 0.977) |
| Retrieval top-1, held-out N03 plus 5 % of other waterfalls (11,556 tiles, shuffled) | not reported | **0.523** (loss 3.02) |
| Chance level (1 match among about 255 candidates) | - | under 0.01 |
| Segmentation mIoU (SSL init vs ImageNet init) | not reported | not yet run |

How to read it:
- The score is a proxy: how often the two altered views of the same tile are matched to each other within a batch of 128. It shows the encoder separates tiles from a sector it never saw, far above chance. It does not show the encoder helps segmentation.
- Training (about 0.97) is higher than validation (0.52). The two are not strictly comparable (different batches and layer mode, and a different sector), so the size of the generalisation gap is uncertain.
- One run, one seed, 3 of 19 sectors, 8,000 iterations. No error bars.

## 5. Next steps
1. Find labelled SSS data (SSS-CAM or SeafloorAI), inspect its real layout, and fix `seg/` to it.
2. Supervised U-Net at 1/5/10/100 % labels, 3 seeds, once with `--init-encoder` and once with ImageNet weights. This is the comparison that decides whether SSL helps.
3. A full-schedule RSOD run if the detection reproduction is to be reported.
4. Add more sectors to the SSL run only if step 2 shows a benefit.

## References
[1] H. Rajani, V. Franchi, B. Martinez-Clavel Valles, R. Ramos, R. Garcia, N. Gracias, "BenthiCat: An opti-acoustic dataset for advancing benthic classification and habitat mapping," arXiv:2510.04876 [cs.CV], v2, Oct. 2025. https://arxiv.org/abs/2510.04876. Data: Harvard Dataverse, doi:10.7910/DVN/2VCN7Y (SSS Pre-training). The preprint is also listed as an ESSD discussion paper, essd-2026-565 (under review).

[2] C. Li, P. Guo, et al., "RSOD: Reliability-Guided Sonar Image Object Detection with Extremely Limited Labels," Proc. AAAI Conf. Artif. Intell., vol. 40, no. 8, pp. 6055-6063, 2026. doi:10.1609/aaai.v40i8.37529. arXiv:2601.12715. Check the full author list against the proceedings page before using this citation in the report.
