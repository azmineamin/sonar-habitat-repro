#!/usr/bin/env python3
"""Make labeled / unlabeled splits of a COCO training set for semi-supervised
detection (STAC / Soft Teacher convention: sample a fraction of IMAGES).

RSOD samples 1/5/10 % (UATD) and 5/10/20 % (FSOD) of training images "randomly"
but gives no seeds, so we use several seeds and report mean +/- std.

Writes, next to the input file:
    instances_{pct}train_seed{s}.json      labeled subset (with boxes)
    instances_{pct}unlabeled_seed{s}.json  the remaining images (boxes kept,
                                           RSOD's unlabeled pipeline loads them
                                           but never uses them as targets)

Usage:
    python tools/make_ssod_splits.py data/uatd_coco/annotations/instances_train.json \
        --percents 1 5 10 --seeds 1 2 3
"""
import argparse
import json
import math
import random
from pathlib import Path


def split(coco, pct, seed, require_boxes=True):
    images = coco['images']
    has_box = {a['image_id'] for a in coco['annotations']}
    pool = [im for im in images if (im['id'] in has_box or not require_boxes)]
    n_lab = max(1, math.ceil(len(images) * pct / 100.0))
    rng = random.Random(seed)
    lab_ids = {im['id'] for im in rng.sample(pool, min(n_lab, len(pool)))}

    def subset(keep):
        ims = [im for im in images if keep(im['id'])]
        ids = {im['id'] for im in ims}
        anns = [a for a in coco['annotations'] if a['image_id'] in ids]
        return dict(images=ims, annotations=anns, categories=coco['categories'])

    return subset(lambda i: i in lab_ids), subset(lambda i: i not in lab_ids)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('train_json')
    ap.add_argument('--percents', type=float, nargs='+', default=[1, 5, 10])
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    a = ap.parse_args(argv)
    src = Path(a.train_json)
    coco = json.loads(src.read_text())
    for pct in a.percents:
        tag = f'{pct:g}'
        for s in a.seeds:
            lab, unl = split(coco, pct, s)
            (src.parent / f'instances_{tag}train_seed{s}.json').write_text(json.dumps(lab))
            (src.parent / f'instances_{tag}unlabeled_seed{s}.json').write_text(json.dumps(unl))
            print(f'{tag:>4}% seed {s}: {len(lab["images"])} labeled '
                  f'({len(lab["annotations"])} boxes), {len(unl["images"])} unlabeled')


if __name__ == '__main__':
    main()
