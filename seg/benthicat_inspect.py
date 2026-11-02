#!/usr/bin/env python3
"""Inspect a downloaded BenthiCat 'SSS Seafloor Segmentation' folder.

Run this FIRST after downloading. It reports the folder tree, file formats,
array shapes/dtypes/value ranges, whether masks are id- or colour-coded, and the
class pixel distribution, so the training script can be pointed at the right
folders (and fixed if the release format differs from what we assumed).

Usage:
    python seg/benthicat_inspect.py --root /data/benthicat_seg
    python seg/benthicat_inspect.py --root /data/benthicat_seg \
        --img-dir train/images --mask-dir train/masks
"""
import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benthicat_common import CLASSES, IGNORE_INDEX, load_array, mask_to_ids, pair_files  # noqa


def tree(root, depth=3, max_files=4):
    root = Path(root)
    for p in sorted(root.rglob('*')):
        rel = p.relative_to(root)
        if len(rel.parts) > depth:
            continue
        if p.is_dir():
            files = [f for f in p.iterdir() if f.is_file()]
            exts = collections.Counter(f.suffix.lower() for f in files)
            print(f'  {rel}/  ({len(files)} files {dict(exts)})')
            for f in sorted(files)[:max_files]:
                print(f'      {f.name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--img-dir')
    ap.add_argument('--mask-dir')
    ap.add_argument('--n', type=int, default=20, help='files to sample')
    a = ap.parse_args()

    print('== folder tree')
    tree(a.root)
    if not (a.img_dir and a.mask_dir):
        print('\nRe-run with --img-dir and --mask-dir (relative to --root) to check pairs.')
        return
    pairs, n_img, n_mask = pair_files(Path(a.root) / a.img_dir, Path(a.root) / a.mask_dir)
    print(f'\n== {n_img} images, {n_mask} masks, {len(pairs)} matched pairs')
    if not pairs:
        print('No pairs matched by filename stem; check naming.')
        return
    counts = collections.Counter()
    for img_p, m_p in pairs[: a.n]:
        img, m = load_array(img_p), load_array(m_p)
        kind = 'colour' if (m.ndim == 3 and m.shape[2] >= 3) else 'id'
        n_uni = len(np.unique(m.reshape(-1, m.shape[-1]), axis=0)) if kind == 'colour' else len(np.unique(m))
        print(f'  {img_p.name}: img {img.shape} {img.dtype} [{img.min():.3g},{img.max():.3g}] | '
              f'mask {m.shape} {m.dtype} ({kind}-coded, {n_uni} unique values)')
        if img.shape[:2] != m.shape[:2]:
            print('    !! image/mask size mismatch')
        ids = mask_to_ids(m)
        counts.update(dict(zip(*np.unique(ids, return_counts=True))))
    total = sum(counts.values())
    print('\n== class pixel share in sample')
    names = {c[0]: c[1] for c in CLASSES}
    for cid, n in sorted(counts.items()):
        nm = 'ignore/undefined/unknown colour' if cid == IGNORE_INDEX else names[cid]
        print(f'  {cid:3d} {nm:32s} {100 * n / total:6.2f}%')
    if counts.get(IGNORE_INDEX, 0) / total > 0.9:
        print('\n!! >90% ignored pixels: mask encoding probably differs from Table A.1; '
              'send this output so the mapping can be fixed.')


if __name__ == '__main__':
    main()
