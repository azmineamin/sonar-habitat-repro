#!/usr/bin/env python3
"""Convert the UATD dataset (Xie et al., Scientific Data 2022) to COCO JSON.

UATD ships as three folders, each with `image/` (BMP) and `annotation/` (XML):
    UATD_Training/  (7,600 pairs)
    UATD_Test_1/    (800 pairs)
    UATD_Test_2/    (800 pairs)

RSOD (AAAI 2026) reports a 7,600 / 800 / 800 train / val / test partition but
does not say which test folder is val. We ASSUME Test_1 -> val, Test_2 -> test
(override with --val-split / --test-split). Record your choice in the paper.

Output layout (what the RSOD configs expect):
    <out>/train/  <out>/val/  <out>/test/          image symlinks (or copies)
    <out>/annotations/instances_{train,val,test}.json

Usage:
    python tools/uatd_to_coco.py --uatd-root /data/UATD --out /data/uatd_coco
"""
import argparse
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_EXTS = ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')


def _num(node, *names):
    """Return the first numeric child value found under any of `names`."""
    for n in names:
        el = node.find(n)
        if el is not None and el.text is not None and el.text.strip():
            return float(el.text)
    return None


def parse_xml(path):
    root = ET.parse(path).getroot()
    size = root.find('size')
    w = h = None
    if size is not None:
        w = _num(size, 'width', 'w')
        h = _num(size, 'height', 'h')
    stem = None
    f = root.find('file')
    if f is not None:
        fn = f.find('filename')
        if fn is not None and fn.text:
            stem = Path(fn.text.strip()).stem
    if stem is None:
        fn = root.find('filename')
        stem = Path(fn.text.strip()).stem if fn is not None and fn.text else Path(path).stem
    objs = []
    for o in root.iter('object'):
        name_el = o.find('name')
        bb = o.find('bndbox')
        if name_el is None or bb is None:
            continue
        x1 = _num(bb, 'xmin', 'x1', 'left')
        y1 = _num(bb, 'ymin', 'y1', 'top')
        x2 = _num(bb, 'xmax', 'x2', 'right')
        y2 = _num(bb, 'ymax', 'y2', 'bottom')
        if None in (x1, y1, x2, y2):
            raise ValueError(f'{path}: unrecognised bndbox fields '
                             f'{[c.tag for c in bb]}')
        objs.append((name_el.text.strip(), x1, y1, x2, y2))
    return stem, w, h, objs


def _norm(s):
    return ''.join(ch for ch in s.lower() if ch.isalnum())


def _count(d, exts):
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)


def find_split_dirs(root, key):
    """Return (image_dir, annotation_dir) for a split such as 'Training' or 'Test_1'.

    Folder names inside the UATD zips are not fixed (image/images/Image,
    nested UATD_Training/UATD_Training/..., or no split folder at all when each
    zip was extracted to its own folder), so we search by CONTENT: under every
    folder whose name contains the split key, take the sub-folder holding the
    most .xml files as annotations and the one holding the most images as images.
    """
    k = _norm(key)
    root = Path(root)
    best = None
    for cand in [root] + [p for p in root.rglob('*') if p.is_dir()]:
        if k not in _norm(cand.name):
            continue
        dirs = [cand] + [p for p in cand.rglob('*') if p.is_dir()]
        ann = max(dirs, key=lambda d: _count(d, ('.xml',)))
        img = max(dirs, key=lambda d: _count(d, IMG_EXTS))
        n_xml, n_img = _count(ann, ('.xml',)), _count(img, IMG_EXTS)
        if n_xml and n_img and (best is None or n_xml > best[2] or
                                (n_xml == best[2] and len(cand.parts) < best[3])):
            best = (img, ann, n_xml, len(cand.parts))
    if best:
        return best[0], best[1]
    raise FileNotFoundError(
        f'No folder whose name contains "{key}" with images and .xml files under {root}')


def image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        return None, None


def write_png(src, dst_dir):
    """Lossless re-encode of a UATD BMP as PNG. UATD stores echo intensity in
    channel 0 and repeats it in channels 1-2 (Xie et al. 2022); if that holds the
    PNG is single-channel, otherwise RGB. Pixel values are never changed."""
    import numpy as np
    from PIL import Image
    dst = dst_dir / (src.stem + '.png')
    if dst.exists():
        return dst, None
    with Image.open(src) as im:
        a = np.asarray(im)
    if a.ndim == 3 and a.shape[2] >= 3 and (a[..., 0] == a[..., 1]).all() and (a[..., 0] == a[..., 2]).all():
        Image.fromarray(a[..., 0]).save(dst, optimize=False)
        return dst, True
    Image.fromarray(a).save(dst, optimize=False)
    return dst, False


def convert_split(img_dir, ann_dir, out_img_dir, cat2id, copy=False, strict=False, to_png=False):
    imgs_by_stem = {p.stem: p for p in img_dir.iterdir()
                    if p.suffix.lower() in IMG_EXTS}
    out_img_dir.mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    missing = 0
    rgb_kept = 0
    for xml_path in sorted(ann_dir.glob('*.xml')):
        stem, w, h, objs = parse_xml(xml_path)
        img_path = imgs_by_stem.get(stem) or imgs_by_stem.get(xml_path.stem)
        if img_path is None:
            missing += 1
            if strict:
                raise FileNotFoundError(f'No image for {xml_path}')
            continue
        if not w or not h:
            w, h = image_size(img_path)
        img_id = len(images) + 1
        if to_png:
            dst, gray = write_png(img_path, out_img_dir)
            rgb_kept += gray is False
            images.append(dict(id=img_id, file_name=dst.name, width=int(w), height=int(h)))
            for name, x1, y1, x2, y2 in objs:
                if name not in cat2id:
                    cat2id[name] = len(cat2id) + 1
                bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
                anns.append(dict(id=0, image_id=img_id, category_id=cat2id[name],
                                 bbox=[x1, y1, bw, bh], area=bw * bh, iscrowd=0))
            continue
        images.append(dict(id=img_id, file_name=img_path.name,
                           width=int(w), height=int(h)))
        dst = out_img_dir / img_path.name
        if not dst.exists():
            if copy:
                shutil.copy2(img_path, dst)
            else:
                try:
                    os.symlink(img_path.resolve(), dst)
                except OSError:  # e.g. Windows without symlink rights
                    shutil.copy2(img_path, dst)
        for name, x1, y1, x2, y2 in objs:
            if name not in cat2id:
                cat2id[name] = len(cat2id) + 1
            bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
            anns.append(dict(id=0, image_id=img_id, category_id=cat2id[name],
                             bbox=[x1, y1, bw, bh], area=bw * bh, iscrowd=0))
    if rgb_kept:
        print(f'  note: {rgb_kept} images had differing channels and were kept as RGB PNG')
    return images, anns, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--uatd-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--train-split', default='Training')
    ap.add_argument('--val-split', default='Test_1')
    ap.add_argument('--test-split', default='Test_2')
    ap.add_argument('--copy', action='store_true', help='copy images instead of symlinking')
    ap.add_argument('--to-png', action='store_true',
                    help='re-encode BMPs as lossless PNG (grayscale when channels are identical); '
                         'saves a lot of disk, e.g. on Google Drive')
    ap.add_argument('--strict', action='store_true', help='fail on XML without image')
    a = ap.parse_args(argv)

    out = Path(a.out)
    (out / 'annotations').mkdir(parents=True, exist_ok=True)
    cat2id = {}
    results = {}
    # Train first so category ids are defined by the training set.
    for split, key in (('train', a.train_split), ('val', a.val_split), ('test', a.test_split)):
        img_dir, ann_dir = find_split_dirs(a.uatd_root, key)
        print(f'{split:5s}: images {img_dir}  annotations {ann_dir}')
        d = img_dir.parent
        images, anns, missing = convert_split(img_dir, ann_dir, out / split, cat2id, a.copy, a.strict, a.to_png)
        results[split] = (images, anns)
        print(f'{split:5s} <- {d.name}: {len(images)} images, {len(anns)} boxes'
              + (f', {missing} XML without image (skipped)' if missing else ''))

    # Deterministic, alphabetical category ids (independent of file order).
    names = sorted(cat2id)
    remap = {cat2id[n]: i + 1 for i, n in enumerate(names)}
    categories = [dict(id=i + 1, name=n) for i, n in enumerate(names)]
    ann_id = 1
    for split, (images, anns) in results.items():
        for an in anns:
            an['id'] = ann_id
            an['category_id'] = remap[an['category_id']]
            ann_id += 1
        with open(out / 'annotations' / f'instances_{split}.json', 'w') as f:
            json.dump(dict(images=images, annotations=anns, categories=categories), f)
    print('classes:', [c['name'] for c in categories])
    if len(categories) != 10:
        print(f'WARNING: expected 10 UATD classes, found {len(categories)}', file=sys.stderr)


if __name__ == '__main__':
    main()
