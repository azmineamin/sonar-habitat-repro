"""Shared BenthiCat helpers (no torch needed).

Class table from BenthiCat (Rajani et al., 2025), Appendix Table A.1.
ID 0 'undefined' is ignored in loss and metrics.
"""
from pathlib import Path

import numpy as np

CLASSES = [
    (0, 'undefined', (255, 255, 255)),
    (1, 'breakwaters', (0, 0, 0)),
    (2, 'artificial_reefs', (57, 48, 138)),
    (3, 'rocks', (107, 90, 58)),
    (4, 'corals', (127, 14, 138)),
    (5, 'sand_ripples', (252, 140, 4)),
    (6, 'fine_sediment', (235, 186, 101)),
    (7, 'cymodocea', (5, 247, 215)),
    (8, 'posidonia', (0, 136, 0)),
    (9, 'cables', (255, 0, 0)),
    (10, 'trawl_scars', (223, 116, 255)),
    (11, 'misc_objects', (140, 129, 247)),
]
IGNORE_INDEX = 255
NUM_CLASSES = len(CLASSES)  # 12 ids; id 0 is mapped to IGNORE_INDEX

IMG_EXT = ('.npy', '.png', '.tif', '.tiff', '.jpg')


def load_array(path):
    path = str(path)
    if path.endswith('.npy'):
        return np.load(path, allow_pickle=False)
    import cv2
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise IOError(f'cannot read {path}')
    if a.ndim == 3:
        a = a[..., ::-1]  # BGR -> RGB
    return a


def mask_to_ids(m):
    """Accept an id mask (H,W) or an RGB colour mask (H,W,3); return uint8 ids,
    with 'undefined' (0) and unknown colours set to IGNORE_INDEX."""
    if m.ndim == 3 and m.shape[2] >= 3:
        out = np.full(m.shape[:2], IGNORE_INDEX, np.uint8)
        rgb = m[..., :3].astype(np.int32)
        for cid, _, col in CLASSES:
            hit = np.all(rgb == np.array(col, np.int32), axis=-1)
            out[hit] = cid
    else:
        out = m.astype(np.int64)
        out = np.where((out >= 0) & (out < NUM_CLASSES), out, IGNORE_INDEX).astype(np.uint8)
    out[out == 0] = IGNORE_INDEX
    return out


def to_unit_float(img):
    """Waterfalls are released log-scaled to [0,1]; raw 12-bit or 8-bit inputs
    are normalised here so the model always sees [0,1]."""
    img = img.astype(np.float32)
    if img.ndim == 3:
        img = img[..., 0]
    mx = float(img.max()) if img.size else 1.0
    if mx > 1.0:
        img = np.log1p(img) / np.log1p(4095.0 if mx > 255 else 255.0)
    return np.clip(img, 0.0, 1.0)


def pair_files(img_dir, mask_dir):
    """Match images to masks by filename stem (masks may carry a suffix)."""
    imgs = sorted(p for p in Path(img_dir).rglob('*') if p.suffix.lower() in IMG_EXT)
    masks = {}
    for p in Path(mask_dir).rglob('*'):
        if p.suffix.lower() in IMG_EXT:
            stem = p.stem
            for suf in ('_mask', '_label', '_gt', '_seg', '_annotation'):
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
            masks[stem] = p
    pairs = [(i, masks[i.stem]) for i in imgs if i.stem in masks]
    return pairs, len(imgs), len(masks)


def tile_coords(H, W, size, stride):
    """Same scheme as BenthiCat's misc/image2patches.py: stride adjusted so the
    tiles evenly cover the whole waterfall."""
    ny = int(np.ceil(max(H - size, 0) / stride)) + 1
    nx = int(np.ceil(max(W - size, 0) / stride)) + 1
    sy = (H - size) // (ny - 1) if ny > 1 else 0
    sx = (W - size) // (nx - 1) if nx > 1 else 0
    return [(y * sy, x * sx) for y in range(ny) for x in range(nx)]
