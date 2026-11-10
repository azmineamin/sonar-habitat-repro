"""Data loading for BenthiCat 'SSS Pre-training' tiles (unlabelled, SSL).

Verified format (sectors N02, N03, N08): one .npy per tile, float32, 384x384,
values in [0, 1] (already log-scaled), no masks. File name:

    <sector>_<survey>_<yymmddhhmmss>_xtf-CH12_batch<b>_ch<0|1>_r<row>_c<col>.npy

The "waterfall id" is the part before `_batch`, e.g. `N02_11_220807052300`.
Tiles of one waterfall overlap in content, so train/val splits are made per
waterfall (or per sector), never per tile.

Everything except `SSLTileDataset` is torch-free so it can be tested anywhere.
"""
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

NAME_RE = re.compile(
    r'^(?P<sector>[NS]\d{2})_(?P<wf>.+?)_batch(?P<batch>\d+)_ch(?P<ch>\d)_r(?P<r>\d+)_c(?P<c>\d+)$')


def parse_tile_name(path):
    """Return dict(sector, waterfall, batch, ch, r, c) or None if the name does not match."""
    stem = Path(path).stem
    m = NAME_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()
    # waterfall id = everything before '_batch' (includes sector)
    wf = stem[: stem.index('_batch')]
    return dict(sector=d['sector'], waterfall=wf, batch=int(d['batch']),
                ch=int(d['ch']), r=int(d['r']), c=int(d['c']))


def list_tiles(root):
    """All .npy tiles under root (recursive), sorted, as list of (path, info)."""
    out = []
    for p in sorted(Path(root).rglob('*.npy')):
        info = parse_tile_name(p)
        if info is None:  # unknown naming: still usable, grouped by parent folder
            info = dict(sector=p.parent.name, waterfall=f'{p.parent.name}/{p.stem}',
                        batch=0, ch=0, r=0, c=0)
        out.append((p, info))
    return out


def split_by_waterfall(tiles, val_frac=0.05, seed=0, val_sectors=()):
    """Split tiles into (train, val) lists with no waterfall in both.

    val_sectors: whole sectors held out for validation (unseen-sector check).
    Remaining val tiles are drawn per waterfall: ceil(val_frac * n_waterfalls).
    """
    val_sectors = set(val_sectors)
    groups = defaultdict(list)
    for t in tiles:
        groups[t[1]['waterfall']].append(t)
    sec_of = {wf: g[0][1]['sector'] for wf, g in groups.items()}
    rest = sorted(wf for wf in groups if sec_of[wf] not in val_sectors)
    rng = random.Random(seed)
    rng.shuffle(rest)
    n_val = int(np.ceil(val_frac * len(rest))) if val_frac > 0 and rest else 0
    val_wf = set(rest[:n_val]) | {wf for wf in groups if sec_of[wf] in val_sectors}
    train = [t for wf, g in groups.items() if wf not in val_wf for t in g]
    val = [t for wf in sorted(val_wf) for t in groups[wf]]
    train.sort(key=lambda t: str(t[0]))
    return train, val


def load_tile(path):
    """Load one tile as float32 (384,384) in [0,1]. Corrupt/NaN values -> 0."""
    a = np.load(str(path), allow_pickle=False).astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] in (1, 3) else a[0]
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(a, 0.0, 1.0)


def valid_fraction(a):
    """Fraction of non-zero pixels (zeros are nadir gap / no-data)."""
    return float((a > 0).mean())


class SSLTileDataset:
    """Torch Dataset yielding one or two augmented views per tile.

    n_views=1: tensor (C,H,W). n_views=2: tuple of two tensors (for
    contrastive / consistency training). `view_fn(np_array) -> np_array` can be
    supplied for custom augmentation; the default is random flips plus mild
    gain/bias jitter (no rotations: sonar geometry is anisotropic).
    """

    def __init__(self, tiles, n_views=1, channels=1, view_fn=None, seed=None):
        import torch  # noqa: F401  (lazy: keeps the rest of this module torch-free)
        self.tiles = [t for t in tiles]
        self.n_views = n_views
        self.channels = channels
        self.view_fn = view_fn or default_view
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        import torch
        a = load_tile(self.tiles[i][0])
        views = []
        for _ in range(self.n_views):
            v = self.view_fn(a, self.rng)
            v = torch.from_numpy(np.ascontiguousarray(v))[None]
            if self.channels > 1:
                v = v.expand(self.channels, -1, -1).contiguous()
            views.append(v)
        return views[0] if self.n_views == 1 else tuple(views)


def default_view(a, rng):
    """Flips (both axes) plus mild gain/bias jitter, clipped to [0,1]."""
    if rng.random() < 0.5:
        a = a[:, ::-1]
    if rng.random() < 0.5:
        a = a[::-1, :]
    gain = rng.uniform(0.85, 1.15)
    bias = rng.uniform(-0.05, 0.05)
    return np.clip(a * gain + bias, 0.0, 1.0).astype(np.float32)
