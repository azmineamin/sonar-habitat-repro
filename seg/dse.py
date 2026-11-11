#!/usr/bin/env python3
"""Label-free checkpoint selection for SSL encoders with DSE.

Implements the Dense representation Structure Estimator (DSE) and the DSE-based
model selection (Algorithm 1) of

    S. Dai, Q. Xu, P. Wen, Y. Liu, Q. Huang, "Exploring Structural Degradation in
    Dense Representations for Self-supervised Learning," NeurIPS 2025 (arXiv:2510.17299).

Idea: longer SSL training can hurt dense tasks such as segmentation (SDD). DSE scores
a checkpoint without any labels:

    DSE = M_inter - M_intra + lambda * M_dim            (higher is better)

  - pseudo-classes: k-means on dense feature vectors (one vector per spatial position)
  - M_intra: per cluster, sum of singular values of the centred vectors / sqrt(n_j - 1), averaged
  - M_inter: per vector, distance to the nearest OTHER cluster centroid, averaged
  - M_dim:   effective rank exp(entropy of normalised singular values) of vectors from
             different images
  - lambda = Std(M_inter - M_intra) / Std(M_dim) over all checkpoints

Choices the paper leaves open (ours, adjustable by flags; report them with the results):
feature layer (default layer4 = encoder output, stride 32), k (default 12, BenthiCat has
11 habitat classes + undefined), image counts, per-cluster SVD cap, selection-window edges.
We implement model selection only, not the DSE regulariser.

Usage:
    python seg/dse.py --root /content/bc_pre --sectors N02 N03 N08 --val-sectors N03 \
        --ckpt-dir /content/drive/MyDrive/benthicat/ssl_snap/snapshots --out dse.json
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ------------------------------------------------------- selection (torch-free)
def select_candidates(scores, window=2, top_t=3):
    """Algorithm 1: indices that are the maximum within [i-window, i+window] (window truncated at the
    ends), best `top_t` by score. Returns indices sorted by score, best first."""
    n = len(scores)
    peaks = [i for i in range(n) if scores[i] == max(scores[max(0, i - window): i + window + 1])]
    peaks.sort(key=lambda i: -scores[i])
    return peaks[:top_t]


def combine(inter_minus_intra, dim, lam=None):
    """DSE per checkpoint. lam=None -> Std(inter-intra)/Std(dim) over the checkpoints."""
    a = np.asarray(inter_minus_intra, np.float64)
    d = np.asarray(dim, np.float64)
    if lam is None:
        lam = float(a.std() / d.std()) if d.std() > 0 and len(a) > 1 else 1.0
    return (a + lam * d).tolist(), lam


# ------------------------------------------------------------------ torch part
def kmeans(x, k, iters=30, seed=0):
    import torch
    g = torch.Generator(device='cpu').manual_seed(seed)
    n = x.shape[0]
    c = x[torch.randperm(n, generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        assign = torch.cdist(x, c).argmin(1)
        newc = torch.zeros_like(c).index_add_(0, assign, x)
        cnt = torch.bincount(assign, minlength=k).clamp(min=1).unsqueeze(1).to(x.dtype)
        newc = newc / cnt
        empty = torch.bincount(assign, minlength=k) == 0
        if empty.any():  # re-seed empty clusters on random points
            newc[empty] = x[torch.randint(0, n, (int(empty.sum()),), generator=g).to(x.device)]
        if torch.allclose(newc, c, atol=1e-6):
            c = newc
            break
        c = newc
    return torch.cdist(x, c).argmin(1), c


def dse_terms(z_cls, z_dim, k=12, svd_cap=8192, seed=0):
    """z_cls: (n,d) dense vectors for the class terms; z_dim: (m,d) vectors from different images.
    Returns dict(m_inter, m_intra, m_dim)."""
    import torch
    assign, _ = kmeans(z_cls, k, seed=seed)
    ids = [j for j in range(k) if int((assign == j).sum()) >= 2]
    if len(ids) < 2:
        raise RuntimeError('fewer than 2 non-trivial clusters; features may have collapsed')
    mu = torch.stack([z_cls[assign == j].mean(0) for j in ids])           # (k',d)
    g = torch.Generator(device='cpu').manual_seed(seed)
    intra, inter = [], []
    dist = torch.cdist(z_cls, mu)                                         # (n,k')
    for pos, j in enumerate(ids):
        zj = z_cls[assign == j]
        if zj.shape[0] > svd_cap:
            zj = zj[torch.randperm(zj.shape[0], generator=g)[:svd_cap].to(zj.device)]
        zc = zj - zj.mean(0, keepdim=True)
        intra.append(torch.linalg.svdvals(zc).sum() / (zj.shape[0] - 1) ** 0.5)
        dj = dist[assign == j].clone()
        dj[:, pos] = float('inf')
        inter.append(dj.min(1).values.mean())
    s = torch.linalg.svdvals(z_dim)
    p = s / s.sum()
    m_dim = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())
    return dict(m_inter=float(torch.stack(inter).mean()), m_intra=float(torch.stack(intra).mean()),
                m_dim=float(m_dim))


def dense_features(encoder, x, layer='layer4'):
    """x: (B,1,H,W) in [0,1] -> (B, h*w, d) dense vectors from the chosen ResNet stage."""
    import torch
    x = (x.repeat(1, 3, 1, 1) - encoder.mean) / encoder.std
    h = encoder.pool(encoder.stem(x))
    for i, blk in enumerate(encoder.enc, start=1):
        h = blk(h)
        if f'layer{i}' == layer:
            break
    return h.flatten(2).transpose(1, 2)


def main():
    import torch
    from pretrain_data import list_tiles, load_tile, split_by_waterfall
    from train_ssl import Encoder

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--sectors', nargs='*', default=None)
    ap.add_argument('--val-sectors', nargs='*', default=[])
    ap.add_argument('--val-frac', type=float, default=0.05)
    ap.add_argument('--ckpt-dir', required=True, help='folder with enc_<step>.pt snapshots')
    ap.add_argument('--out', required=True, help='results .json')
    ap.add_argument('--layer', default='layer4', choices=['layer1', 'layer2', 'layer3', 'layer4'])
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--n-cls-images', type=int, default=1024, help='images whose every position feeds k-means / class terms')
    ap.add_argument('--n-dim-images', type=int, default=4096, help='images contributing one random position each to M_dim')
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--window', type=int, default=2)
    ap.add_argument('--top-t', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--allow-cpu', action='store_true')
    a = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and not a.allow_cpu:
        sys.exit('No GPU found (add --allow-cpu to force CPU; slow).')
    ckpts = sorted(Path(a.ckpt_dir).glob('enc_*.pt'), key=lambda p: int(re.findall(r'\d+', p.stem)[-1]))
    if len(ckpts) < 3:
        sys.exit(f'need at least 3 snapshots in {a.ckpt_dir}, found {len(ckpts)}')
    steps = [int(re.findall(r'\d+', p.stem)[-1]) for p in ckpts]

    root = Path(a.root)
    dirs = [root / s for s in a.sectors] if a.sectors else [root]
    tiles = [t for d in dirs for t in list_tiles(d)]
    train, _ = split_by_waterfall(tiles, a.val_frac, 1, a.val_sectors)   # same split as train_ssl (seed 1)
    rng = random.Random(a.seed)
    pick = rng.sample(train, min(a.n_dim_images, len(train)))
    n_cls = min(a.n_cls_images, len(pick))
    print(f'{len(pick)} train tiles ({n_cls} for class terms), {len(ckpts)} checkpoints, layer {a.layer}, k={a.k}, {device}', flush=True)
    X = torch.from_numpy(np.stack([load_tile(t[0]) for t in pick]))[:, None]   # (N,1,H,W) kept on CPU
    if a.size != X.shape[-1]:
        X = torch.nn.functional.interpolate(X, size=(a.size, a.size), mode='bilinear', align_corners=False)

    rows = []
    for step, path in zip(steps, ckpts):
        enc = Encoder(pretrained=False)
        enc.load_state_dict(torch.load(path, map_location='cpu'))
        enc = enc.to(device).eval()
        cls_feats, dim_feats = [], []
        g = torch.Generator(device='cpu').manual_seed(a.seed)
        with torch.no_grad():
            for i in range(0, len(X), 64):
                f = dense_features(enc, X[i:i + 64].to(device), a.layer).float()   # (B,P,d)
                if i < n_cls:
                    cls_feats.append(f[: max(0, n_cls - i)].reshape(-1, f.shape[-1]))
                pos = torch.randint(0, f.shape[1], (f.shape[0],), generator=g).to(device)
                dim_feats.append(f[torch.arange(f.shape[0], device=device), pos])
        t = dse_terms(torch.cat(cls_feats), torch.cat(dim_feats), a.k, seed=a.seed)
        t['step'] = step
        rows.append(t)
        print(f"step {step:>6}  inter {t['m_inter']:.3f}  intra {t['m_intra']:.3f}  dim {t['m_dim']:.1f}", flush=True)

    dse, lam = combine([r['m_inter'] - r['m_intra'] for r in rows], [r['m_dim'] for r in rows])
    for r, v in zip(rows, dse):
        r['dse'] = v
    cand = select_candidates(dse, a.window, a.top_t)
    result = dict(config=vars(a), lam=lam, checkpoints=rows,
                  candidates=[rows[i]['step'] for i in cand], best_step=rows[cand[0]]['step'])
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(f'lambda {lam:.4f}; DSE by step: ' + ', '.join(f"{r['step']}:{r['dse']:.2f}" for r in rows))
    print('candidates (best first):', result['candidates'], '-> best', result['best_step'])


if __name__ == '__main__':
    main()
