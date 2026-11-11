"""Tests for seg/dse.py (needs torch; CPU is fine). Run: python tests/test_dse.py"""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'seg'))
from dse import combine, dse_terms, kmeans, select_candidates

def test_select():
    assert select_candidates([1, 3, 2, 5, 4, 6, 2, 2], window=2, top_t=3) == [5]
    s = [1, 5, 1, 1, 1, 1, 1, 4, 1, 1, 1]
    assert select_candidates(s, 2, 2) == [1, 7]
    assert select_candidates([1.0, 2.0, 3.0], 2, 3) == [2]          # monotone: best is the last one

def test_combine():
    d, lam = combine([1, 2, 3], [10, 20, 30])
    assert abs(lam - 0.1) < 1e-9 and np.allclose(d, [2, 4, 6])
    d, lam = combine([1, 2, 3], [5, 5, 5])                           # zero variance -> lam 1
    assert lam == 1.0

def blobs(sep, n=600, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(3, d, generator=g) * sep
    z = torch.cat([centers[i] + torch.randn(n, d, generator=g) for i in range(3)])
    return z

def test_kmeans_and_terms():
    z = blobs(10.0)
    a, c = kmeans(z, 3)
    assert sorted(torch.bincount(a).tolist()) == [600, 600, 600]
    sep = dse_terms(z, z[::3], k=3)
    mix = dse_terms(blobs(0.2), blobs(0.2)[::3], k=3)
    assert sep['m_inter'] - sep['m_intra'] > mix['m_inter'] - mix['m_intra'], (sep, mix)

def test_effective_rank():
    g = torch.Generator().manual_seed(0)
    iso = torch.randn(2000, 32, generator=g)
    low = torch.randn(2000, 1, generator=g) @ torch.randn(1, 32, generator=g) + 0.01 * torch.randn(2000, 32, generator=g)
    assert dse_terms(iso, iso, k=2)['m_dim'] > 20
    assert dse_terms(low, low, k=2)['m_dim'] < 3

if __name__ == '__main__':
    test_select(); test_combine(); test_kmeans_and_terms(); test_effective_rank(); print('ok')
