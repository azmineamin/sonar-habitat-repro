"""Torch-free tests for seg/pretrain_data.py. Run: python tests/test_pretrain_data.py"""
import sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'seg'))
from pretrain_data import list_tiles, load_tile, parse_tile_name, split_by_waterfall

def test_parse():
    d = parse_tile_name('N02_11_220807052300_xtf-CH12_batch0_ch1_r4_c2.npy')
    assert d == dict(sector='N02', waterfall='N02_11_220807052300_xtf-CH12', batch=0, ch=1, r=4, c=2), d
    d = parse_tile_name('N02_SS_05_220807111700_xtf-CH12_batch1_ch1_r4_c2.npy')
    assert d['waterfall'] == 'N02_SS_05_220807111700_xtf-CH12' and d['sector'] == 'N02'
    assert parse_tile_name('foo.npy') is None

def test_split_and_load():
    with tempfile.TemporaryDirectory() as t:
        for sec in ('N02', 'N03'):
            for wf in range(10):
                for b in range(2):
                    for r in range(3):
                        np.save(Path(t) / f'{sec}_{wf}_2208070{wf:05d}_xtf-CH12_batch{b}_ch0_r{r}_c0.npy',
                                np.random.rand(384, 384).astype(np.float32))
        tiles = list_tiles(t)
        assert len(tiles) == 120
        tr, va = split_by_waterfall(tiles, val_frac=0.1, seed=1)
        assert len(tr) + len(va) == 120 and va
        assert not ({x[1]['waterfall'] for x in tr} & {x[1]['waterfall'] for x in va})
        tr, va = split_by_waterfall(tiles, val_frac=0, val_sectors=['N03'])
        assert {x[1]['sector'] for x in va} == {'N03'} and {x[1]['sector'] for x in tr} == {'N02'}
        a = np.full((384, 384), np.nan, np.float32); a[0, 0] = 2.0
        np.save(Path(t) / 'bad.npy', a)
        b = load_tile(Path(t) / 'bad.npy')
        assert b.shape == (384, 384) and b.max() == 1.0 and np.isfinite(b).all()

if __name__ == '__main__':
    test_parse(); test_split_and_load(); print('ok')
