#!/usr/bin/env python3
"""Supervised habitat-segmentation baseline on BenthiCat side-scan sonar.

This is the "supervised, few labels" lower bound our semi-supervised method must
beat. A U-Net with an ImageNet ResNet-34 encoder is trained on a fraction of the
labelled TRAIN waterfalls (sampled per waterfall, not per tile, to avoid leakage
between overlapping tiles) and evaluated on the official TEST waterfalls, which
come from different coastal sectors (BenthiCat sec. 5.1).

Tiles: 384x384, stride 192 (as in BenthiCat / BenthicDINO).
Metrics: mIoU over classes present in the test set (id 0 'undefined' ignored),
pixel accuracy, per-class IoU -> <out>/metrics.json

Usage (GPU):
    python seg/train_seg_baseline.py --root /data/benthicat_seg \
        --train-img train/images --train-mask train/masks \
        --test-img test/images --test-mask test/masks \
        --label-frac 0.05 --seed 1 --out runs/seg_sup_5pct_seed1
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benthicat_common import (CLASSES, IGNORE_INDEX, NUM_CLASSES, load_array,  # noqa
                              mask_to_ids, pair_files, tile_coords, to_unit_float)


# ----------------------------------------------------------------- data
class TileDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, size=384, stride=192, train=True, cache=True):
        self.size, self.train = size, train
        self.items, self.cache = [], {}
        for k, (ip, mp) in enumerate(pairs):
            img = to_unit_float(load_array(ip))
            m = mask_to_ids(load_array(mp))
            if img.shape[:2] != m.shape[:2]:
                raise ValueError(f'size mismatch {ip.name}: {img.shape} vs {m.shape}')
            H, W = img.shape[:2]
            if H < size or W < size:  # pad tiny waterfalls
                ph, pw = max(0, size - H), max(0, size - W)
                img = np.pad(img, ((0, ph), (0, pw)))
                m = np.pad(m, ((0, ph), (0, pw)), constant_values=IGNORE_INDEX)
                H, W = img.shape
            if cache:
                self.cache[k] = (img, m)
            for (y, x) in tile_coords(H, W, size, stride):
                if (m[y:y + size, x:x + size] != IGNORE_INDEX).mean() > 0.05:
                    self.items.append((k, y, x))
        self.pairs = pairs

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        k, y, x = self.items[i]
        img, m = self.cache.get(k) or (to_unit_float(load_array(self.pairs[k][0])),
                                       mask_to_ids(load_array(self.pairs[k][1])))
        s = self.size
        t, l = img[y:y + s, x:x + s].copy(), m[y:y + s, x:x + s].copy()
        if self.train:
            if random.random() < 0.5:   # across-track flip (port <-> starboard)
                t, l = t[:, ::-1], l[:, ::-1]
            if random.random() < 0.5:   # along-track flip
                t, l = t[::-1], l[::-1]
            t = np.clip(t * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1), 0, 1)  # gain jitter
        t = torch.from_numpy(np.ascontiguousarray(t))[None].float()
        return t, torch.from_numpy(np.ascontiguousarray(l)).long()


# ---------------------------------------------------------------- model
class UNetR34(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet34(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)          # /2, 64
        self.pool = r.maxpool
        self.enc = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])  # /4../32
        chs = [64, 64, 128, 256, 512]
        self.dec = nn.ModuleList()
        for cin, cskip, cout in [(512, 256, 256), (256, 128, 128), (128, 64, 64), (64, 64, 64)]:
            self.dec.append(nn.Sequential(
                nn.Conv2d(cin + cskip, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)))
        self.head = nn.Conv2d(64, n_classes, 1)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        H, W = x.shape[-2:]
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        s0 = self.stem(x)
        feats, h = [s0], self.pool(s0)
        for layer in self.enc:
            h = layer(h)
            feats.append(h)
        h = feats[-1]
        for blk, skip in zip(self.dec, [feats[3], feats[2], feats[1], feats[0]]):
            h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            h = blk(torch.cat([h, skip], 1))
        return F.interpolate(self.head(h), size=(H, W), mode='bilinear', align_corners=False)


# --------------------------------------------------------------- metrics
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        v = y != IGNORE_INDEX
        conf += torch.bincount(y[v] * NUM_CLASSES + p[v], minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
    conf = conf.cpu().double()
    tp, gt, pr = conf.diag(), conf.sum(1), conf.sum(0)
    iou = tp / (gt + pr - tp).clamp(min=1)
    present = gt > 0
    names = {c[0]: c[1] for c in CLASSES}
    return dict(
        mIoU=float(iou[present].mean()),
        pixel_acc=float(tp.sum() / gt.sum().clamp(min=1)),
        per_class_iou={names[i]: round(float(iou[i]), 4) for i in range(NUM_CLASSES) if present[i]},
        test_pixels_per_class={names[i]: int(gt[i]) for i in range(NUM_CLASSES) if present[i]})


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--train-img', required=True); ap.add_argument('--train-mask', required=True)
    ap.add_argument('--test-img', required=True); ap.add_argument('--test-mask', required=True)
    ap.add_argument('--label-frac', type=float, default=1.0, help='fraction of TRAIN waterfalls labelled')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-pretrained', action='store_true')
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(a.root)
    train_pairs, _, _ = pair_files(root / a.train_img, root / a.train_mask)
    test_pairs, _, _ = pair_files(root / a.test_img, root / a.test_mask)
    if not train_pairs or not test_pairs:
        sys.exit('No image/mask pairs found; run seg/benthicat_inspect.py first.')
    n_lab = max(1, round(len(train_pairs) * a.label_frac))
    lab_pairs = random.Random(a.seed).sample(train_pairs, n_lab)
    (out / 'labelled_files.txt').write_text('\n'.join(p[0].name for p in sorted(lab_pairs)))

    tr, te = TileDataset(lab_pairs, train=True), TileDataset(test_pairs, train=False)
    print(f'labelled waterfalls {n_lab}/{len(train_pairs)} -> {len(tr)} tiles; test tiles {len(te)}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tl = torch.utils.data.DataLoader(tr, a.batch, shuffle=True, num_workers=a.workers, drop_last=len(tr) > a.batch,
                                     pin_memory=device == 'cuda', persistent_workers=a.workers > 0)
    el = torch.utils.data.DataLoader(te, a.batch, shuffle=False, num_workers=a.workers)
    model = UNetR34(NUM_CLASSES, pretrained=not a.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=device == 'cuda')

    it, best, t0, log = 0, None, time.time(), []
    while it < a.iters:
        for x, y in tl:
            model.train()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, enabled=device == 'cuda'):
                loss = F.cross_entropy(model(x), y, ignore_index=IGNORE_INDEX)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            it += 1
            if it % 100 == 0:
                print(f'iter {it}/{a.iters} loss {loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
            if it % a.eval_every == 0 or it == a.iters:
                m = evaluate(model, el, device); m['iter'] = it; log.append(m)
                print(f'  eval @ {it}: mIoU {100 * m["mIoU"]:.2f}  acc {100 * m["pixel_acc"]:.2f}', flush=True)
                torch.save(model.state_dict(), out / 'last.pth')
            if it >= a.iters:
                break
    final = log[-1]
    final.update(label_frac=a.label_frac, seed=a.seed, n_labelled_waterfalls=n_lab, train_tiles=len(tr),
                 note='final-iteration result on TEST (no test-set model selection)')
    (out / 'metrics.json').write_text(json.dumps(dict(final=final, history=log), indent=2))
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
