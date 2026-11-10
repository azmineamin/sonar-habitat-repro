#!/usr/bin/env python3
"""SimCLR-style self-supervised pretraining of a ResNet-34 encoder on unlabelled
BenthiCat 'SSS Pre-training' tiles (384x384 float32 .npy, see pretrain_data.py).

The encoder uses the same module names as UNetR34 in train_seg_baseline.py
(`stem`, `enc`), so the saved `encoder.pt` can be loaded straight into the U-Net
(`train_seg_baseline.py --init-encoder encoder.pt`).

Views (done on GPU): random resized crop (scale 0.3-1), flips, gain/bias,
multiplicative speckle noise, additive Gaussian noise, random blur. No rotations
(sonar geometry is anisotropic).

Train/val split is by waterfall (or whole held-out sectors), never by tile.
Validation reports the contrastive loss and top-1 retrieval accuracy of the
two views of each tile, a cheap proxy for representation quality.

Usage (GPU):
    python seg/train_ssl.py --root /content/bc_pre --sectors N02 N03 N08 \
        --val-sectors N03 --iters 20000 --batch 128 --out runs/ssl_r34
Re-running the same command resumes from <out>/last.pt (a finished run has a DONE marker).
Smoke test: add --iters 20 --batch 8 --workers 0 --no-pretrained
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
from pretrain_data import SSLTileDataset, list_tiles, split_by_waterfall  # noqa


class Encoder(nn.Module):
    """ResNet-34 trunk with UNetR34's module names (stem, pool, enc)."""

    def __init__(self, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet34(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.enc = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):  # x: (B,1,H,W) in [0,1]
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        h = self.pool(self.stem(x))
        for layer in self.enc:
            h = layer(h)
        return h.mean(dim=(2, 3))  # (B,512)


class SimCLR(nn.Module):
    def __init__(self, pretrained=True, proj_dim=128):
        super().__init__()
        self.encoder = Encoder(pretrained)
        self.proj = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, proj_dim))

    def forward(self, x):
        return F.normalize(self.proj(self.encoder(x)), dim=1)


# ----------------------------------------------------------- augmentation
def gpu_view(x, out=384):
    """x: (B,1,H,W) float [0,1] on GPU -> augmented view, same size."""
    B, _, H, W = x.shape
    views = []
    for i in range(B):
        s = random.uniform(0.3, 1.0)
        ar = random.uniform(0.75, 1.33)
        h = min(H, max(32, int(round(H * (s * ar) ** 0.5))))
        w = min(W, max(32, int(round(W * (s / ar) ** 0.5))))
        y0 = random.randint(0, H - h)
        x0 = random.randint(0, W - w)
        v = x[i:i + 1, :, y0:y0 + h, x0:x0 + w]
        v = F.interpolate(v, size=(out, out), mode='bilinear', align_corners=False)
        if random.random() < 0.5:
            v = v.flip(-1)
        if random.random() < 0.5:
            v = v.flip(-2)
        views.append(v)
    v = torch.cat(views, 0)
    dev = v.device
    gain = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
    bias = torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.08, 0.08)
    speckle = 1.0 + 0.1 * torch.randn_like(v) * (torch.rand(B, 1, 1, 1, device=dev) < 0.5)
    v = v * gain * speckle + bias + 0.02 * torch.randn_like(v) * (torch.rand(B, 1, 1, 1, device=dev) < 0.5)
    if random.random() < 0.3:  # light blur for the whole batch
        k = torch.tensor([1., 2., 1.], device=dev)
        k2 = (k[:, None] * k[None, :]) / 16.0
        v = F.conv2d(F.pad(v, (1, 1, 1, 1), mode='reflect'), k2.view(1, 1, 3, 3))
    return v.clamp(0, 1)


def nt_xent(z1, z2, temp=0.2):
    """Symmetric NT-Xent. Returns loss and top-1 retrieval accuracy."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], 0)
    sim = z @ z.t() / temp
    sim.fill_diagonal_(float('-inf'))
    tgt = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    loss = F.cross_entropy(sim, tgt)
    acc = (sim.argmax(1) == tgt).float().mean()
    return loss, acc


@torch.no_grad()
def evaluate(model, loader, device, max_batches=20):
    model.eval()
    ls, accs = [], []
    for i, x in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        if x.shape[0] < 4:
            continue
        loss, acc = nt_xent(model(gpu_view(x)), model(gpu_view(x)))
        ls.append(loss.item()); accs.append(acc.item())
    model.train()
    return dict(loss=float(np.mean(ls)) if ls else None, top1=float(np.mean(accs)) if accs else None)


def infinite(loader):
    while True:
        for b in loader:
            yield b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='folder containing extracted sector folders')
    ap.add_argument('--sectors', nargs='*', default=None, help='only these sector folders (default: all under root)')
    ap.add_argument('--val-sectors', nargs='*', default=[], help='sectors held out entirely for validation')
    ap.add_argument('--val-frac', type=float, default=0.05, help='extra fraction of waterfalls held out')
    ap.add_argument('--max-tiles', type=int, default=0, help='debug: cap number of train tiles')
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--temp', type=float, default=0.2)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--allow-cpu', action='store_true', help='run on CPU (smoke tests only)')
    ap.add_argument('--stop-after', type=int, default=0, help='debug: exit after this step (simulates a crash)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-pretrained', action='store_true', help='random init instead of ImageNet weights')
    args = ap.parse_args()

    if args.iters < 6:
        sys.exit('--iters must be at least 6 (one-cycle LR schedule needs a few steps)')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and not args.allow_cpu:
        sys.exit('No GPU found. In Colab: Runtime > Change runtime type > T4 GPU, then re-run '
                 '(add --allow-cpu to force CPU, only sensible for tiny smoke tests).')

    root = Path(args.root)
    dirs = [root / s for s in args.sectors] if args.sectors else [root]
    tiles = [t for d in dirs for t in list_tiles(d)]
    if not tiles:
        sys.exit(f'no .npy tiles under {dirs}')
    train, val = split_by_waterfall(tiles, args.val_frac, args.seed, args.val_sectors)
    if args.max_tiles:
        random.Random(args.seed).shuffle(train); train = train[:args.max_tiles]
    print(f'{len(tiles)} tiles -> train {len(train)} / val {len(val)}; device {device}', flush=True)
    (out / 'split.json').write_text(json.dumps(dict(
        train_waterfalls=sorted({t[1]['waterfall'] for t in train}),
        val_waterfalls=sorted({t[1]['waterfall'] for t in val})), indent=1))

    ident = lambda a, rng: a  # noqa: E731  (augmentation happens on GPU)
    mk = lambda ts, shuffle: torch.utils.data.DataLoader(  # noqa: E731
        SSLTileDataset(ts, 1, 1, view_fn=ident), batch_size=args.batch, shuffle=shuffle,
        num_workers=args.workers, drop_last=shuffle, pin_memory=(device == 'cuda'),
        persistent_workers=args.workers > 0)
    tl = mk(train, True)
    vl = mk(val, False) if val else None

    model = SimCLR(pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters,
                                                pct_start=min(0.5, max(0.05, 3.0 / args.iters)))  # >=3 warm-up steps, even for tiny smoke runs
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    log, t0, start = [], time.time(), 0
    ck = out / 'last.pt'
    if (out / 'DONE').exists():
        print('DONE marker found, nothing to do:', out); return
    if ck.exists():
        c = torch.load(ck, map_location=device)
        model.load_state_dict(c['model']); opt.load_state_dict(c['opt'])
        sched.load_state_dict(c['sched']); scaler.load_state_dict(c['scaler'])
        start, log = c['step'], c.get('log', [])
        print(f'resumed from {ck} at step {start}', flush=True)
    it = infinite(tl)
    for step in range(start + 1, args.iters + 1):
        x = next(it).to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            z1, z2 = model(gpu_view(x)), model(gpu_view(x))
        loss, acc = nt_xent(z1.float(), z2.float(), args.temp)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if step % 50 == 0 or step == 1:
            print(f'it {step}/{args.iters} loss {loss.item():.4f} top1 {acc.item():.3f} '
                  f'lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s', flush=True)
        if step % args.eval_every == 0 or step == args.iters:
            rec = dict(step=step, train_loss=loss.item(), train_top1=acc.item())
            if vl is not None:
                rec['val'] = evaluate(model, vl, device)
            log.append(rec); print('eval', rec, flush=True)
            torch.save(model.encoder.state_dict(), out / 'encoder.pt')
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            scaler=scaler.state_dict(), step=step, log=log), ck.with_suffix('.tmp'))
            ck.with_suffix('.tmp').replace(ck)  # atomic: a disconnect mid-save cannot corrupt last.pt
            (out / 'log.json').write_text(json.dumps(log, indent=1))
        if args.stop_after and step >= args.stop_after and step < args.iters:
            print('stopping early (debug) at step', step); return
    (out / 'DONE').write_text('ok')
    print('done ->', out / 'encoder.pt')


if __name__ == '__main__':
    main()
