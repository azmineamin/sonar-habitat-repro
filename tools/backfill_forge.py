#!/usr/bin/env python3
"""Upload finished runs that were trained WITHOUT Forge logging to CoreWeave Forge / W&B.

MMEngine keeps every logged scalar on disk (work_dir/<timestamp>/vis_data/<timestamp>.json),
so training curves (loss, lr, validation mAP) and the test-set mAP can be sent afterwards.
Each run is uploaded under the same name / id / job type that live logging would have used,
so the Forge report treats backfilled and live runs alike. No system metrics are uploaded.

A run is skipped if it is unfinished (no DONE), was logged live (FORGE_LOGGED), or was
already backfilled (FORGE_BACKFILLED).

Usage:
    python tools/backfill_forge.py --work-root /content/drive/MyDrive/sonar-habitat/work_dirs
"""
import argparse
import json
import os
import re
from pathlib import Path

NAME_RE = re.compile(r'^(?P<method>rsod|sup)_(?P<dataset>uatd|fsod)_(?P<pct>[\d.]+)pct_seed(?P<seed>\d+)'
                     r'(?:_it(?P<iters>\d+))?$')


def read_scalars(run_dir):
    """Return (train_rows, test_rows) from all vis_data json logs, oldest first."""
    train, test = [], []
    for ts_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        vis = ts_dir / 'vis_data'
        if not vis.is_dir():
            continue
        files = sorted(vis.glob(f'{ts_dir.name}.json')) or sorted(vis.glob('scalars.json'))
        rows = []
        for f in files:
            for line in f.read_text(errors='ignore').splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
        if any('loss' in r for r in rows):
            train += rows               # training session (also holds validation rows)
        else:
            test += [r for r in rows if any(k.endswith('bbox_mAP') for k in r)]
    return train, test


def upload(wandb, entity, project, name, job_type, meta, rows):
    run = wandb.init(entity=entity or None, project=project, id=name, name=name, resume='allow',
                     job_type=job_type, group=f"{meta['method']}_{meta['dataset']}_{meta['pct']}pct",
                     tags=[meta['method'], meta['dataset'], f"{meta['pct']}pct", f"seed{meta['seed']}", 'backfilled'],
                     settings=wandb.Settings(x_disable_stats=True))
    for i, r in enumerate(rows):
        data = {k: v for k, v in r.items() if k != 'step' and isinstance(v, (int, float))}
        if data:
            run.log(data, step=i)
    url = run.url
    run.finish()
    return url


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-root', required=True)
    ap.add_argument('--entity', default=os.getenv('WANDB_ENTITY'))
    ap.add_argument('--project', default=os.getenv('WANDB_PROJECT'))
    ap.add_argument('--dry-run', action='store_true', help='only list what would be uploaded')
    a = ap.parse_args()
    if not a.project and not a.dry_run:
        raise SystemExit('--project (or WANDB_PROJECT) is required')
    root = Path(a.work_root)
    if not root.is_dir():
        print('backfill: no runs yet'); return
    wandb = None
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = NAME_RE.match(d.name)
        if not m or not (d / 'DONE').exists():
            continue
        if (d / 'FORGE_LOGGED').exists() or (d / 'FORGE_BACKFILLED').exists():
            continue
        train, test = read_scalars(d)
        print(f'backfill {d.name}: {len(train)} training log rows, {len(test)} test rows')
        if a.dry_run:
            continue
        if wandb is None:
            import wandb as _w
            wandb = _w
        meta = m.groupdict()
        if train:
            print('   train ->', upload(wandb, a.entity, a.project, d.name, 'train', meta, train))
        if test:
            print('   test  ->', upload(wandb, a.entity, a.project, d.name + '-test', 'eval', meta, test[-1:]))
        (d / 'FORGE_BACKFILLED').touch()


if __name__ == '__main__':
    main()
