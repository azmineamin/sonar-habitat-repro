#!/usr/bin/env python3
"""Collect test mAP from MMDetection logs and compare with the paper's numbers.

Scans <work_root>/<config_name>/**/*.log for the LAST test-time COCO line and
writes results.csv plus a printed table: mean +/- std over seeds vs. target.

Usage:
    python tools/collect_results.py --work-root work_dirs --out results.csv
"""
import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

TARGETS = Path(__file__).resolve().parent.parent / 'targets' / 'paper_targets.csv'
NAME_RE = re.compile(r'(?P<method>rsod|sup)_(?P<dataset>uatd|fsod)_(?P<pct>[\d.]+)pct_seed(?P<seed>\d+)'
                     r'(?:_it(?P<iters>\d+))?$')
FULL = 'full'  # paper schedule (180k iterations); shortened runs carry their iteration count
METRIC_RE = re.compile(r'(?:^|\s)((?:teacher/|student/)?coco/bbox_mAP(?:_50|_s)?):\s*([0-9.]+)')


def last_test_metrics(log_files):
    best = {}
    for lf in sorted(log_files):
        for line in lf.read_text(errors='ignore').splitlines():
            if 'Iter(test)' not in line and 'Epoch(test)' not in line:
                continue
            found = dict(METRIC_RE.findall(line))
            if found:
                best = found
    return best


def load_targets():
    t = {}
    if TARGETS.exists():
        with open(TARGETS) as f:
            for r in csv.DictReader(f):
                t[(r['method'], r['dataset'], r['pct'], r['metric'])] = float(r['value'])
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-root', default='work_dirs')
    ap.add_argument('--out', default='results.csv')
    ap.add_argument('--wandb', action='store_true',
                    help='also log the comparison table to W&B / CoreWeave Forge (needs WANDB_PROJECT)')
    a = ap.parse_args(argv)

    rows, groups = [], defaultdict(list)
    if not Path(a.work_root).is_dir():
        raise SystemExit(f'No runs found: {a.work_root} does not exist. Training has not started '
                         '(or failed before its first log); check the output of the training cell.')
    pending = []
    for d in sorted(Path(a.work_root).iterdir()):
        m = NAME_RE.search(d.name)
        if not d.is_dir() or not m:
            continue
        met = last_test_metrics(d.rglob('*.log'))
        if not met:
            state = 'checkpoint saved, test not run yet' if (d / 'last_checkpoint').exists() else 'no checkpoint yet'
            if (d / 'last_checkpoint').exists():
                try:
                    state += f" (last checkpoint: {Path((d / 'last_checkpoint').read_text().strip()).name})"
                except OSError:
                    pass
            pending.append((d.name, state))
            continue
        get = lambda k: next((float(v) for kk, v in met.items() if kk.endswith(k)), None)
        mAP, mAP50, mAPs = get('coco/bbox_mAP'), get('coco/bbox_mAP_50'), get('coco/bbox_mAP_s')
        r = dict(m.groupdict(), mAP=mAP, mAP50=mAP50, mAP_small=mAPs, run=d.name)
        r['iters'] = r['iters'] or FULL
        rows.append(r)
        groups[(r['method'], r['dataset'], r['pct'], r['iters'])].append(r)

    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'dataset', 'pct', 'seed', 'iters', 'mAP', 'mAP50', 'mAP_small', 'run'])
        w.writeheader()
        w.writerows(rows)

    targets = load_targets()
    summary = []
    print(f'{"setting":30s} {"n":>2s} {"mAP (ours)":>14s} {"paper":>6s} {"diff":>6s}  {"mAP_s":>12s} {"paper":>6s}')
    for key in sorted(groups):
        g = groups[key]
        method, dataset, pct, iters = key

        def stats(vals):
            vals = [100 * v for v in vals if v is not None]
            if not vals:
                return None, None
            return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

        mu, sd = stats([r['mAP'] for r in g])
        mu_s, sd_s = stats([r['mAP_small'] for r in g])
        tgt = targets.get((method, dataset, pct, 'mAP'))
        tgt_s = targets.get((method, dataset, pct, 'mAP_small'))
        comparable = iters == FULL
        diff = (mu - tgt) if (comparable and mu is not None and tgt is not None) else None
        fmt = lambda m, s: '-' if m is None else f'{m:5.1f} ± {s:4.1f}'
        label = f'{method}_{dataset}_{pct}%' + ('' if comparable else f' ({iters} it)')
        print(f'{label:30s} {len(g):2d} {fmt(mu, sd):>14s} {tgt if tgt else "-":>6} '
              f'{(f"{diff:+5.1f}" if diff is not None else "-"):>6s}  {fmt(mu_s, sd_s):>12s} {tgt_s if tgt_s else "-":>6}')
        summary.append(dict(method=method, dataset=dataset, label_pct=float(pct), schedule=iters, n_seeds=len(g),
                            mAP_mean=mu, mAP_std=sd, paper_mAP=tgt, diff_vs_paper=diff,
                            mAP_small_mean=mu_s, paper_mAP_small=tgt_s))
    if any(r['iters'] != FULL for r in rows):
        print('\nRows marked "(N it)" used a shortened schedule: they test the pipeline and are '
              'NOT comparable with the paper (diff left blank).')
    if pending:
        print('\nRuns without test results yet (re-run the training cell to resume them):')
        for name, state in pending:
            print(f'  {name}: {state}')
    if not rows:
        print('\nNo finished runs with test results yet - nothing to compare or upload.')
        return
    if a.wandb:
        log_to_wandb(summary, rows)
    print(f'\nper-run rows written to {a.out}')


def log_to_wandb(summary, rows):
    """One W&B run ('results-summary') holding the comparison tables, viewable in
    CoreWeave Forge under forge.coreweave.com/wandb/<entity>/<project>."""
    import os
    import wandb
    if not os.getenv('WANDB_PROJECT'):
        raise SystemExit('--wandb needs WANDB_PROJECT (and WANDB_API_KEY) in the environment')
    run = wandb.init(project=os.environ['WANDB_PROJECT'], entity=os.getenv('WANDB_ENTITY') or None,
                     name='results-summary', job_type='summary')
    cols = list(summary[0].keys()) if summary else []
    run.log({'rsod_reproduction/summary': wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in summary]),
             'rsod_reproduction/per_run': wandb.Table(
                 columns=['method', 'dataset', 'pct', 'seed', 'iters', 'mAP', 'mAP50', 'mAP_small', 'run'],
                 data=[[r[c] for c in ['method', 'dataset', 'pct', 'seed', 'iters', 'mAP', 'mAP50', 'mAP_small', 'run']]
                       for r in rows])})
    # Ours vs. paper, one bar pair per setting (full schedule only).
    bars = [[f"{r['method']} {r['label_pct']:g}% ({src})", v]
            for r in summary if r['schedule'] == FULL
            for src, v in (('ours', r['mAP_mean']), ('paper', r['paper_mAP'])) if v is not None]
    if bars:
        t = wandb.Table(columns=['setting', 'mAP'], data=bars)
        run.log({'rsod_reproduction/ours_vs_paper': wandb.plot.bar(t, 'setting', 'mAP', title='UATD mAP: ours vs. RSOD paper')})
    print('logged summary to W&B:', run.url)
    run.finish()


if __name__ == '__main__':
    main()
