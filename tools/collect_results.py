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
NAME_RE = re.compile(r'(?P<method>rsod|sup)_(?P<dataset>uatd|fsod)_(?P<pct>[\d.]+)pct_seed(?P<seed>\d+)')
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
    a = ap.parse_args(argv)

    rows, groups = [], defaultdict(list)
    for d in sorted(Path(a.work_root).iterdir()):
        m = NAME_RE.search(d.name)
        if not d.is_dir() or not m:
            continue
        met = last_test_metrics(d.rglob('*.log'))
        if not met:
            continue
        get = lambda k: next((float(v) for kk, v in met.items() if kk.endswith(k)), None)
        mAP, mAP50, mAPs = get('coco/bbox_mAP'), get('coco/bbox_mAP_50'), get('coco/bbox_mAP_s')
        r = dict(m.groupdict(), mAP=mAP, mAP50=mAP50, mAP_small=mAPs, run=d.name)
        rows.append(r)
        groups[(r['method'], r['dataset'], r['pct'])].append(r)

    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'dataset', 'pct', 'seed', 'mAP', 'mAP50', 'mAP_small', 'run'])
        w.writeheader()
        w.writerows(rows)

    targets = load_targets()
    print(f'{"setting":24s} {"n":>2s} {"mAP (ours)":>14s} {"paper":>6s} {"diff":>6s}  {"mAP_s":>12s} {"paper":>6s}')
    for key in sorted(groups):
        g = groups[key]

        def fmt(vals):
            vals = [100 * v for v in vals if v is not None]
            if not vals:
                return '-', None
            mu = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return f'{mu:5.1f} ± {sd:4.1f}', mu

        s_map, mu = fmt([r['mAP'] for r in g])
        s_small, _ = fmt([r['mAP_small'] for r in g])
        tgt = targets.get((*key, 'mAP'))
        tgt_s = targets.get((*key, 'mAP_small'))
        diff = f'{mu - tgt:+5.1f}' if (mu is not None and tgt is not None) else '-'
        print(f'{"_".join(key)+"%":24s} {len(g):2d} {s_map:>14s} {tgt if tgt else "-":>6} {diff:>6s}  '
              f'{s_small:>12s} {tgt_s if tgt_s else "-":>6}')
    print(f'\nper-run rows written to {a.out}')


if __name__ == '__main__':
    main()
