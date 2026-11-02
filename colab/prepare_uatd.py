#!/usr/bin/env python3
"""Download UATD from figshare, convert to lossless PNG + COCO on Drive, make
label splits, and copy everything to fast local disk. Called by notebook cell 4.

Safe to re-run:
  * downloads resume (wget -c) and are size-checked against figshare;
  * each zip is extracted into its own folder, once;
  * raw files are deleted ONLY after the conversion succeeded;
  * finished steps are skipped.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
ARTICLE = 21331143  # https://doi.org/10.6084/m9.figshare.21331143
WANTED = ('UATD_Training', 'UATD_Test_1', 'UATD_Test_2')  # UATD_OpenSLT is not used by RSOD


def sh(cmd):
    print('$', ' '.join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def show_tree(root, depth=3):
    root = Path(root)
    for p in sorted(root.rglob('*')):
        rel = p.relative_to(root)
        if p.is_dir() and len(rel.parts) <= depth:
            files = [f for f in p.iterdir() if f.is_file()]
            exts = {}
            for f in files:
                exts[f.suffix.lower()] = exts.get(f.suffix.lower(), 0) + 1
            print(f'   {rel}/  {len(files)} files {exts}')


def download(raw):
    zips, ext = raw / 'zips', raw / 'extracted'
    zips.mkdir(parents=True, exist_ok=True)
    ext.mkdir(parents=True, exist_ok=True)
    files = json.load(urllib.request.urlopen(f'https://api.figshare.com/v2/articles/{ARTICLE}/files'))
    for f in files:
        print(f"  {f['name']}: {f['size'] / 1e9:.2f} GB")
    for f in files:
        stem = Path(f['name']).stem
        if not stem.startswith(WANTED):
            print('  skipping', f['name'])
            continue
        out_dir = ext / stem
        if (out_dir / '.extracted').exists():
            continue
        z = zips / f['name']
        if not z.exists() or z.stat().st_size != f['size']:
            sh(['wget', '-q', '-c', '-O', z, f['download_url']])
        if z.stat().st_size != f['size']:
            sys.exit(f"{f['name']}: size {z.stat().st_size} != expected {f['size']}; re-run to resume.")
        print('  extracting', f['name'], flush=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out_dir)
        (out_dir / '.extracted').touch()
        z.unlink()  # zip no longer needed once extracted (extracted files are kept)
    return ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drive-data', required=True, help='e.g. /content/drive/MyDrive/sonar-habitat/data/uatd_coco')
    ap.add_argument('--local-data', default='/content/data/uatd_coco')
    ap.add_argument('--raw', default='/content/uatd_raw')
    ap.add_argument('--pcts', nargs='+', default=['5'])
    ap.add_argument('--seeds', nargs='+', default=['1'])
    a = ap.parse_args()
    drive, local, raw = Path(a.drive_data), Path(a.local_data), Path(a.raw)
    ann = drive / 'annotations'

    # 1. download + convert (once)
    if not (ann / 'instances_test.json').exists():
        print('[1/3] downloading UATD from figshare (first time only)')
        ext = download(raw)
        print('  extracted layout:')
        show_tree(ext)
        print('[1/3] converting to lossless PNG + COCO on Drive')
        try:
            sh([sys.executable, KIT / 'tools' / 'uatd_to_coco.py', '--uatd-root', ext,
                '--out', drive, '--to-png'])
        except subprocess.CalledProcessError:
            sys.exit('Conversion failed. Raw files are KEPT in ' + str(raw) +
                     '; paste the layout above and the error so the converter can be fixed.')
        shutil.rmtree(raw)
    else:
        print('[1/3] dataset already on Drive, skipping download')

    # 2. label splits (once per pct/seed)
    need = [p for p in a.pcts for s in a.seeds if not (ann / f'instances_{p}train_seed{s}.json').exists()]
    if need:
        print('[2/3] making label splits')
        sh([sys.executable, KIT / 'tools' / 'make_ssod_splits.py', ann / 'instances_train.json',
            '--percents', *a.pcts, '--seeds', *a.seeds])
    else:
        print('[2/3] label splits already exist')

    # 3. copy to local disk (training reads thousands of images; Drive is slow)
    print('[3/3] syncing to local disk', local)
    local.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in drive.rglob('*'):
        dst = local / src.relative_to(drive)
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif not dst.exists() or dst.stat().st_size != src.stat().st_size or src.suffix == '.json':
            shutil.copy2(src, dst)
            copied += 1
    print(f'  copied {copied} new/changed files')
    for split in ('train', 'val', 'test'):
        d = json.loads((local / 'annotations' / f'instances_{split}.json').read_text())
        n_files = sum(1 for _ in (local / split).iterdir())
        print(f'  {split}: {len(d["images"])} images in json, {n_files} files on disk, {len(d["annotations"])} boxes')
    print('classes:', [c['name'] for c in d['categories']])
    print('READY  UATD_DATA_ROOT =', local)


if __name__ == '__main__':
    main()
