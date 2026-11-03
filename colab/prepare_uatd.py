#!/usr/bin/env python3
"""Download UATD from figshare, convert to lossless PNG + COCO on Drive, make
label splits, and copy everything to fast local disk. Called by notebook cell 4.

Safe to re-run:
  * downloads resume (wget -c) and are size-checked against figshare;
  * each zip is extracted into its own folder, once;
  * conversion runs on fast local disk; the result is stored on Drive as ONE tar
    file (thousands of small files can be left half-uploaded by a disconnect);
  * raw files are deleted ONLY after the conversion succeeded;
  * finished steps are skipped; later sessions just unpack the tar.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
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


def is_complete(root):
    """True if root holds train/val/test json files and every listed image."""
    try:
        for split in ('train', 'val', 'test'):
            d = json.loads((root / 'annotations' / f'instances_{split}.json').read_text())
            have = {p.name for p in (root / split).iterdir()}
            if not d['images'] or any(im['file_name'] not in have for im in d['images']):
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def save_archive(local, archive):
    """Store the converted dataset on Drive as ONE uncompressed tar (PNGs are already
    compressed). One large file uploads to Drive far more reliably than ~9,200 small
    ones, which a disconnect can leave half-synced."""
    tmp = archive.with_suffix('.tar.partial')
    print(f'  writing {archive} (one file, uploads reliably)', flush=True)
    with tarfile.open(tmp, 'w') as tf:
        tf.add(local, arcname=local.name)
    with open(tmp, 'rb+') as f:
        os.fsync(f.fileno())
    tmp.replace(archive)
    ok = archive.with_name(archive.name + '.ok')
    ok.write_text(str(archive.stat().st_size))
    print(f'  saved {archive.stat().st_size / 1e9:.2f} GB to Drive')


def archive_ok(archive):
    ok = archive.with_name(archive.name + '.ok')
    try:
        return archive.exists() and int(ok.read_text()) == archive.stat().st_size
    except (OSError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drive-data', required=True,
                    help='e.g. /content/drive/MyDrive/sonar-habitat/data/uatd_coco; the archive is '
                         'stored next to it as uatd_coco.tar')
    ap.add_argument('--local-data', default='/content/data/uatd_coco')
    ap.add_argument('--raw', default='/content/uatd_raw')
    ap.add_argument('--pcts', nargs='+', default=['5'])
    ap.add_argument('--seeds', nargs='+', default=['1'])
    a = ap.parse_args()
    drive, local, raw = Path(a.drive_data), Path(a.local_data), Path(a.raw)
    archive = drive.parent / (drive.name + '.tar')
    archive.parent.mkdir(parents=True, exist_ok=True)

    # 1. get the converted dataset onto local disk
    if is_complete(local):
        print('[1/3] dataset already on local disk')
    elif archive_ok(archive):
        print(f'[1/3] unpacking {archive} from Drive to local disk')
        local.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tf:
            tf.extractall(local.parent)
    elif is_complete(drive):  # older kit versions wrote loose files to Drive
        print('[1/3] copying loose dataset folder from Drive')
        shutil.copytree(drive, local, dirs_exist_ok=True)
        save_archive(local, archive)
    else:
        print('[1/3] downloading UATD from figshare (no complete copy on Drive)')
        ext = download(raw)
        print('  extracted layout:')
        show_tree(ext)
        print('[1/3] converting to lossless PNG + COCO on local disk', flush=True)
        try:
            sh([sys.executable, KIT / 'tools' / 'uatd_to_coco.py', '--uatd-root', ext,
                '--out', local, '--to-png'])
        except subprocess.CalledProcessError:
            sys.exit('Conversion failed. Raw files are KEPT in ' + str(raw) +
                     '; paste the layout above and the error so the converter can be fixed.')
        if not is_complete(local):
            sys.exit('Converted dataset is incomplete; raw files kept in ' + str(raw))
        save_archive(local, archive)
        shutil.rmtree(raw)
    if not is_complete(local):
        sys.exit(f'{local} is incomplete. Delete {archive} and re-run to rebuild it.')

    # 2. label splits: deterministic (fixed seeds), so they are simply re-made locally
    ann = local / 'annotations'
    need = [p for p in a.pcts for s in a.seeds if not (ann / f'instances_{p}train_seed{s}.json').exists()]
    if need:
        print('[2/3] making label splits')
        sh([sys.executable, KIT / 'tools' / 'make_ssod_splits.py', ann / 'instances_train.json',
            '--percents', *a.pcts, '--seeds', *a.seeds])
    else:
        print('[2/3] label splits already exist')

    # 3. report
    print('[3/3] checking', local)
    for split in ('train', 'val', 'test'):
        d = json.loads((ann / f'instances_{split}.json').read_text())
        n_files = sum(1 for _ in (local / split).iterdir())
        print(f'  {split}: {len(d["images"])} images in json, {n_files} files on disk, {len(d["annotations"])} boxes')
    print('classes:', [c['name'] for c in d['categories']])
    print('Drive copy:', archive, 'OK' if archive_ok(archive) else 'MISSING')
    print('READY  UATD_DATA_ROOT =', local)


if __name__ == '__main__':
    main()
