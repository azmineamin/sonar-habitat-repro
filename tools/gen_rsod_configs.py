#!/usr/bin/env python3
"""Generate RSOD and supervised-baseline configs for UATD (and FSOD) splits.

RSOD's public repo (github.com/LICZ9/RSOD) only ships an FSOD-10% config. This
script derives the other settings from it by text substitution, so everything
except the dataset, label split and seed stays exactly as the authors released.

Configs are written to <rsod_repo>/projects/RSOD/configs/repro/.

Usage:
    python tools/gen_rsod_configs.py --rsod-repo ../RSOD \
        --dataset uatd --coco-train data/uatd_coco/annotations/instances_train.json \
        --percents 1 5 10 --seeds 1 2 3
    python tools/gen_rsod_configs.py --rsod-repo ../RSOD --dataset fsod \
        --percents 5 10 20 --seeds 1 2 3          # FSOD classes are taken from the repo

Notes on deviations between the paper and the released config (kept as released):
  * learning rate: paper text says 0.001, released config uses SGD lr=0.01.
  * metric: released config uses COCO bbox mAP@[.5:.95]; we also log mAP@.5.
"""
import argparse
import json
import re
from pathlib import Path

SRC_DATA = 'FSOD10_sonar_detection.py'
SRC_MODEL = 'FSOD10_faster-rcnn_r50_fpn_100_sonar-s1-p10.py'


def sub_once(pattern, repl, text, what):
    new, n = re.subn(pattern, repl, text, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError(f'Could not patch {what!r}; has the RSOD repo changed?')
    return new


def dataset_config(src_text, classes, env_var, default_root):
    t = src_text
    cls = '(' + ', '.join(repr(c) for c in classes) + (',)' if len(classes) == 1 else ')')
    t = sub_once(r"'classes':\s*\(.*?\),\s*\n\s*\}", f"'classes':\n        {cls},\n}}", t, 'classes')
    t = sub_once(r"os\.getenv\('RSOD_DATA_ROOT', 'data/sonar'\)",
                 f"os.getenv('{env_var}', '{default_root}')", t, 'data_root')
    return t


def rsod_config(src_text, ds_file, pct, seed, num_classes):
    t = src_text
    t = t.replace("'../../../configs/_base_/", "'../../../../configs/_base_/")
    t = sub_once(r"'FSOD10_sonar_detection\.py'", f"'{ds_file}'", t, 'dataset base')
    t = sub_once(r"labeled_dataset\.ann_file = '[^']*'",
                 f"labeled_dataset.ann_file = 'annotations/instances_{pct}train_seed{seed}.json'",
                 t, 'labeled ann')
    t = sub_once(r"unlabeled_dataset\.ann_file = '[^']*'",
                 f"unlabeled_dataset.ann_file = 'annotations/instances_{pct}unlabeled_seed{seed}.json'",
                 t, 'unlabeled ann')
    t += f"""
# ---- added by gen_rsod_configs.py ----
detector.roi_head.bbox_head.num_classes = {num_classes}
randomness = dict(seed={seed}, deterministic=False)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=20000, by_epoch=False,
                                     max_keep_ckpts=2, save_best='teacher/coco/bbox_mAP',
                                     rule='greater'))
"""
    return t


SUP_TEMPLATE = '''# Supervised-only Faster R-CNN baseline (RSOD Table 1, first row).
# Same detector, backbone, pipeline and schedule as RSOD, trained on the labeled subset only.
_base_ = [
    '../../../../configs/_base_/models/faster-rcnn_r50_fpn_RAW.py',
    '../../../../configs/_base_/default_runtime.py',
    '{ds_file}'
]

model = _base_.model
model.data_preprocessor = dict(
    type='DetDataPreprocessor', mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0],
    bgr_to_rgb=False, pad_size_divisor=32)
model.backbone = dict(
    type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3), frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=False), norm_eval=True, style='caffe',
    init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://detectron2/resnet50_caffe'))
# Plain (non-reliability) RoI head and losses.
model.roi_head.type = 'StandardRoIHead'
model.roi_head.bbox_head.num_classes = {num_classes}
model.roi_head.bbox_head.loss_cls = dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
model.roi_head.bbox_head.loss_bbox = dict(type='L1Loss', loss_weight=1.0)

sup_pipeline_plain = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=_base_.scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=_base_.color_space, aug_num=1),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(
    _delete_=True, batch_size=4, num_workers=4, persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type=_base_.dataset_type, data_root=_base_.data_root, metainfo=_base_.metainfo,
                 ann_file='annotations/instances_{pct}train_seed{seed}.json',
                 data_prefix=dict(img='train/'), filter_cfg=dict(filter_empty_gt=True, min_size=32),
                 pipeline=sup_pipeline_plain))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=180000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [dict(type='LinearLR', start_factor=0.25, by_epoch=False, begin=0, end=500)]
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
                     clip_grad=dict(max_norm=20, norm_type=2))
log_processor = dict(by_epoch=False)
randomness = dict(seed={seed}, deterministic=False)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=20000, by_epoch=False,
                                     max_keep_ckpts=2, save_best='coco/bbox_mAP', rule='greater'))
resume = True
'''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rsod-repo', required=True)
    ap.add_argument('--dataset', choices=['uatd', 'fsod'], required=True)
    ap.add_argument('--coco-train', help='instances_train.json (needed for uatd: class names)')
    ap.add_argument('--percents', type=float, nargs='+', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    a = ap.parse_args(argv)

    cfg_dir = Path(a.rsod_repo) / 'projects' / 'RSOD' / 'configs'
    out_dir = cfg_dir / 'repro'
    out_dir.mkdir(parents=True, exist_ok=True)
    data_src = (cfg_dir / SRC_DATA).read_text()
    model_src = (cfg_dir / SRC_MODEL).read_text()

    if a.dataset == 'uatd':
        if not a.coco_train:
            ap.error('--coco-train is required for uatd')
        cats = json.loads(Path(a.coco_train).read_text())['categories']
        classes = [c['name'] for c in sorted(cats, key=lambda c: c['id'])]
        env_var, default_root = 'UATD_DATA_ROOT', 'data/uatd_coco'
    else:
        m = re.search(r"'classes':\s*\((.*?)\),", data_src, re.S)
        classes = [s.strip().strip('"\'') for s in m.group(1).split(',') if s.strip()]
        env_var, default_root = 'FSOD_DATA_ROOT', 'data/FSOD'

    ds_file = f'{a.dataset}_dataset.py'
    (out_dir / ds_file).write_text(dataset_config(data_src, classes, env_var, default_root))
    written = [ds_file]
    for pct in a.percents:
        tag = f'{pct:g}'
        for s in a.seeds:
            name = f'rsod_{a.dataset}_{tag}pct_seed{s}.py'
            (out_dir / name).write_text(rsod_config(model_src, ds_file, tag, s, len(classes)))
            sup = f'sup_{a.dataset}_{tag}pct_seed{s}.py'
            (out_dir / sup).write_text(SUP_TEMPLATE.format(ds_file=ds_file, pct=tag, seed=s,
                                                           num_classes=len(classes)))
            written += [name, sup]
    print(f'{len(classes)} classes: {classes}')
    print(f'wrote {len(written)} files to {out_dir}')


if __name__ == '__main__':
    main()
