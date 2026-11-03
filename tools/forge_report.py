#!/usr/bin/env python3
"""Create (or refresh) a CoreWeave Forge / W&B report that shows ONLY the
reproduction results: ours vs. paper, test mAP per run, and training curves.
No system metrics (GPU memory, power) are included.

Runs are selected by job type, which scripts/run_uatd.sh and collect_results.py set:
    train   -> one run per training setting (loss, lr, validation mAP)
    eval    -> one '<run>-test' run per setting (test-set COCO mAP)
    summary -> 'results-summary' (ours-vs-paper tables, from collect_results.py --wandb)

Usage (needs WANDB_API_KEY or `wandb login`):
    pip install wandb wandb-workspaces
    python tools/forge_report.py --entity walidur-rahman01-north-south-university --project sonar-habitat-repro
"""
import argparse
import os

TITLE = 'RSOD reproduction on UATD (sonar object detection)'

INTRO = (
    'Reproduction of RSOD (Li et al., AAAI 2026): semi-supervised Faster R-CNN with reliability-guided '
    'pseudo-labels, against a labels-only Faster R-CNN, on the UATD forward-looking sonar dataset '
    '(7,600 train / 800 val / 800 test images, 10 classes). Metric: COCO bbox mAP@[.5:.95] on the test set. '
    'Paper targets (Table 1): labels-only 13.7 / 22.9 / 26.2 and RSOD 23.2 / 34.7 / 34.8 at 1 / 5 / 10 % labels. '
    'Runs whose name ends in _itN used a shortened N-iteration schedule (pipeline check) and are not '
    'comparable with the paper, which trains for 180k iterations.'
)


def build(entity, project):
    import wandb_workspaces.reports.v2 as wr

    def runset(name, job_type):
        return wr.Runset(entity=entity, project=project, name=name,
                         filters=f"Metric('JobType') == '{job_type}'")

    return wr.Report(
        entity=entity, project=project, title=TITLE,
        description='Ours vs. paper, test mAP per run, and training curves (no system metrics).',
        width='fluid',
        blocks=[
            wr.H1(TITLE),
            wr.P(INTRO),

            wr.H2('1. Results vs. the paper'),
            wr.P('Mean ± std over seeds. diff_vs_paper is filled only for full-schedule runs.'),
            wr.PanelGrid(runsets=[runset('summary', 'summary')], panels=[
                wr.WeavePanelSummaryTable(table_name='rsod_reproduction/summary'),
                wr.WeavePanelSummaryTable(table_name='rsod_reproduction/per_run'),
            ]),

            wr.H2('2. Test-set mAP per run'),
            wr.PanelGrid(runsets=[runset('test runs', 'eval')], panels=[
                wr.BarPlot(title='Test mAP @[.5:.95]', metrics=['coco/bbox_mAP'], orientation='h'),
                wr.BarPlot(title='Test mAP @0.5', metrics=['coco/bbox_mAP_50'], orientation='h'),
                wr.BarPlot(title='Test mAP, small objects', metrics=['coco/bbox_mAP_s'], orientation='h'),
            ]),

            wr.H2('3. Training curves'),
            wr.P('x-axis: training iteration. RSOD logs supervised (sup_*) and pseudo-label (unsup_*) losses separately.'),
            wr.PanelGrid(runsets=[runset('training runs', 'train')], panels=[
                wr.LinePlot(title='Total loss', x='iter', y=['loss'], smoothing_factor=0.6),
                wr.LinePlot(title='All loss terms', x='iter', metric_regex='.*loss.*'),
                wr.LinePlot(title='Learning rate', x='iter', y=['lr']),
            ]),

            wr.H2('4. Validation mAP during training'),
            wr.P('Evaluated every 4,000 iterations (absent for runs shorter than that). '
                 'RSOD reports the teacher model (teacher/coco/bbox_mAP).'),
            wr.PanelGrid(runsets=[runset('training runs', 'train')], panels=[
                wr.LinePlot(title='Validation mAP (labels-only)', y=['coco/bbox_mAP']),
                wr.LinePlot(title='Validation mAP (RSOD teacher)', y=['teacher/coco/bbox_mAP']),
                wr.LinePlot(title='Validation mAP (RSOD student)', y=['student/coco/bbox_mAP']),
            ]),
        ],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--entity', default=os.getenv('WANDB_ENTITY'))
    ap.add_argument('--project', default=os.getenv('WANDB_PROJECT'))
    ap.add_argument('--dry-run', action='store_true', help='build the report but do not save it')
    a = ap.parse_args()
    if not a.project:
        raise SystemExit('--project (or WANDB_PROJECT) is required')
    report = build(a.entity, a.project)
    if a.dry_run:
        print('built report with', len(report.blocks), 'blocks (not saved)')
        return
    report.save()
    print('report saved:', report.url)
    print('(Forge serves W&B pages under https://forge.coreweave.com/wandb/; wandb.ai links redirect there.)')


if __name__ == '__main__':
    main()
