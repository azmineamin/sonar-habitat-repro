#!/usr/bin/env python3
"""Create a tiny fake UATD tree (BMP + XML in the documented layout) for smoke tests."""
import random
import sys
from pathlib import Path

from PIL import Image

CLASSES = ['cube', 'ball', 'cylinder', 'human body', 'tyre', 'circle cage',
           'square cage', 'metal bucket', 'plane', 'rov']
XML = """<annotation>
  <sonar><range>14.9941</range><azimuth>120</azimuth><elevation>12</elevation>
    <soundspeed>1582.4</soundspeed><frequency>1200</frequency></sonar>
  <file><filename>{stem}</filename></file>
  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
{objs}</annotation>
"""
OBJ = """  <object><name>{n}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>
    <xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>
"""


def make(root, counts=(('UATD_Training', 200), ('UATD_Test_1', 20), ('UATD_Test_2', 20))):
    rng = random.Random(0)
    for split, n in counts:
        for sub in ('image', 'annotation'):
            (Path(root) / split / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            stem = f'{i + 1:05d}'
            w, h = 256, 180
            Image.new('RGB', (w, h), (rng.randint(0, 60),) * 3).save(Path(root) / split / 'image' / f'{stem}.bmp')
            objs = ''
            for _ in range(rng.randint(1, 3)):
                x1, y1 = rng.randint(0, 150), rng.randint(0, 100)
                objs += OBJ.format(n=rng.choice(CLASSES), x1=x1, y1=y1,
                                   x2=x1 + rng.randint(10, 80), y2=y1 + rng.randint(10, 60))
            (Path(root) / split / 'annotation' / f'{stem}.xml').write_text(
                XML.format(stem=stem, w=w, h=h, objs=objs))


if __name__ == '__main__':
    make(sys.argv[1])
