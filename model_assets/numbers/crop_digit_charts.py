"""
crop_digit_charts.py
----------------------
Cuts the two digit charts supplied by the site owner (one ISL 0-9 chart, one ASL 1-10 chart)
into one PNG per digit, for the Math page's digit guide:
  webapp/public/signs/num_isl/<0-9>.png   webapp/public/signs/num_asl/<1-10>.png
Usage: python crop_digit_charts.py <isl_chart.png> <asl_chart.png>   (run from the repo root)
Each row is split into hands by empty-column gaps; the printed digit label under each hand is dropped.
"""
import pathlib
import sys

import numpy as np
from PIL import Image

isl_path, asl_path = sys.argv[1], sys.argv[2]


def ink(a, thr=170):
    return a.min(2) < thr


def segments(mask_1d, min_gap):
    """Runs of True separated by >= min_gap False."""
    idx = np.where(mask_1d)[0]
    if len(idx) == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > min_gap:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return out


def crop_row(img, y0, y1, expect, min_gap, drop_label=True, asl=False):
    a = np.array(img.convert('RGB'))[y0:y1]
    cols = segments(ink(a).any(0), min_gap)
    cols = [c for c in cols if c[1] - c[0] > 25]
    assert len(cols) == expect, (len(cols), cols)
    pieces = []
    for (x0, x1) in cols:
        sub = a[:, x0:x1 + 1]
        if asl:
            # ASL chart: hands have long wrist lines; the printed label is a compact blob at the bottom.
            rows = segments(ink(sub).any(1), 3)
            rows = [r for r in rows if r[1] - r[0] > 4]  # ignore 1-4px specks (neighbouring row's edge)
            r0, r1 = rows[0][0], rows[-1][1]
            if drop_label and len(rows) > 1:
                last = rows[-1]
                if last[1] - last[0] <= 22 and last[0] > r0 + 0.65 * (r1 - r0):
                    r1 = rows[-2][1]
        else:
            # ISL chart: the label sits below a clear gap.
            rows = segments(ink(sub).any(1), 6)
            r0, r1 = rows[0][0], rows[0][1]
            if drop_label and len(rows) > 1:
                r1 = max(r[1] for r in rows[:-1])
            elif len(rows) > 1:
                r1 = rows[-1][1]
        pieces.append(Image.fromarray(sub[r0:r1 + 1]))
    return pieces


def save(pieces, digits, folder):
    out = pathlib.Path(f'webapp/public/signs/{folder}')
    out.mkdir(parents=True, exist_ok=True)
    for im, d in zip(pieces, digits):
        s = 300 / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        canvas = Image.new('RGB', (340, 340), 'white')
        canvas.paste(im, ((340 - im.width) // 2, (340 - im.height) // 2))
        canvas.save(out / f'{d}.png', optimize=True)


isl = Image.open(isl_path)
save(crop_row(isl, 10, 205, 6, 14), range(0, 6), 'num_isl')
save(crop_row(isl, 232, 415, 4, 14), range(6, 10), 'num_isl')
asl = Image.open(asl_path).convert('RGB')
# The ASL chart image has a dark frame and a browser "image search" badge on the bottom-left: trim / blank them.
asl = asl.crop((4, 8, 602, 331))  # measured: dark frame is 3px left, 7px right, 6px top, 14px bottom
_a = np.array(asl)
_a[int(asl.height * 0.78):, :34] = 255
asl = Image.fromarray(_a)
h = asl.height
save(crop_row(asl, 0, h // 2 + 6, 5, 8, asl=True), range(1, 6), 'num_asl')
save(crop_row(asl, h // 2 - 8, h, 5, 8, asl=True), range(6, 11), 'num_asl')
print('done')
