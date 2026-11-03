"""Pure geometry for proposal coverage, never an official detection evaluator."""
from __future__ import annotations

import math


def canonical_detections(rows, width, height):
    """Original-image xyxy, score, class; drop only clipped zero-area boxes."""
    if width <= 0 or height <= 0:
        raise ValueError('Goruntu boyutu gecersiz')
    clean, dropped = [], 0
    for row in rows:
        if len(row) != 6 or not all(math.isfinite(float(v)) for v in row):
            raise ValueError('Tahminde sonlu olmayan veya eksik alan')
        x1, y1, x2, y2, score, cls = map(float, row)
        if cls != int(cls) or not 0 <= cls < 10 or not 0 <= score <= 1:
            raise ValueError('Tahmin sinifi veya skoru gecersiz')
        if x2 < x1 or y2 < y1:
            raise ValueError('Tahmin koordinat sirasi ters')
        x1, x2 = max(0., min(width, x1)), max(0., min(width, x2))
        y1, y2 = max(0., min(height, y1)), max(0., min(height, y2))
        if x2 <= x1 or y2 <= y1:
            dropped += 1
            continue
        clean.append([x1, y1, x2, y2, score, int(cls)])
    clean.sort(key=lambda r: (-r[4], r[5], *r[:4]))
    return clean, dropped


def select_regions(detections, width, height, threshold, top_k, context_scale):
    """Top-K after score filter; symmetric context, integer bounds, exact dedup."""
    if not 0 <= threshold <= 1 or top_k < 1 or context_scale < 1:
        raise ValueError('Proposal ayari gecersiz')
    selected = [r for r in detections if r[4] >= threshold][:top_k]
    regions, seen = [], set()
    for r in selected:
        x1, y1, x2, y2 = r[:4]
        cx, cy = (x1+x2)/2, (y1+y2)/2
        half_w, half_h = (x2-x1)*context_scale/2, (y2-y1)*context_scale/2
        box = (max(0, math.floor(cx-half_w)), max(0, math.floor(cy-half_h)),
               min(width, math.ceil(cx+half_w)), min(height, math.ceil(cy+half_h)))
        if box[2] > box[0] and box[3] > box[1] and box not in seen:
            seen.add(box)
            regions.append(box)
    return regions


def union_area(rectangles):
    """Exact axis-aligned rectangle union by x-sweep, not sum of crop areas."""
    if not rectangles:
        return 0
    xs = sorted({x for r in rectangles for x in (r[0], r[2])})
    area = 0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((r[1], r[3]) for r in rectangles if r[0] < right and r[2] > left)
        length, start, end = 0, None, None
        for a, b in intervals:
            if start is None:
                start, end = a, b
            elif a > end:
                length += end-start
                start, end = a, b
            else:
                end = max(end, b)
        if start is not None:
            length += end-start
        area += (right-left)*length
    return area


def load_targets(text, width, height):
    """Derived clean YOLO targets; returns xyxy+class in original pixels."""
    result = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        values = [float(v) for v in line.split()]
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f'Hedef etiketi gecersiz, satir {number}')
        cls, cx, cy, w, h = values
        if cls != int(cls) or not 0 <= cls < 10 or w <= 0 or h <= 0:
            raise ValueError(f'Hedef sinifi/boyutu gecersiz, satir {number}')
        box = [(cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height]
        if min(box[0], box[1]) < -1e-5 or box[2] > width+1e-5 or box[3] > height+1e-5:
            raise ValueError('Hedef kutusu goruntu disinda')
        box = [max(0., box[0]), max(0., box[1]), min(width, box[2]), min(height, box[3])]
        result.append(box+[int(cls)])
    return result


def coverage_counts(targets, regions):
    """Class-agnostic spatial coverage by one crop; never matching/AP/recall."""
    import numpy as np
    gt = np.asarray(targets, dtype=np.float64).reshape(-1, 5)
    crops = np.asarray(regions, dtype=np.float64).reshape(-1, 4)
    n = len(gt)
    area = (gt[:, 2]-gt[:, 0])*(gt[:, 3]-gt[:, 1])
    if n and (area <= 0).any():
        raise ValueError('Pozitif olmayan hedef alani')
    if n and len(crops):
        wh = np.maximum(0, np.minimum(gt[:, None, 2:4], crops[None, :, 2:4])-
                        np.maximum(gt[:, None, :2], crops[None, :, :2]))
        best = ((wh[:, :, 0]*wh[:, :, 1])/area[:, None]).max(axis=1)
        centers = (gt[:, :2]+gt[:, 2:4])/2
        center = ((centers[:, None, :] >= crops[None, :, :2]).all(axis=2) &
                  (centers[:, None, :] < crops[None, :, 2:4]).all(axis=2)).any(axis=1)
    else:
        best, center = np.zeros(n), np.zeros(n, dtype=bool)

    def counts(mask):
        return {'targets': int(mask.sum()), 'center': int((center & mask).sum()),
                'ioa50': int(((best >= .5-1e-10) & mask).sum()),
                'ioa90': int(((best >= .9-1e-10) & mask).sum()),
                'ioa100': int(((best >= 1-1e-9) & mask).sum())}
    return {'all': counts(np.ones(n, dtype=bool)),
            'small_native_area_lt_1024': counts(area < 1024),
            'classes': {str(c): counts(gt[:, 4] == c) for c in range(10)}}
