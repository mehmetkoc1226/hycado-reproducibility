"""Independent Python implementation of the pinned VisDrone DET toolkit rules.

Reference commit 005445782213e20cb91bc50a597db3dd949e749a.
This is a development diagnostic, not a certified official leaderboard score.
The reference implementation's class-image multiplicity and ignore denominator
are intentionally preserved. A conventional class macro average is separate.
"""
import csv
import math
import numpy as np

REFERENCE_COMMIT = '005445782213e20cb91bc50a597db3dd949e749a'


def raw_annotation(text):
    rows = []
    for fields in csv.reader(text.splitlines()):
        while fields and not fields[-1].strip(): fields.pop()
        if not fields: continue
        if len(fields) != 8: raise ValueError('Ham annotation 8 alan olmali')
        row = [float(x) for x in fields]
        if (not all(math.isfinite(x) for x in row) or row[4] not in (0, 1) or
                row[5] != int(row[5]) or not 0 <= row[5] <= 11):
            raise ValueError('Ham annotation sinif/skor/sayi gecersiz')
        rows.append(row)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 8)


def xyxy_to_det(rows):
    a = np.asarray(rows, dtype=np.float64).reshape(-1, 6).copy()
    a[:, 2:4] -= a[:, :2]
    a[:, 5] += 1
    return a


def ignore_integral(gt, width, height):
    regions = np.maximum(1, gt[gt[:, 5] == 0, :4])
    if not len(regions): return None
    if not np.equal(regions, np.floor(regions)).all():
        raise ValueError('Referans piksel ignore haritasi icin tamsayi geometri gerekli')
    mask = np.zeros((height, width), dtype=np.int32)
    for x, y, w, h in regions.astype(np.int64):
        # MATLAB 1-based inclusive rectangle; reference marks x through x+w.
        mask[y-1:min(height, y+h), x-1:min(width, x+w)] = 1
    np.cumsum(mask, axis=0, dtype=np.int32, out=mask)
    np.cumsum(mask, axis=1, dtype=np.int32, out=mask)
    return mask


def outside_ignore(rows, integral):
    if integral is None or not len(rows): return rows.copy()
    height, width = integral.shape
    raw = rows[:, :4]
    # MATLAB round: ties away from zero, then all four values clamped to >=1.
    q = np.maximum(1, np.sign(raw) * np.floor(np.abs(raw) + .5)).astype(np.int64)
    x = np.minimum(width, q[:, 0]) - 1
    y = np.minimum(height, q[:, 1]) - 1
    right = np.minimum(width - 1, x + q[:, 2])
    bottom = np.minimum(height - 1, y + q[:, 3])
    area = integral[y, x].astype(np.float64) + integral[bottom, right] - integral[y, right] - integral[bottom, x]
    return rows[area / (q[:, 2].astype(np.float64) * q[:, 3]) < .5].copy()


def prepare_groundtruth(raw, integral):
    gt = outside_ignore(raw[raw[:, 5] != 0], integral)
    gt[:, 4] = 1 - gt[:, 4]  # score 0 becomes class-specific ignored GT.
    return gt


def overlaps(det, gt):
    if not len(det) or not len(gt): return np.zeros((len(det), len(gt)))
    lo = np.maximum(det[:, None, :2], gt[None, :, :2])
    hi = np.minimum(det[:, None, :2] + det[:, None, 2:4], gt[None, :, :2] + gt[None, :, 2:4])
    inter = np.maximum(0, hi - lo).prod(axis=2)
    da = det[:, 2] * det[:, 3]
    ga = gt[:, 2] * gt[:, 3]
    union = np.where(gt[None, :, 4] == 1, da[:, None], da[:, None] + ga[None, :] - inter)
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_flags(det, gt, threshold):
    """Score-sorted detections; positive GT first, >= threshold, last equal tie."""
    order = np.argsort(gt[:, 4], kind='stable')
    gt = gt[order]
    ov = overlaps(det, gt)
    state = -gt[:, 4].astype(np.int8)
    flags = np.zeros(len(det), dtype=np.int8)
    for d in range(len(det)):
        best, chosen, matched = threshold, -1, 0
        for g in range(len(gt)):
            if state[g] == 1: continue
            if matched and state[g] == -1: break
            if ov[d, g] < best: continue
            best, chosen = ov[d, g], g
            matched = 1 if state[g] == 0 else -1
        flags[d] = matched
        if matched == 1: state[chosen] = 1
    return flags


def precision_area(scores, flags, denominator):
    if not len(scores): return 0., 0.
    idx = np.argsort(-scores, kind='stable')
    tp = np.cumsum(flags[idx] == 1)
    fp = np.cumsum(flags[idx] == 0)
    recall = tp / max(1, denominator)
    precision = tp / np.maximum(1, tp + fp)
    r = np.r_[0., recall, 1.]
    p = np.r_[0., precision, 0.]
    p = np.maximum.accumulate(p[::-1])[::-1]
    area = float(np.sum(np.diff(r)[np.diff(r) != 0] * p[1:][np.diff(r) != 0]))
    return area * 100, float(recall[-1]) * 100


def evaluate(images, variant):
    """images: stable image order, prepared gt and globally score-sorted det arrays.

    Prefix reuse for maxDets 1/10/100/500 is exact: greedy matching of earlier
    detections is independent of later detections. Cap BEFORE class filtering.
    """
    caps = (1, 10, 100, 500)
    ap = np.zeros((10, 10)); ar = np.zeros((10, 10, 4)); presence = np.zeros(10)
    per_class = []
    for cls in range(1, 11):
        print(f'AP hesaplama: {variant}, sinif {cls}/10', flush=True)
        data = []
        denominator = 0
        for image in images:
            gt = image['gt'][image['gt'][:, 5] == cls, :5]
            det = image[variant][:500]
            positions = np.flatnonzero(det[:, 5] == cls)
            dt = det[positions, :5]
            denominator += len(gt)
            presence[cls - 1] += bool(len(gt))
            data.append((gt, dt, positions))
        for t, thr in enumerate(np.linspace(.5, .95, 10)):
            score_lists = [[] for _ in caps]; flag_lists = [[] for _ in caps]
            for gt, dt, positions in data:
                flags = match_flags(dt, gt, float(thr))
                for j, cap in enumerate(caps):
                    keep = positions < cap
                    score_lists[j].append(dt[keep, 4]); flag_lists[j].append(flags[keep])
            for j in range(4):
                scores = np.concatenate(score_lists[j]) if data else np.empty(0)
                flags = np.concatenate(flag_lists[j]) if data else np.empty(0, dtype=int)
                area, recall = precision_area(scores, flags, denominator)
                ar[cls-1, t, j] = recall
                if j == 3: ap[cls-1, t] = area
        per_class.append({'class_id': cls, 'gt_count_in_reference_denominator': denominator,
                          'image_presence_count': int(presence[cls-1]), 'AP': float(ap[cls-1].mean()),
                          'AP50': float(ap[cls-1, 0]), 'AP75': float(ap[cls-1, 5]),
                          'AR500': float(ar[cls-1, :, 3].mean())})
    if not presence.sum(): raise ValueError('Degerlendirmede hedef sinif yok')
    weights = presence / presence.sum()
    active = presence > 0
    return {'scope': 'TRAIN internal development; provisional toolkit-aligned Python implementation',
            'reference_commit': REFERENCE_COMMIT, 'official_binary_parity_verified': False,
            'AP': float(np.sum(ap.mean(axis=1) * weights)),
            'AP50': float(np.sum(ap[:, 0] * weights)), 'AP75': float(np.sum(ap[:, 5] * weights)),
            **{f'AR{cap}': float(np.sum(ar[:, :, j].mean(axis=1) * weights)) for j, cap in enumerate(caps)},
            'class_macro_AP_separate_diagnostic': float(ap[active].mean()),
            'aggregation': 'literal reference evalClass repeated once per image containing that class',
            'ignored_GT_denominator': 'literal reference numel(gtMatch), including class-specific ignored GT',
            'units': 'percent', 'per_class': per_class}
