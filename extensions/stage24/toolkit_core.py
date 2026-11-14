"""Pinned original toolkit-aligned helpers, no training imports."""
import numpy as np
from visdrone_metrics import overlaps,precision_area
THRESHOLDS=np.linspace(.5,.95,10)
CAPS=(1,10,100,500)

def fast_flags(det, gt, threshold):
    """Equivalent greedy matching: normal first, last equal IoU, ignore unlimited."""
    result = np.zeros(len(det), dtype=np.int8)
    for cls in range(1, 11):
        did = np.flatnonzero(det[:, 5] == cls)
        target = gt[gt[:, 5] == cls, :5]
        target = target[np.argsort(target[:, 4], kind='stable')]
        state = -target[:, 4].astype(np.int8)
        matrix = overlaps(det[did, :5], target)
        for j, d in enumerate(did):
            available = np.flatnonzero((state == 0) & (matrix[j] >= threshold))
            if len(available):
                best = available[matrix[j, available] == matrix[j, available].max()][-1]
                state[best] = 1
                result[d] = 1
            elif np.any((state == -1) & (matrix[j] >= threshold)):
                result[d] = -1
    return result

def aggregate(parts, variant):
    denominator = np.sum([p['gt_denominators'] for p in parts], axis=0)
    presence = np.sum([p['presence'] for p in parts], axis=0)
    if presence.sum() == 0:
        raise ValueError('Hedef sinif yok')
    ap = np.zeros((10, 10)); ar = np.zeros((10, 10, 4)); per_class = []
    for c in range(1, 11):
        positions = [np.flatnonzero(p[variant + '_classes'] == c) for p in parts]
        for t in range(10):
            for j, cap in enumerate(CAPS):
                ids = [q[q < cap] for q in positions]
                scores = np.concatenate([p[variant + '_scores'][q] for p, q in zip(parts, ids)])
                flags = np.concatenate([p[variant + '_flags'][q, t] for p, q in zip(parts, ids)])
                area, recall = precision_area(scores, flags, int(denominator[c - 1]))
                ar[c - 1, t, j] = recall
                if j == 3: ap[c - 1, t] = area
        per_class.append({'class_id': c, 'gt_count_in_reference_denominator': int(denominator[c - 1]),
                          'image_presence_count': int(presence[c - 1]), 'AP': float(ap[c - 1].mean()),
                          'AP50': float(ap[c - 1, 0]), 'AP75': float(ap[c - 1, 5]),
                          'AR500': float(ar[c - 1, :, 3].mean())})
    weights = presence / presence.sum()
    output = {'AP': float(np.sum(ap.mean(axis=1) * weights)),
              'AP50': float(np.sum(ap[:, 0] * weights)), 'AP75': float(np.sum(ap[:, 5] * weights)),
              **{f'AR{cap}': float(np.sum(ar[:, :, j].mean(axis=1) * weights)) for j, cap in enumerate(CAPS)},
              'class_macro_AP_separate_diagnostic': float(ap[presence > 0].mean()), 'per_class': per_class}
    output['counts_at_fixed_max500'] = {}
    for t, key in [(0, '0.5'), (5, '0.75')]:
        flags = np.concatenate([p[variant + '_flags'][:, t] for p in parts])
        output['counts_at_fixed_max500'][key] = {
            'TP': int(np.count_nonzero(flags == 1)), 'FP': int(np.count_nonzero(flags == 0)),
            'ignored_detections': int(np.count_nonzero(flags == -1))}
    return output
