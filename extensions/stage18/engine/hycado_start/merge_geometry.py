"""Deterministic, GT-free crop merging with bounds against ORIGINAL members."""
from __future__ import annotations

import heapq
import math


POLICIES = [
    {'name': 'overlap_z1p5', 'max_gap_ratio': 0., 'max_axis_growth': 1.5, 'max_pair_hull_ratio': 1.5},
    {'name': 'near_z2', 'max_gap_ratio': .25, 'max_axis_growth': 2., 'max_pair_hull_ratio': 1.5},
    {'name': 'near_z3', 'max_gap_ratio': .5, 'max_axis_growth': 3., 'max_pair_hull_ratio': 2.},
]


def cluster(rect, members, min_width=None, min_height=None):
    return {'rect': tuple(rect), 'members': tuple(sorted(members)),
            'min_width': rect[2]-rect[0] if min_width is None else min_width,
            'min_height': rect[3]-rect[1] if min_height is None else min_height}


def initial_clusters(regions):
    ordered = sorted(set(tuple(r) for r in regions))
    for r in ordered:
        if len(r) != 4 or not all(isinstance(v, int) for v in r) or r[2] <= r[0] or r[3] <= r[1]:
            raise ValueError('Crop kutusu pozitif tamsayi koordinatlariyla verilmeli')
    return [cluster(r, [i]) for i, r in enumerate(ordered)]


def join(a, b):
    x, y = a['rect'], b['rect']
    rect = (min(x[0],y[0]), min(x[1],y[1]), max(x[2],y[2]), max(x[3],y[3]))
    return cluster(rect, a['members']+b['members'], min(a['min_width'],b['min_width']),
                   min(a['min_height'],b['min_height']))


def pair_priority(a, b, policy):
    """Eligibility and ranking use crop geometry only, never GT labels."""
    x, y = a['rect'], b['rect']
    merged = join(a,b)
    r = merged['rect']
    growth = max((r[2]-r[0])/merged['min_width'], (r[3]-r[1])/merged['min_height'])
    if growth > policy['max_axis_growth']+1e-12:
        return None
    gap_x, gap_y = max(0,x[0]-y[2],y[0]-x[2]), max(0,x[1]-y[3],y[1]-x[3])
    gap = math.hypot(gap_x,gap_y)/min(merged['min_width'],merged['min_height'])
    if gap > policy['max_gap_ratio']+1e-12:
        return None
    area_x, area_y = (x[2]-x[0])*(x[3]-x[1]), (y[2]-y[0])*(y[3]-y[1])
    inter = max(0,min(x[2],y[2])-max(x[0],y[0]))*max(0,min(x[3],y[3])-max(x[1],y[1]))
    union = area_x+area_y-inter
    hull = (r[2]-r[0])*(r[3]-r[1])
    ratio = hull/union
    if ratio > policy['max_pair_hull_ratio']+1e-12:
        return None
    return growth, (hull-union)/union, gap


def merge_to_budgets(regions, policy, budgets=(5,10,20)):
    """One greedy trajectory; snapshots at budgets, or last feasible state.

    No source region is discarded to force a budget. Bounds use minimum original
    member width/height, preventing repeated local merges from hiding zoom loss.
    """
    if (not budgets or any(not isinstance(k,int) or k<1 for k in budgets) or
            not all(math.isfinite(float(policy[k])) for k in ['max_axis_growth','max_gap_ratio','max_pair_hull_ratio']) or
            policy['max_axis_growth']<1 or policy['max_gap_ratio']<0 or policy['max_pair_hull_ratio']<1):
        raise ValueError('Birlesim politikasi/hedefi gecersiz')
    initial = initial_clusters(regions)
    active = dict(enumerate(initial))
    by_rect = {c['rect']: i for i,c in active.items()}
    heap, serial, snapshots = [], len(active), {}

    def push(i,j):
        priority = pair_priority(active[i],active[j],policy)
        if priority is not None:
            heapq.heappush(heap, (*priority,min(i,j),max(i,j)))

    def capture():
        for budget in sorted(set(budgets),reverse=True):
            if budget not in snapshots and len(active)<=budget:
                snapshots[budget] = sorted(active.values(),key=lambda c:c['rect'])

    ids = list(active)
    for n,i in enumerate(ids):
        for j in ids[n+1:]:
            push(i,j)
    capture()
    while heap and len(active)>min(budgets):
        *_,i,j = heapq.heappop(heap)
        if i not in active or j not in active:
            continue
        a,b = active.pop(i),active.pop(j)
        del by_rect[a['rect']],by_rect[b['rect']]
        merged = join(a,b)
        # Equal hulls are a single inference region, with all memberships retained.
        duplicate = by_rect.pop(merged['rect'],None)
        if duplicate is not None:
            merged = join(merged,active.pop(duplicate))
        active[serial] = merged
        by_rect[merged['rect']] = serial
        for old in list(active):
            if old != serial:
                push(old,serial)
        serial += 1
        capture()
    final = sorted(active.values(),key=lambda c:c['rect'])
    for budget in budgets:
        snapshots.setdefault(budget,final)
    # Every original region belongs exactly once to an enclosing final region.
    for result in snapshots.values():
        memberships = [m for c in result for m in c['members']]
        if sorted(memberships) != list(range(len(initial))):
            raise RuntimeError('Crop kaybi veya cift uyelik')
        for c in result:
            r = c['rect']
            if max((r[2]-r[0])/c['min_width'],(r[3]-r[1])/c['min_height'])>policy['max_axis_growth']+1e-12:
                raise RuntimeError('Orijinal uyelere gore buyume siniri asildi')
            for m in c['members']:
                s = initial[m]['rect']
                if not (r[0]<=s[0] and r[1]<=s[1] and r[2]>=s[2] and r[3]>=s[3]):
                    raise RuntimeError('Birlesimde kaynak crop kapsami kayboldu')
    return snapshots


def projected_views(targets, regions, side=960):
    """Visible short side after square scale-fill, conditional on single-crop IoA90.

    This is a geometry proxy, not a detector accuracy or recognition threshold.
    Matches the square scale_fill=True input geometry of pinned RTDETRPredictor.
    """
    import numpy as np
    if side<=0:
        raise ValueError('Giris boyutu pozitif olmali')
    gt = np.asarray(targets,dtype=np.float64).reshape(-1,5)
    crops = np.asarray(regions,dtype=np.float64).reshape(-1,4)
    best = np.zeros(len(gt),dtype=np.float64)
    eligible = np.zeros(len(gt),dtype=bool)
    if len(gt) and len(crops):
        area = (gt[:,2]-gt[:,0])*(gt[:,3]-gt[:,1])
        sizes = crops[:,2:4]-crops[:,:2]
        if (area<=0).any() or (sizes<=0).any():
            raise ValueError('Projeksiyonda pozitif olmayan alan')
        inter = np.maximum(0,np.minimum(gt[:,None,2:4],crops[None,:,2:4])-
                           np.maximum(gt[:,None,:2],crops[None,:,:2]))
        sufficient = inter[:,:,0]*inter[:,:,1]/area[:,None] >= .9-1e-10
        visible_side = (inter*side/sizes[None,:,:]).min(axis=2)
        eligible = sufficient.any(axis=1)
        best = np.where(sufficient,visible_side,0).max(axis=1)
    return eligible,best


def view_counts(targets, regions, base_eligible, base_side, side=960):
    import numpy as np
    gt = np.asarray(targets,dtype=np.float64).reshape(-1,5)
    eligible, visible = projected_views(targets,regions,side)
    small = (gt[:,2]-gt[:,0])*(gt[:,3]-gt[:,1]) < 1024
    if (base_eligible & ~eligible).any():
        raise RuntimeError('Birlesimde IoA90 kapsami kaybi')
    counts = {}
    for name,mask in [('all',np.ones(len(gt),dtype=bool)),('small',small)]:
        counts[name] = {'targets':int(mask.sum()),'ioa90':int((eligible&mask).sum()),
                        **{f'visible_min{n}px':int((eligible&mask&(visible>=n-1e-9)).sum()) for n in [4,8,16]}}
    paired = small & base_eligible
    counts['small_view_ratios_vs_baseline'] = (visible[paired]/base_side[paired]).tolist()
    before,after = small & base_eligible & (base_side>=8-1e-9),small & eligible & (visible>=8-1e-9)
    counts['small_lost_min8px'] = int((before&~after).sum())
    counts['small_gained_min8px'] = int((~before&after).sum())
    return counts
