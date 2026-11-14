"""Descriptive micro recall at pooled score thresholds; complete score ties.

Uses frozen toolkit-aligned global-max500 matches, NOT COCO per-class AR.
"""
import numpy as np
BUDGETS=[.5,1.,2.,5.,10.]
IOU=[.5,.75]

def compile_curve(scores,flags,images,gt_counts):
    scores=np.asarray(scores,dtype=np.float64);flags=np.asarray(flags,dtype=np.int8)
    images=np.asarray(images,dtype=np.int32);gt_counts=np.asarray(gt_counts,dtype=np.int64)
    if flags.shape!=(len(scores),2) or images.shape!=scores.shape or not np.isfinite(scores).all():raise ValueError('Recall input dimensions/values differ')
    if not np.isin(flags,[-1,0,1]).all():raise ValueError('Recall match flags differ')
    order=np.argsort(-scores,kind='stable');scores=scores[order]
    ends=np.r_[np.flatnonzero(scores[1:]!=scores[:-1]),len(scores)-1] if len(scores) else np.empty(0,dtype=np.int64)
    return {'scores':scores,'flags':flags[order],'images':images[order],'gt_counts':gt_counts,'ends':ends}

def curve_values(block,frequency,full=False):
    frequency=np.asarray(frequency,dtype=np.int64)
    if frequency.shape!=block['gt_counts'].shape or np.any(frequency<0):raise ValueError('Invalid recall frequency')
    n=int(frequency.sum());den=int(frequency@block['gt_counts'])
    if n==0 or den==0:return np.full((2,len(BUDGETS)),np.nan),[]
    weights=frequency[block['images']]
    result=np.zeros((2,len(BUDGETS)));curves=[]
    for t in range(2):
        tp=np.r_[0,np.cumsum(weights*(block['flags'][:,t]==1),dtype=np.int64)[block['ends']]]
        fp=np.r_[0,np.cumsum(weights*(block['flags'][:,t]==0),dtype=np.int64)[block['ends']]]
        for b,budget in enumerate(BUDGETS):
            k=np.searchsorted(fp,budget*n,side='right')-1
            result[t,b]=100*tp[k]/den
        if full:
            curves.append({'scores':[None]+block['scores'][block['ends']].tolist(),
                           'TP':tp.tolist(),'FP':fp.tolist(),'images':n,'normal_GT':den,
                           'recall_percent':(100*tp/den).tolist(),'FP_per_image':(fp/n).tolist()})
    return result,curves
