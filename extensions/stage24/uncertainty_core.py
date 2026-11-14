"""Exact COCO 101-recall-point accumulation on cached per-image matches.

Canonical image-order replication, including stable equal-score image blocks.
Only maxDets=100 and area=all are bootstrapped; no new matching is performed.
"""
import numpy as np
from resampling import repeated_indices

KEYS=['AP','AP50','AP75','AR100']

def compile_category(entries,image_ids,cap=100):
    scores=[];flags=[];images=[];counts=np.zeros(len(image_ids),dtype=np.int64)
    for i,image in enumerate(image_ids):
        e=entries.get(image)
        if e is None:continue
        counts[i]=np.count_nonzero(~e['gtIgnore'])
        n=min(cap,len(e['dtScores']))
        scores.append(np.asarray(e['dtScores'][:n],dtype=np.float64))
        flags.append(np.where(e['dtIgnore'][:,:n].T,-1,np.where(e['dtMatches'][:,:n].T,1,0)).astype(np.int8))
        images.append(np.full(n,i,dtype=np.int32))
    scores=np.concatenate(scores) if scores else np.empty(0,dtype=np.float64)
    flags=np.concatenate(flags) if flags else np.empty((0,10),dtype=np.int8)
    images=np.concatenate(images) if images else np.empty(0,dtype=np.int32)
    order=np.argsort(-scores,kind='stable');scores=scores[order];flags=flags[order];images=images[order]
    start=np.r_[0,np.flatnonzero((scores[1:]!=scores[:-1])|(images[1:]!=images[:-1]))+1] if len(scores) else np.empty(0,dtype=np.int64)
    length=np.diff(np.r_[start,len(scores)]) if len(scores) else np.empty(0,dtype=np.int64)
    return {'flags':flags,'images':images,'counts':counts,
            'begin':np.repeat(start,length),'size':np.repeat(length,length)}

def category_values(block,frequency):
    frequency=np.asarray(frequency,dtype=np.int64)
    if frequency.shape!=block['counts'].shape or np.any(frequency<0):raise ValueError('Invalid bootstrap frequency')
    den=int(frequency@block['counts'])
    if den==0:return None
    ids=repeated_indices(block,frequency)
    if not len(ids):return np.zeros(4,dtype=np.float64)
    flags=block['flags'][ids];values=np.zeros(10);recall=np.zeros(10)
    for t in range(10):
        tp=np.cumsum(flags[:,t]==1,dtype=np.int64)
        fp=np.cumsum(flags[:,t]==0,dtype=np.int64)
        rc=tp/den
        pr=tp/(tp+fp+np.spacing(1))
        pr=np.maximum.accumulate(pr[::-1])[::-1]
        selected=np.searchsorted(rc,np.linspace(0,1,101),side='left')
        valid=selected<len(pr);q=np.zeros(101);q[valid]=pr[selected[valid]]
        values[t]=q.mean()*100;recall[t]=rc[-1]*100
    return np.array([values.mean(),values[0],values[5],recall.mean()])

def evaluate(engine,frequency):
    result=[]
    for method in engine:
        values=[v for block in method if (v:=category_values(block,frequency)) is not None]
        result.append(np.mean(values,axis=0) if values else np.full(4,np.nan))
    return np.asarray(result)
