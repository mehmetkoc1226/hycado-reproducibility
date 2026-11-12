"""Paired frequency bootstrap of the unchanged reference-aligned AP statistic.

Matching is per image and can be reused. Scores are sorted once. Repeated
copies preserve stable ordering within (score, image) ties exactly as literal
canonical-image-order sample expansion; scores/ties are never approximated.
"""
from __future__ import annotations
from pathlib import PurePosixPath
import re
import numpy as np

KEYS=['AP','AP50','AP75','AR500']
NAMES=['baseline','rtdetr_full960','fixed_grid_a025','hybrid_a025',
       'relocated_s0_a025','relocated_s1_a025','relocated_s2_a025']
MODES=['cluster','image']


def make_groups(records):
    parents={};by_hash={};initial=[]
    def find(x):
        parents.setdefault(x,x)
        if parents[x]!=x:parents[x]=find(parents[x])
        return parents[x]
    def join(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    for rec in records:
        stem=PurePosixPath(rec['image']).stem
        group=stem.split('_',1)[0] if re.fullmatch(r'\d{7}_\d{5}_d_\d{7}',stem) else stem
        initial.append(group);find(group)
        if rec['image_sha256'] in by_hash:join(group,by_hash[rec['image_sha256']])
        else:by_hash[rec['image_sha256']]=group
    labels=sorted({find(x) for x in initial});ids={x:i for i,x in enumerate(labels)}
    duplicate={}
    for rec in records:duplicate.setdefault(rec['image_sha256'],[]).append(rec['image'])
    return {'labels':labels,'image_to_group':[ids[find(x)] for x in initial],
        'initial_filename_groups':len(set(initial)),'groups_after_hash_union':len(labels),
        'duplicate_image_hash_groups':[{'image_sha256':k,'images':v} for k,v in duplicate.items() if len(v)>1],
        'rule':'first filename token for standard DET names, otherwise full stem; exact duplicate image hashes union groups',
        'verified_video_or_scene_identity':False,'scope':'filename/hash proxy clusters; independence across clusters is unverified; all original records retained'}


def compile_engine(parts,names=NAMES):
    if not parts:raise ValueError('23 bos eslesme listesi')
    engine={'names':list(names),'counts':np.array([x['gt_denominators'] for x in parts],dtype=np.int64),
            'presence':np.array([x['presence'] for x in parts],dtype=np.int64),'classes':{}}
    for name in names:
        classes=[]
        for c in range(1,11):
            ids=[np.flatnonzero(x[name+'_classes']==c) for x in parts]
            scores=np.concatenate([x[name+'_scores'][q] for x,q in zip(parts,ids)])
            flags=np.concatenate([x[name+'_flags'][q] for x,q in zip(parts,ids)],axis=0)
            images=np.concatenate([np.full(len(q),i,dtype=np.int32) for i,q in enumerate(ids)])
            order=np.argsort(-scores,kind='stable');scores=scores[order];flags=flags[order];images=images[order]
            if len(scores):
                start=np.r_[0,np.flatnonzero((scores[1:]!=scores[:-1])|(images[1:]!=images[:-1]))+1]
                length=np.diff(np.r_[start,len(scores)])
                begin=np.repeat(start,length);size=np.repeat(length,length)
            else:begin=size=np.empty(0,dtype=np.int64)
            classes.append({'flags':flags,'images':images,'begin':begin,'size':size})
        engine['classes'][name]=classes
    return engine


def repeated_indices(block,frequency):
    weights=frequency[block['images']]
    repeated=np.repeat(np.arange(len(weights)),weights)
    if not len(repeated):return repeated
    offset=np.r_[0,np.cumsum(weights)]
    begin=block['begin'][repeated]
    # A tied image block [TP, FP] copied twice is TP,FP,TP,FP, not TP,TP,FP,FP.
    return begin+(np.arange(len(repeated))-offset[begin])%block['size'][repeated]


def evaluate(engine,frequency):
    frequency=np.asarray(frequency)
    if frequency.shape!=(len(engine['counts']),) or not np.issubdtype(frequency.dtype,np.integer) or np.any(frequency<0):raise ValueError('23 bootstrap frekansi gecersiz')
    # Compact persisted uint16 draws must not promote signed offset arithmetic to floats.
    frequency=frequency.astype(np.int64,copy=False)
    denominator=frequency@engine['counts'];presence=frequency@engine['presence']
    if presence.sum()==0:return np.full((len(engine['names']),4),np.nan)
    weights=presence/presence.sum();result=np.zeros((len(engine['names']),4))
    for i,name in enumerate(engine['names']):
        for c,block in enumerate(engine['classes'][name]):
            if weights[c]==0:continue
            flags=block['flags'][repeated_indices(block,frequency)]
            if not len(flags):continue
            positive=flags==1;tp=np.cumsum(positive,axis=0,dtype=np.int64)
            fp=np.cumsum(flags==0,axis=0,dtype=np.int64)
            precision=tp/np.maximum(1,tp+fp)
            envelope=np.maximum.accumulate(precision[::-1],axis=0)[::-1]
            den=max(1,int(denominator[c]))
            ap=np.sum(np.where(positive,envelope,0.),axis=0)/den*100
            ar=tp[-1]/den*100
            result[i]+=weights[c]*np.array([ap.mean(),ap[0],ap[5],ar.mean()])
    return result


def frequencies(groups,mode,index,seed=20261102):
    if mode not in MODES or type(index) is not int or index<0:raise ValueError('23 yeniden ornekleme kimligi gecersiz')
    mapping=np.asarray(groups['image_to_group'],dtype=np.int32);n=len(mapping)
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,MODES.index(mode),index])))
    units=len(groups['labels']) if mode=='cluster' else n
    if units<2:raise ValueError('23 en az iki ornekleme birimi gerekli')
    count=np.bincount(rng.integers(0,units,size=units),minlength=units)
    return count[mapping] if mode=='cluster' else count


def contrasts(values,names=NAMES):
    values=np.asarray(values);lookup={v:i for i,v in enumerate(names)};h=values[...,lookup['hybrid_a025'],:]
    output={f'hybrid_minus_{v}':h-values[...,lookup[v],:] for v in names if v!='hybrid_a025'}
    mean=np.mean([values[...,lookup[f'relocated_s{s}_a025'],:] for s in range(3)],axis=0)
    output['hybrid_minus_relocated_mean']=h-mean
    return output


def intervals(point,replicates,names=NAMES):
    valid=np.isfinite(replicates).all(axis=(1,2));samples=replicates[valid]
    if len(samples)<2:raise ValueError('23 guven araligi icin yeterli tanimli tekrar yok')
    def describe(estimates,original):
        low,high=np.quantile(estimates,[.025,.975],axis=0,method='linear')
        se=estimates.std(axis=0,ddof=1)
        return {k:{'point':float(original[i]),'lower95':float(low[i]),'upper95':float(high[i]),'bootstrap_SE':float(se[i])} for i,k in enumerate(KEYS)}
    methods={name:describe(samples[:,i,:],point[i]) for i,name in enumerate(names)}
    rid=[i for i,name in enumerate(names) if name.startswith('relocated_s')]
    methods['relocated_mean']=describe(samples[:,rid,:].mean(axis=1),point[rid].mean(axis=0))
    d=contrasts(samples,names);dp=contrasts(point,names)
    return {'replicates_requested':len(replicates),'valid_replicates':int(valid.sum()),'undefined_no_target_replicates':int((~valid).sum()),
        'interval':'paired nonparametric percentile, 2.5/97.5 percentiles, numpy linear quantile',
        'methods':methods,'differences_pp':{name:describe(v,dp[name]) for name,v in d.items()},
        'primary_exploratory_AP_contrasts':['hybrid_minus_baseline','hybrid_minus_relocated_mean'],
        'multiplicity':'marginal intervals; no familywise correction, hypothesis-test p-values or confirmatory significance claim',
        'conditional_scope':'fixed trained weights and fixed three location randomizations; sample composition uncertainty only'}
