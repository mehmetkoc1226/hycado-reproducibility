"""Existing filename/hash grouping, PCG64 draw and exact tie-order helpers; no AP statistic imported."""
from pathlib import PurePosixPath
import re
import numpy as np
MODES=["cluster","image"]

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

def repeated_indices(block,frequency):
    weights=frequency[block['images']]
    repeated=np.repeat(np.arange(len(weights)),weights)
    if not len(repeated):return repeated
    offset=np.r_[0,np.cumsum(weights)]
    begin=block['begin'][repeated]
    # A tied image block [TP, FP] copied twice is TP,FP,TP,FP, not TP,TP,FP,FP.
    return begin+(np.arange(len(repeated))-offset[begin])%block['size'][repeated]

def frequencies(groups,mode,index,seed=20261102):
    if mode not in MODES or type(index) is not int or index<0:raise ValueError('23 yeniden ornekleme kimligi gecersiz')
    mapping=np.asarray(groups['image_to_group'],dtype=np.int32);n=len(mapping)
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,MODES.index(mode),index])))
    units=len(groups['labels']) if mode=='cluster' else n
    if units<2:raise ValueError('23 en az iki ornekleme birimi gerekli')
    count=np.bincount(rng.integers(0,units,size=units),minlength=units)
    return count[mapping] if mode=='cluster' else count
