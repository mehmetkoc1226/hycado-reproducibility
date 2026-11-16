"""Decode retained float32 crop queries and trace the unchanged alpha=.25 fusion."""
import hashlib,io,json,zlib
from collections import Counter
import numpy as np
import frozen_cpu as f

BINS=np.array([0,.01,.05,.1,.25,.5,.75,.9,.95,1.0000000001])
CLASS_NAMES=['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']

def plan(value,rec):
    f.need(isinstance(value,list) and len(value)<=60,'Invalid frozen crop plan')
    for r in value:
        f.need(isinstance(r,list) and len(r)==4 and all(type(v) is int for v in r),'Crop rectangle must contain integers')
        x,y,xx,yy=r;f.need(0<=x<xx<=rec['width'] and 0<=y<yy<=rec['height'],'Crop outside native image')
    f.need(len({tuple(r) for r in value})==len(value),'Duplicate rectangle in frozen plan')
    return value

def raw_queries(db,rec,union,expected_hash):
    row=db.execute('SELECT digest,payload FROM crop_raw WHERE stem=?',(rec['stem'],)).fetchone()
    f.need(row is not None,'Missing crop_raw record: '+rec['stem'])
    f.need(hashlib.sha256(row[1]).hexdigest()==row[0]==expected_hash,'Raw crop archive SHA256 differs: '+rec['stem'])
    with np.load(io.BytesIO(row[1]),allow_pickle=False) as data:
        f.need(set(data.files)=={'rects','queries'},'Unexpected raw archive members')
        rects=data['rects'];queries=data['queries']
        f.need(rects.dtype==np.int32 and rects.shape==(len(union),4) and rects.tolist()==union,'Raw crop union geometry differs')
        f.need(queries.dtype==np.float32 and queries.shape==(len(union),300,14) and np.isfinite(queries).all(),'Raw query shape/dtype/finiteness differs')
        f.need(np.all(queries[:,:,2:4]>=0) and np.all((queries[:,:,4:]>=0)&(queries[:,:,4:]<=1)),'Raw widths/heights or class probabilities out of range')
        return queries.copy()

def decode(raw,rect):
    # Same float64 arithmetic, strict score >.001, first argmax and clipping
    # as retained dev_evaluation.map_queries. Query IDs survive filtering.
    a=np.asarray(raw,dtype=np.float64)
    f.need(a.shape==(300,14) and np.isfinite(a).all(),'Invalid crop query shape')
    f.need(np.all(a[:,2:4]>=0) and np.all((a[:,4:]>=0)&(a[:,4:]<=1)),'Invalid crop query ranges')
    cls=np.argmax(a[:,4:],axis=1);score=a[np.arange(len(a)),cls+4]
    ids=np.flatnonzero(score>.001);q=a[ids];cls=cls[ids];score=score[ids]
    cx,cy,w,h=q[:,:4].T;x,y,xx,yy=rect;width,height=xx-x,yy-y
    rows=np.column_stack([np.clip(cx-w/2,0,1)*width+x,np.clip(cy-h/2,0,1)*height+y,
                          np.clip(cx+w/2,0,1)*width+x,np.clip(cy+h/2,0,1)*height+y,score,cls])
    good=(rows[:,2]>rows[:,0])&(rows[:,3]>rows[:,1])
    return rows[good],ids[good]

def candidates(raw,union,ordered,baseline):
    lookup={tuple(r):i for i,r in enumerate(union)}
    rows=list(baseline);origins=[[0,-1,i] for i in range(len(baseline))]
    for j,rect in enumerate(ordered):
        a,ids=decode(raw[lookup[tuple(rect)]],rect)
        a[:,4]*=.25;rows.extend(a.tolist());origins.extend([[1,j,int(q)] for q in ids])
    return np.asarray(rows,dtype=np.float64).reshape(-1,6),np.asarray(origins,dtype=np.int32).reshape(-1,3)

def trace(rows,origins,width,height):
    a=np.asarray(rows,dtype=np.float64).reshape(-1,6);o=np.asarray(origins,dtype=np.int32).reshape(-1,3)
    f.need(len(a)==len(o),'Candidate/origin count differs')
    clean=f.canonical(a.tolist(),width,height)
    order=np.array(sorted(range(len(a)),key=lambda i:(-a[i,4],a[i,5],*a[i,:4],i)),dtype=np.int64)
    f.need(a[order].tolist()==clean,'Noncanonical pre-NMS candidates')
    remaining=order;selected=[];status=np.zeros(len(a),dtype=np.int8);suppressed_by=np.full(len(a),-1,dtype=np.int64)
    area=(a[:,2]-a[:,0])*(a[:,3]-a[:,1])
    while len(remaining) and len(selected)<1000:
        i=int(remaining[0]);selected.append(i);status[i]=1;other=remaining[1:]
        inter=np.maximum(0,np.minimum(a[i,2:4],a[other,2:4])-np.maximum(a[i,:2],a[other,:2])).prod(axis=1)
        union=area[i]+area[other]-inter;iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
        bad=(a[other,5]==a[i,5])&(iou>.7)
        status[other[bad]]=2;suppressed_by[other[bad]]=i;remaining=other[~bad]
    status[remaining]=3
    return np.asarray(selected,dtype=np.int64),status,suppressed_by

def replay(raw,union,ordered,baseline,rec):
    rows,_=candidates(raw,union,ordered,baseline)
    return f.nms(rows.tolist(),rec['width'],rec['height'])

def ratio_and_edges(rows,origins,ordered):
    a=np.asarray(rows,dtype=np.float64).reshape(-1,6);o=np.asarray(origins,dtype=np.int32).reshape(-1,3)
    if not len(a):return np.empty(0),np.empty(0,dtype=np.int8)
    rr=np.asarray(ordered,dtype=np.float64).reshape(-1,4)[o[:,1]]
    ratios=(a[:,2]-a[:,0])*(a[:,3]-a[:,1])/((rr[:,2]-rr[:,0])*(rr[:,3]-rr[:,1]))
    f.need(np.all(ratios>0) and np.all(ratios<=1+1e-10),'Decoded box/crop area ratio out of range')
    edges=np.sum(np.isclose(a[:,:4],rr,rtol=0,atol=1e-6),axis=1).astype(np.int8)
    return np.clip(ratios,0,1),edges

def group_stats(rows,origins,ordered):
    a=np.asarray(rows,dtype=np.float64).reshape(-1,6)
    ratios,edges=ratio_and_edges(a,origins,ordered)
    return {'count':len(a),'area_ratio_histogram':np.histogram(ratios,bins=BINS)[0].tolist(),
            'area_ratio_sum':float(ratios.sum()),'ratio_ge_0_5':int(np.count_nonzero(ratios>=.5)),
            'ratio_ge_0_8':int(np.count_nonzero(ratios>=.8)),'ratio_ge_0_95':int(np.count_nonzero(ratios>=.95)),
            'touches_at_least_two_crop_edges':int(np.count_nonzero(edges>=2)),
            'touches_all_four_crop_edges':int(np.count_nonzero(edges==4)),
            'class_counts':np.bincount(a[:,5].astype(np.int32),minlength=10).tolist(),
            'max_scaled_score':float(a[:,4].max()) if len(a) else None}

def image_summary(rec,raw,union,ordered,baseline,expected):
    rows,origins=candidates(raw,union,ordered,baseline)
    ids,status,by=trace(rows,origins,rec['width'],rec['height'])
    f.need(rows[ids].tolist()==expected,'Raw replay differs from frozen HyCADO-1280: '+rec['stem'])
    crop=origins[:,0]==1;retained=(status==1)&crop;visible=retained&(rows[:,4]>=.05)
    subsets={'pre_nms_crop':crop,'retained_crop':retained,'retained_crop_display_ge_0_05':visible}
    stats={k:group_stats(rows[v],origins[v],ordered) for k,v in subsets.items()}
    counters=Counter()
    for kind,label in [(0,'global'),(1,'crop')]:
        for code,name in [(1,'retained'),(2,'suppressed'),(3,'discarded_by_cap1000')]:
            counters[label+'_'+name]=int(np.count_nonzero((origins[:,0]==kind)&(status==code)))
    for i in np.flatnonzero(status==2):
        a='global' if origins[i,0]==0 else 'crop';b='global' if origins[by[i],0]==0 else 'crop'
        counters[a+'_suppressed_by_'+b]+=1
    ratios,edges=ratio_and_edges(rows[retained],origins[retained],ordered)
    selected=np.flatnonzero(retained);examples=[];visible_examples=[]
    ranked=sorted(range(len(selected)),key=lambda j:(-ratios[j],-rows[selected[j],4],int(selected[j])))
    all_top=set(ranked[:12]);visible_top=set([j for j in ranked if rows[selected[j],4]>=.05][:12])
    for j in sorted(all_top|visible_top,key=lambda j:(-ratios[j],-rows[selected[j],4],int(selected[j]))):
        i=int(selected[j]);crop_id=int(origins[i,1]);rect=ordered[crop_id]
        example={'image':rec['image'],'class_id_zero_based':int(rows[i,5]),'class_name':CLASS_NAMES[int(rows[i,5])],
                         'box_xyxy':rows[i,:4].tolist(),'crop_xyxy':rect,'crop_index':crop_id,'query_id':int(origins[i,2]),
                         'scaled_score':float(rows[i,4]),'raw_score':float(rows[i,4]/.25),'box_crop_area_ratio':float(ratios[j]),
                         'crop_edges_touched':int(edges[j]),'display_threshold_ge_0_05':bool(rows[i,4]>=.05),
                         'box_image_area_ratio':float((rows[i,2]-rows[i,0])*(rows[i,3]-rows[i,1])/(rec['width']*rec['height'])),
                         'GT_match_or_FP_status':'not evaluated; geometry alone does not identify a false positive'}
        if j in all_top:examples.append(example)
        if j in visible_top:visible_examples.append(example)
    return {'image':rec['image'],'crop_count':len(ordered),'raw_union_crop_count':len(union),
            'full_queries_in_union':int(raw.shape[0]*300),'exact_HyCADO1280_replay':True,
            'counts':dict(counters),'distributions':stats,'largest_retained_box_crop_ratios':examples,
            'largest_display_box_crop_ratios':visible_examples}
