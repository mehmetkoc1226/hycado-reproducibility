"""Fixed alpha sensitivity and COCO-matched crop geometry; CPU arrays only."""
import contextlib,hashlib,json,sqlite3
import numpy as np
import frozen_cpu as f
import trace29 as t

GRID=[('a000',0.),('a0125',.125),('a025',.25),('a050',.5),('a100',1.)]

def variants(raw,union,ordered,baseline,rec):
    rows,origins=t.candidates(raw,union,ordered,baseline)
    crop=origins[:,0]==1;answer={}
    for name,alpha in GRID:
        if alpha==0:
            answer[name]={'alpha':0.,'detections':baseline,'meaning':'global-only endpoint; crop branch disabled, not zero-score crop injection'}
            continue
        scaled=rows.copy();scaled[crop,4]*=alpha/.25
        if alpha==.25:
            ids,_,_=t.trace(scaled,origins,rec['width'],rec['height'])
            rr=np.full(len(ids),np.nan);is_crop=origins[ids,0]==1
            rr[is_crop]=t.ratio_and_edges(scaled[ids][is_crop],origins[ids][is_crop],ordered)[0]
            answer[name]={'alpha':alpha,'detections':scaled[ids].tolist(),
                          'origins':origins[ids].tolist(),'box_crop_area_ratios':[float(r) if np.isfinite(r) else None for r in rr],
                          'rects':ordered}
        else:answer[name]={'alpha':alpha,'detections':f.nms(scaled.tolist(),rec['width'],rec['height'])}
    return answer

def state(path,signature):
    db=sqlite3.connect(path,timeout=30)
    try:
        db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL')
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS binding(id INTEGER PRIMARY KEY,signature TEXT)')
            db.execute('INSERT OR IGNORE INTO binding VALUES(1,?)',(signature,))
            f.need(db.execute('SELECT signature FROM binding WHERE id=1').fetchone()==(signature,),'Output belongs to another protocol/code; preserve it and report this error')
            db.execute('CREATE TABLE IF NOT EXISTS predictions(method TEXT,image_id INTEGER,digest TEXT,payload BLOB,PRIMARY KEY(method,image_id))')
            db.execute('CREATE TABLE IF NOT EXISTS entries(method TEXT,image_id INTEGER,category INTEGER,area INTEGER,digest TEXT,payload BLOB,PRIMARY KEY(method,image_id,category,area))')
            db.execute('CREATE TABLE IF NOT EXISTS done(method TEXT,image_id INTEGER,marker TEXT,PRIMARY KEY(method,image_id))')
            db.execute('CREATE TABLE IF NOT EXISTS diagnostics(image_id INTEGER PRIMARY KEY,digest TEXT,payload BLOB)')
        return db
    except BaseException:db.close();raise

def load_prediction(db,method,rec,signature):
    row=db.execute('SELECT digest,payload FROM predictions WHERE method=? AND image_id=?',(method,rec['id'])).fetchone()
    if row is None:return None
    doc=f.unpack(*row)
    f.need(doc['signature']==signature and doc['image_sha256']==rec['image_sha256'] and doc['image_id']==rec['id'],'Prediction cache binding differs')
    f.need(doc['alpha']==dict(GRID)[method],'Cached alpha differs');f.validate_rows(doc['detections'],rec)
    if method=='a025':
        f.need(len(doc['origins'])==len(doc['box_crop_area_ratios'])==len(doc['detections']),'Cached origin coverage differs')
    return doc

def done(db,method,image):
    row=db.execute('SELECT marker FROM done WHERE method=? AND image_id=?',(method,image)).fetchone()
    if row is None:return False
    parts=list(db.execute('SELECT category,area,digest FROM entries WHERE method=? AND image_id=? ORDER BY category,area',(method,image)))
    f.need(f.fp(parts)==row[0],'Evaluation-cache coverage marker differs')
    if method=='a025':f.need(db.execute('SELECT 1 FROM diagnostics WHERE image_id=?',(image,)).fetchone() is not None,'Missing coupled geometry diagnostic')
    return True

def save_evaluation(db,method,rec,entries,diagnostic=None):
    import coco_core as c
    with db:
        db.execute('DELETE FROM entries WHERE method=? AND image_id=?',(method,rec['id']))
        for entry in entries:
            d,h,p=c.encode(entry);db.execute('INSERT INTO entries VALUES(?,?,?,?,?,?)',(method,rec['id'],d['category_id'],d['area_index'],h,p))
        if method=='a025':
            f.need(diagnostic is not None,'Alpha0.25 requires coupled geometry diagnostic')
            h,p=f.pack(diagnostic);db.execute('INSERT OR REPLACE INTO diagnostics VALUES(?,?,?)',(rec['id'],h,p))
        parts=list(db.execute('SELECT category,area,digest FROM entries WHERE method=? AND image_id=? ORDER BY category,area',(method,rec['id'])))
        db.execute('INSERT OR REPLACE INTO done VALUES(?,?,?)',(method,rec['id'],f.fp(parts)))

def load_entries(db,method,cat,area,allowed):
    import coco_core as c
    result={}
    for image,h,p in db.execute('SELECT image_id,digest,payload FROM entries WHERE method=? AND category=? AND area=? ORDER BY image_id',(method,cat,area)):
        d=c.decode(h,p)
        f.need(image in allowed and d['image_id']==image and d['category_id']==cat and d['area_index']==area,'COCO cache entry identity differs')
        result[image]=d
    return result

def filtered_detections(doc,integral):
    import visdrone_preprocess as v
    # Retain native final-row IDs through the unchanged ignore-region filter.
    dt=v.xyxy_to_det(doc['detections'])
    tagged=np.column_stack((dt,np.arange(len(dt),dtype=np.int64)))
    filtered=v.outside_ignore(tagged,integral)
    return filtered[:,:6],filtered[:,6].astype(np.int64)

def geometry_diagnostic(doc,rec,entries,filtered,source_ids):
    """COCO area=all maxDets100 at IoU.5/.75; score>=.05 display subgroup.

    dtIds from unencoded COCOeval entries refer to the filtered dataset's
    one-based detection IDs. They map back to the exact final source rows.
    """
    import coco_core as c
    f.need(len(filtered)==len(source_ids),'Filtered source-ID coverage differs')
    groups=['crop_all','crop_ratio_ge_0_95','crop_ratio_lt_0_95',
            'display_crop_all','display_crop_ratio_ge_0_95','display_crop_ratio_lt_0_95']
    counts={str(iou):{g:{'TP':0,'FP':0,'ignored':0} for g in groups} for iou in [.5,.75]}
    examples=[];seen=set()
    origins=np.asarray(doc['origins'],dtype=np.int32).reshape(-1,3)
    for e in entries:
        if list(e['aRng'])!=c.AREAS[0]:continue
        for rank,did in enumerate(e['dtIds'][:100]):
            j=int(did)-1;f.need(0<=j<len(filtered),'COCO dtId outside filtered dataset')
            native=int(source_ids[j]);f.need(native not in seen,'Repeated detection in all-area class lists');seen.add(native)
            row=doc['detections'][native];f.need(int(row[5])+1==e['category_id'] and float(row[4])==float(e['dtScores'][rank]),'COCO detection/source score or class differs')
            if origins[native,0]!=1:continue
            ratio=doc['box_crop_area_ratios'][native];f.need(ratio is not None and 0<ratio<=1,'Missing crop ratio')
            large=ratio>=.95;display=row[4]>=.05
            membership=['crop_all','crop_ratio_ge_0_95' if large else 'crop_ratio_lt_0_95']
            if display:membership+=['display_crop_all','display_crop_ratio_ge_0_95' if large else 'display_crop_ratio_lt_0_95']
            states={}
            for iou,ti in [(.5,0),(.75,5)]:
                status='ignored' if e['dtIgnore'][ti,rank] else 'TP' if e['dtMatches'][ti,rank]>0 else 'FP'
                states[str(iou)]=status
                for g in membership:counts[str(iou)][g][status]+=1
            if display and large:
                crop_index=int(origins[native,1]);rect=doc['rects'][crop_index]
                examples.append({'image':rec['image'],'class_id_zero_based':int(row[5]),'class_name':t.CLASS_NAMES[int(row[5])],
                                 'box_xyxy':row[:4],'crop_xyxy':rect,'crop_index':crop_index,'query_id':int(origins[native,2]),
                                 'scaled_score':float(row[4]),'box_crop_area_ratio':float(ratio),'COCO_IoU_status':states})
    # Limit report size deterministically, separately for matched and unmatched examples.
    selected=[]
    for label in ['TP','FP','ignored']:
        subset=[x for x in examples if x['COCO_IoU_status']['0.5']==label]
        selected.extend(sorted(subset,key=lambda x:(-x['box_crop_area_ratio'],-x['scaled_score'],x['class_id_zero_based'],x['crop_index'],x['query_id']))[:8])
    return {'image_id':rec['id'],'image':rec['image'],'image_sha256':rec['image_sha256'],'alpha':.25,
            'counts':counts,'examples_display_crop_filling':selected,'cap100_applied_before_diagnostic_per_class':True,
            'scope':'COCO class-aware greedy matches, all-area; ignore filtering first; score ties as upstream; IoU-specific FP is not a semantic invalid-box label'}
