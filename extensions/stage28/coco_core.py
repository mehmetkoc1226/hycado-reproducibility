"""Bounding-box COCOeval adapter and memory-bounded accumulation; no model inference."""
import contextlib, copy, hashlib, io, json, zlib
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
CATS=list(range(1,11))
CAPS=[1,10,100,500]
AREAS=[[0,1e10],[0,1024],[1024,9216],[9216,1e10]]
LABELS=['all','small','medium','large']

def coco_object(images, annotations):
    c=COCO()
    c.dataset={'images':copy.deepcopy(images),'annotations':copy.deepcopy(annotations),
               'categories':[{'id':k,'name':str(k)} for k in CATS]}
    with contextlib.redirect_stdout(io.StringIO()):c.createIndex()
    return c

def objects(rec,gt,dt):
    """gt: x,y,w,h,class1; dt: x,y,w,h,score,class1."""
    image=[{'id':rec['id'],'width':rec['width'],'height':rec['height']}]
    g=[{'id':i+1,'image_id':rec['id'],'category_id':int(r[4]),'bbox':list(r[:4]),
        'area':float(r[2]*r[3]),'iscrowd':0} for i,r in enumerate(gt)]
    d=[{'id':i+1,'image_id':rec['id'],'category_id':int(r[5]),'bbox':list(r[:4]),
        'area':float(r[2]*r[3]),'score':float(r[4]),'iscrowd':0} for i,r in enumerate(dt)]
    return coco_object(image,g),coco_object(image,d)

def one_image(rec,gt,dt):
    g,d=objects(rec,gt,dt)
    e=COCOeval(g,d,'bbox');e.params.imgIds=[rec['id']];e.params.catIds=CATS;e.params.maxDets=CAPS
    with contextlib.redirect_stdout(io.StringIO()):e.evaluate()
    return [x for x in e.evalImgs if x is not None]

def encode(entry):
    d={'image_id':int(entry['image_id']),'category_id':int(entry['category_id']),
       'area_index':AREAS.index(list(entry['aRng'])),
       'dtScores':np.asarray(entry['dtScores'],dtype=np.float64).tolist(),
       'dtMatches':(np.asarray(entry['dtMatches'])>0).tolist(),
       'dtIgnore':np.asarray(entry['dtIgnore'],dtype=bool).tolist(),
       'gtIgnore':np.asarray(entry['gtIgnore'],dtype=bool).tolist()}
    payload=zlib.compress(json.dumps(d,separators=(',',':'),allow_nan=False).encode(),1)
    return d,hashlib.sha256(payload).hexdigest(),payload

def decode(digest,payload):
    if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Evaluation cache digest differs')
    d=json.loads(zlib.decompress(payload))
    d['dtScores']=np.asarray(d['dtScores'],dtype=np.float64)
    d['dtMatches']=np.asarray(d['dtMatches'],dtype=bool).reshape(10,-1)
    d['dtIgnore']=np.asarray(d['dtIgnore'],dtype=bool).reshape(10,-1)
    d['gtIgnore']=np.asarray(d['gtIgnore'],dtype=bool)
    return d

def accumulate_one(image_ids,category,area,entries):
    e=COCOeval(None,None,'bbox')
    e.params.imgIds=list(image_ids);e.params.catIds=[category]
    e.params.areaRng=[AREAS[area]];e.params.areaRngLbl=[LABELS[area]];e.params.maxDets=CAPS
    e._paramsEval=copy.deepcopy(e.params)
    e.evalImgs=[entries.get(i) for i in image_ids]
    with contextlib.redirect_stdout(io.StringIO()):e.accumulate()
    return e.eval['precision'][:,:,0,0,:],e.eval['recall'][:,0,0,:]

def mean(a):
    a=np.asarray(a);valid=a[a>=0]
    return float(valid.mean()*100) if len(valid) else None

def aggregate(image_ids,loader):
    p=np.full((10,101,10,4,4),-1.,dtype=np.float64)
    r=np.full((10,10,4,4),-1.,dtype=np.float64)
    for c in CATS:
        for a in range(4):
            pp,rr=accumulate_one(image_ids,c,a,loader(c,a))
            p[:,:,c-1,a,:]=pp;r[:,c-1,a,:]=rr
    results={}
    for cap in (100,500):
        m=CAPS.index(cap)
        vals={'AP':mean(p[:,:,:,0,m]),'AP50':mean(p[0,:,:,0,m]),'AP75':mean(p[5,:,:,0,m]),
              'AP_small':mean(p[:,:,:,1,m]),'AP_medium':mean(p[:,:,:,2,m]),
              'AP_large':mean(p[:,:,:,3,m]),'AR':mean(r[:,:,0,m])}
        vals['per_class']=[{'category_id':c,'AP':mean(p[:,:,c-1,0,m]),
                           'AP50':mean(p[0,:,c-1,0,m]),'AP75':mean(p[5,:,c-1,0,m])} for c in CATS]
        results[str(cap)]=vals
    return results,p,r
