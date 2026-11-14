from pathlib import Path
import hashlib,json,sqlite3,zlib,os
from proposal_geometry import canonical_detections

def need(ok,msg):
    if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def write(p,v):
    t=p.with_name(p.name+'.partial');t.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');os.replace(t,p)
def readonly(p):
    p=Path(p).resolve();need(p.is_file(),'Missing source database: '+str(p))
    db=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True,timeout=30);db.execute('PRAGMA query_only=ON');return db
def state(p,signature):
    db=sqlite3.connect(p,timeout=30)
    try:
        db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL')
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS binding(id INTEGER PRIMARY KEY,signature TEXT)')
            db.execute('INSERT OR IGNORE INTO binding VALUES(1,?)',(signature,))
            need(db.execute('SELECT signature FROM binding WHERE id=1').fetchone()==(signature,),'24 cache belongs to different inputs/environment; report this error, do not delete it')
            db.execute('CREATE TABLE IF NOT EXISTS prediction(stem TEXT PRIMARY KEY,digest TEXT,payload BLOB)')
            db.execute('CREATE TABLE IF NOT EXISTS matches(stem TEXT PRIMARY KEY,ann_sha TEXT,digest TEXT,payload BLOB,coco_marker TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS coco(image_id INTEGER,category INTEGER,area INTEGER,digest TEXT,payload BLOB,PRIMARY KEY(image_id,category,area))')
            db.execute('CREATE TABLE IF NOT EXISTS bootstrap(draw INTEGER PRIMARY KEY,digest TEXT,payload TEXT)')
    except BaseException:db.close();raise
    return db

def validate_prediction(body,rec,signature):
    need(body.get('protocol_sha256')==signature and body.get('image_sha256')==rec['image_sha256'] and body.get('image')==rec['image'],'Prediction identity differs: '+rec['stem'])
    need((body.get('width'),body.get('height'))==(rec['width'],rec['height']),'Prediction dimensions differ')
    need(isinstance(body.get('detections'),list),'Prediction detections must be a list')
    rows,bad=canonical_detections(body['detections'],rec['width'],rec['height'])
    need(rows==body['detections'] and bad==0,'Prediction is not canonical')
    # SAHI merged archives legitimately exceed 1000 candidates. Never cap here.
    return body

def load_prediction(db,rec,signature):
    row=db.execute('SELECT digest,payload FROM prediction WHERE stem=?',(rec['stem'],)).fetchone()
    if row is None:return None
    raw=zlib.decompress(row[1]);need(hashlib.sha256(raw).hexdigest()==row[0],'Prediction cache hash differs: '+rec['stem'])
    return validate_prediction(json.loads(raw),rec,signature)

def save_prediction(db,rec,signature,rows,dropped):
    body={'protocol_sha256':signature,'image':rec['image'],'image_sha256':rec['image_sha256'],
          'width':rec['width'],'height':rec['height'],'detections':rows,'dropped_zero_area':int(dropped)}
    validate_prediction(body,rec,signature)
    raw=json.dumps(body,separators=(',',':'),allow_nan=False).encode();digest=hashlib.sha256(raw).hexdigest()
    with db:db.execute('INSERT INTO prediction VALUES(?,?,?)',(rec['stem'],digest,zlib.compress(raw,1)))
    return body

def infer(db,records,dataset,signature,out,factory):
    import time
    predictor=None;new=0;reused=0;start=time.perf_counter()
    for i,rec in enumerate(records,1):
        image=dataset/'images'/(rec['stem']+'.jpg')
        need(sha(image)==rec['image_sha256'],'Test-dev image changed: '+rec['stem'])
        body=load_prediction(db,rec,signature)
        if body is None:
            if predictor is None:predictor=factory()
            w,h,rows,dropped=predictor(image)
            need((w,h)==(rec['width'],rec['height']),'SAHI prediction dimensions differ')
            rows,bad=canonical_detections(rows,w,h);save_prediction(db,rec,signature,rows,dropped+bad);new+=1
        else:reused+=1
        if i in (1,5,len(records)) or i%25==0:
            sec=time.perf_counter()-start;remaining=(len(records)-i)*sec/new if new else None
            print('24 SAHI640 test-dev '+str(i)+'/'+str(len(records))+'; new='+str(new)+'; reused='+str(reused)+
                  ('; rough remaining ~'+str(round(remaining/60,1))+' min' if remaining is not None else ''),flush=True)
            write(out/'progress.json',{'phase':'INFERENCE','done':i,'total':len(records),'new':new,'reused':reused,'rough_remaining_seconds':remaining,'training':False})
    need(db.execute('SELECT COUNT(*) FROM prediction').fetchone()[0]==len(records),'Prediction coverage differs')
    need({r[0] for r in db.execute('SELECT stem FROM prediction')}=={r['stem'] for r in records},'Prediction stems differ')
    index=[{'image':r['image'],'stem':r['stem'],'sha256':db.execute('SELECT digest FROM prediction WHERE stem=?',(r['stem'],)).fetchone()[0],
            'detections':len(load_prediction(db,r,signature)['detections'])} for r in records]
    freeze={'protocol_sha256':signature,'records':index,'all_predictions_committed':True,'annotations_read_by_infer':False}
    write(out/'prediction_index.json',freeze);return freeze
