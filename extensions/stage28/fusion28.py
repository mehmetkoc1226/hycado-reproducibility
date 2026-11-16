"""CPU-only, fixed two-resolution late fusion. No torch or model imports."""
from __future__ import annotations
import contextlib, hashlib, json, math, os, sqlite3, zlib
from pathlib import Path
import numpy as np

METHOD='yolo800_1280_fusion'

def need(ok,message):
    if not ok:raise ValueError(message)

def fp(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_bytes())

def write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)

def readonly(path):
    path=Path(path).resolve();need(path.is_file(),'Missing source database: '+str(path))
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=30)
    db.execute('PRAGMA query_only=ON');return db

@contextlib.contextmanager
def run_lock(path):
    # OS releases the lock after a crash/restart; the harmless file may remain.
    with Path(path).open('a+b') as f:
        f.seek(0,2)
        if not f.tell():f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def validate_rows(rows,rec,cap=1000):
    need(isinstance(rows,list) and len(rows)<=cap,'Candidate list/cap differs')
    for r in rows:
        need(isinstance(r,list) and len(r)==6 and all(type(v) in (int,float) and math.isfinite(v) for v in r),'Invalid candidate values')
        x,y,xx,yy,s,c=r
        need(0<=x<xx<=rec['width'] and 0<=y<yy<=rec['height'] and 0<=s<=1 and c==int(c) and 0<=c<10,'Candidate geometry/score/class differs')

def canonical(rows,width,height):
    clean=[]
    for row in rows:
        need(len(row)==6 and all(math.isfinite(float(v)) for v in row),'Invalid NMS candidate')
        x,y,xx,yy,s,c=map(float,row)
        need(x<=xx and y<=yy and 0<=s<=1 and c==int(c) and 0<=c<10,'Invalid NMS geometry/score/class')
        x,xx=max(0.,min(width,x)),max(0.,min(width,xx))
        y,yy=max(0.,min(height,y)),max(0.,min(height,yy))
        if xx>x and yy>y:clean.append([x,y,xx,yy,s,int(c)])
    clean.sort(key=lambda r:(-r[4],r[5],*r[:4]))
    return clean

def nms(rows,width,height,threshold=.7,limit=1000):
    a=np.asarray(canonical(rows,width,height),dtype=np.float64).reshape(-1,6)
    remaining=np.arange(len(a));selected=[]
    areas=(a[:,2]-a[:,0])*(a[:,3]-a[:,1])
    while len(remaining) and len(selected)<limit:
        i=remaining[0];selected.append(int(i));other=remaining[1:]
        inter=np.maximum(0,np.minimum(a[i,2:4],a[other,2:4])-np.maximum(a[i,:2],a[other,:2])).prod(axis=1)
        union=areas[i]+areas[other]-inter
        iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
        remaining=other[~((a[other,5]==a[i,5])&(iou>threshold))]
    return a[selected].tolist()

def fuse(y800,y1280,width,height):
    # Native xyxy coordinates: never multiply coordinates by 1280/800.
    # Both retained scores have multiplier 1.0. No score retuning/cutoff.
    return nms(list(y800)+list(y1280),width,height)

def state(path,signature):
    db=sqlite3.connect(path,timeout=30)
    try:
        db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL')
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS binding(id INTEGER PRIMARY KEY,signature TEXT)')
            db.execute('INSERT OR IGNORE INTO binding VALUES(1,?)',(signature,))
            need(db.execute('SELECT signature FROM binding WHERE id=1').fetchone()==(signature,),'Output cache belongs to different code/inputs; preserve it and report this error')
            db.execute('CREATE TABLE IF NOT EXISTS predictions(image_id INTEGER PRIMARY KEY,digest TEXT,payload BLOB)')
            db.execute('CREATE TABLE IF NOT EXISTS entries(method TEXT,image_id INTEGER,category INTEGER,area INTEGER,digest TEXT,payload BLOB,PRIMARY KEY(method,image_id,category,area))')
            db.execute('CREATE TABLE IF NOT EXISTS done(method TEXT,image_id INTEGER,marker TEXT,PRIMARY KEY(method,image_id))')
            db.execute('CREATE TABLE IF NOT EXISTS draws(draw INTEGER PRIMARY KEY,digest TEXT,payload TEXT)')
        return db
    except BaseException:db.close();raise

def pack(doc):
    payload=zlib.compress(json.dumps(doc,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode(),1)
    return hashlib.sha256(payload).hexdigest(),payload

def unpack(digest,payload):
    need(hashlib.sha256(payload).hexdigest()==digest,'Output prediction cache digest differs')
    return json.loads(zlib.decompress(payload))

def source_body(db,method,rec,expected):
    row=db.execute('SELECT digest,payload FROM prediction WHERE method=? AND stem=?',(method,rec['stem'])).fetchone()
    need(row is not None,'Missing frozen output: '+method+'/'+rec['stem'])
    plain=zlib.decompress(row[1]);need(hashlib.sha256(plain).hexdigest()==row[0]==expected,'Frozen output digest differs: '+method+'/'+rec['stem'])
    b=json.loads(plain);need(b['image_sha256']==rec['image_sha256'],'Frozen image binding differs')
    validate_rows(b['detections'],rec);return b['detections']

def save_entries(db,method,rec,entries):
    import coco_core as c
    with db:
        db.execute('DELETE FROM entries WHERE method=? AND image_id=?',(method,rec['id']))
        for entry in entries:
            d,h,p=c.encode(entry)
            db.execute('INSERT INTO entries VALUES(?,?,?,?,?,?)',(method,rec['id'],d['category_id'],d['area_index'],h,p))
        parts=list(db.execute('SELECT category,area,digest FROM entries WHERE method=? AND image_id=? ORDER BY category,area',(method,rec['id'])))
        db.execute('INSERT OR REPLACE INTO done VALUES(?,?,?)',(method,rec['id'],fp(parts)))

def entries_done(db,method,image):
    row=db.execute('SELECT marker FROM done WHERE method=? AND image_id=?',(method,image)).fetchone()
    if row is None:return False
    parts=list(db.execute('SELECT category,area,digest FROM entries WHERE method=? AND image_id=? ORDER BY category,area',(method,image)))
    need(fp(parts)==row[0],'COCO cache coverage/marker differs')
    return True

def load_entries(db,method,cat,area,allowed):
    import coco_core as c
    result={}
    for image,h,p in db.execute('SELECT image_id,digest,payload FROM entries WHERE method=? AND category=? AND area=? ORDER BY image_id',(method,cat,area)):
        d=c.decode(h,p)
        need(image in allowed and d['image_id']==image and d['category_id']==cat and d['area_index']==area,'COCO payload identity differs')
        result[image]=d
    return result
