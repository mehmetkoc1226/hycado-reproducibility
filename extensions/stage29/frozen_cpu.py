"""Retained CPU-only helpers from stage28; no model or evaluator imports."""

from __future__ import annotations

import contextlib, hashlib, json, math, os, sqlite3, zlib

from pathlib import Path

import numpy as np

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

def pack(doc):
    payload=zlib.compress(json.dumps(doc,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode(),1)
    return hashlib.sha256(payload).hexdigest(),payload

def unpack(digest,payload):
    need(hashlib.sha256(payload).hexdigest()==digest,'Output prediction cache digest differs')
    return json.loads(zlib.decompress(payload))
