"""Pinned CPU COCO scoring of frozen predictions. Run preflight before full."""
from __future__ import annotations
import argparse, contextlib, csv, hashlib, io, json, math, os, platform
from pathlib import Path
import sqlite3, sys, time, traceback, unittest, zipfile
HERE=Path(__file__).resolve().parent
RUN='supplementary_coco_v2_seed42'

def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def fp(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def need(ok,msg):
    if not ok:raise ValueError(msg)
def write(path,d):
    temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)

def cache_open(path,signature):
    db=sqlite3.connect(path,timeout=30)
    db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL')
    with db:
        db.execute('CREATE TABLE IF NOT EXISTS binding(id INTEGER PRIMARY KEY,signature TEXT)')
        db.execute('INSERT OR IGNORE INTO binding VALUES(1,?)',(signature,))
        need(db.execute('SELECT signature FROM binding WHERE id=1').fetchone()==(signature,),'New cache belongs to another protocol; keep it and use a new run version.')
        db.execute('CREATE TABLE IF NOT EXISTS entries(split TEXT,method TEXT,image_id INTEGER,category INTEGER,area INTEGER,digest TEXT,payload BLOB,PRIMARY KEY(split,method,image_id,category,area))')
        db.execute('CREATE TABLE IF NOT EXISTS done(split TEXT,method TEXT,image_id INTEGER,marker TEXT NOT NULL,PRIMARY KEY(split,method,image_id))')
    return db

def records_and_methods(root,audit):
    val=[]
    for i,r in enumerate(sorted(read(HERE/'references/val_images.json')['records'],key=lambda x:x['source_image']),1):
        val.append({'id':i,'image':r['source_image'],'stem':Path(r['source_image']).stem,
                    'width':r['width'],'height':r['height'],'image_sha256':r['sha256'],
                    'annotation_sha256':r['annotation_sha256'],'annotation':Path(r['source_image']).stem+'.txt'})
    td=read(HERE/'references/stage18/dataset_manifest.json')['records']
    td=[dict(r,id=i) for i,r in enumerate(sorted(td,key=lambda x:x['image']),1)]
    labels={x['stem']:x['sha256'] for x in read(HERE/'references/stage18/annotation_index.json')['records']}
    for r in td:r['annotation_sha256']=labels[r['stem']]
    return {'val':{'records':val,'annotations':root/'data/prepared/visdrone_annotations/val',
                   'methods':['yolo800_repeat','yolo1280','yolo1536','sahi640','sahi800','hybrid800_legacy']},
            'testdev':{'records':td,'annotations':Path(audit['sections']['testdev_dataset']['path'])/'annotations',
                       'methods':['yolo800','yolo1280','hybrid_a025','relocated_s0_a025','relocated_s1_a025','relocated_s2_a025','hybrid1280','yolo1280_rois800']}}

def source_state(root):
    from input_inventory import readonly
    old=root/'experiments/testdev_frozen_v1_seed42'
    new=root/'experiments/testdev1280_extension_v1_seed42'
    refs={}
    for stage,folder in [(18,old),(20,new)]:
        expected=read(HERE/'references'/('stage'+str(stage))/'result.json')['prediction_database_sha256']
        path=folder/'predictions.sqlite3'
        print('Source database SHA256:',path,flush=True)
        actual=sha(path);need(actual==expected,'Complete source database identity differs: '+str(path))
        refs[str(stage)]={'sha256':actual,'stat':(path.stat().st_size,path.stat().st_mtime_ns)}
    return refs

class Sources:
    def __init__(self,root):
        import input_inventory as i
        self.root=root;self.stack=contextlib.ExitStack()
        self.dbs={n:self.stack.enter_context(contextlib.closing(i.readonly(root/'experiments'/folder/'predictions.sqlite3')))
                  for n,folder in [(18,'testdev_frozen_v1_seed42'),(20,'testdev1280_extension_v1_seed42')]}
        self.index={n:{Path(r['image']).stem:r for r in read(HERE/'references'/('stage'+str(n))/'prediction_index.json')['records']} for n in (18,20)}
        self.p2=root/'experiments/p2_strong_baselines_v1_seed42'
        self.pidx={m:{r['source_image']:r for r in rows} for m,rows in read(HERE/'references/p2_val/official_val_prediction_index.json')['methods'].items()}
        self.old=root/'experiments/official_val_a025_v1_seed42'
        self.oidx={r['source_image']:r for r in read(self.old/'prediction_index.json')['records']}
        self.frozen=read(HERE/'references/legacy_frozen.json')
    def close(self):self.stack.close()
    def sqlite_body(self,stage,method,rec):
        import zlib
        digest,payload=self.dbs[stage].execute('SELECT digest,payload FROM prediction WHERE method=? AND stem=?',(method,rec['stem'])).fetchone()
        plain=zlib.decompress(payload)
        need(hashlib.sha256(plain).hexdigest()==digest==self.index[stage][rec['stem']]['variant_sha256'][method],'Source prediction hash differs')
        return json.loads(plain)
    def get(self,split,rec):
        import input_inventory as i
        ans={}
        if split=='testdev':
            docs={m:self.sqlite_body(18,m,rec) for m in ['yolo800','yolo1280','hybrid_a025','relocated_s0_a025','relocated_s1_a025','relocated_s2_a025']}
            check={m:self.sqlite_body(20,m,rec) for m in ['yolo1280','hybrid800_reference','hybrid1280','yolo1280_rois800']}
            need(check['yolo1280']['detections']==docs['yolo1280']['detections'] and
                 check['hybrid800_reference']['detections']==docs['hybrid_a025']['detections'],'Source18/20 reused output parity differs')
            docs.update({k:check[k] for k in ['hybrid1280','yolo1280_rois800']})
            for m,b in docs.items():i.validate_rows(b,rec);ans[m]=b['detections']
        else:
            for m,idx in self.pidx.items():
                x=idx[rec['image']];p=i.safe(self.p2,x['path'])
                need(sha(p)==x['sha256'],'VAL prediction identity differs')
                b=read(p);need(b['protocol_sha256']==read(HERE/'references/p2_val/official_val_prediction_index.json')['protocol_sha256'],'P2 protocol differs')
                i.validate_p2_rows(b,rec,m,x);ans[m]=b['detections']
            x=self.oidx[rec['image']];p=i.safe(self.old,x['path']);need(sha(p)==x['sha256'],'Legacy prediction identity differs')
            b=read(p);need(b['protocol_sha256']==self.frozen['original_protocol_content_sha256'] and
                           b['content_sha256']==fp({k:v for k,v in b.items() if k!='content_sha256'}),'Legacy signed content differs')
            need(b['baseline']==ans['yolo800_repeat'],'VAL baseline replay differs')
            i.validate_rows({'image_sha256':b['image_sha256'],'detections':b['hybrid_a025']},rec)
            ans['hybrid800_legacy']=b['hybrid_a025']
        return ans

def gt_and_integral(folder,rec):
    import numpy as np
    import visdrone_preprocess as v
    p=folder/rec['annotation'];need(sha(p)==rec['annotation_sha256'],'GT identity differs')
    raw=v.raw_annotation(p.read_text(encoding='utf-8-sig'))
    need(not np.any((raw[:,5]>=1)&(raw[:,5]<=10)&(raw[:,4]==0)),'Class-specific ignored GT requires protocol review')
    need(not np.any((raw[:,2]<=0)|(raw[:,3]<=0)),'Nonpositive GT geometry')
    integral=v.ignore_integral(raw,rec['width'],rec['height'])
    normal=v.prepare_groundtruth(raw,integral)
    normal=normal[(normal[:,5]>=1)&(normal[:,5]<=10)&(normal[:,4]==0)]
    return np.column_stack((normal[:,:4],normal[:,5])).tolist(),integral

def store_entries(db,split,method,rec,entries):
    import coco_core as c
    with db:
        db.execute('DELETE FROM entries WHERE split=? AND method=? AND image_id=?',(split,method,rec['id']))
        for entry in entries:
            d,h,b=c.encode(entry)
            db.execute('INSERT INTO entries VALUES(?,?,?,?,?,?,?)',(split,method,rec['id'],d['category_id'],d['area_index'],h,b))
        parts=list(db.execute('SELECT category,area,digest FROM entries WHERE split=? AND method=? AND image_id=? ORDER BY category,area',(split,method,rec['id'])))
        db.execute('INSERT OR REPLACE INTO done VALUES(?,?,?,?)',(split,method,rec['id'],fp(parts)))

def calculate(root,out,sets,mode,signature):
    import numpy as np
    import coco_core as c
    import visdrone_preprocess as v
    timings={};metrics={};totals={}
    dbpath=out/('preflight_cache.sqlite3' if mode=='preflight' else 'evaluation_cache.sqlite3')
    with contextlib.closing(cache_open(dbpath,fp({'signature':signature,'mode':mode}))) as cache:
        src=Sources(root)
        try:
            for split,s in sets.items():
                rows=s['records']
                if mode=='preflight':
                    inds=sorted({int(x) for x in np.linspace(0,len(rows)-1,12)})
                    rows=[rows[k] for k in inds]
                start=time.perf_counter();gt_total=0;new=0;reused=0
                for n,rec in enumerate(rows,1):
                    gt,integral=gt_and_integral(s['annotations'],rec);gt_total+=len(gt)
                    docs=src.get(split,rec)
                    for method in s['methods']:
                        old=cache.execute('SELECT marker FROM done WHERE split=? AND method=? AND image_id=?',(split,method,rec['id'])).fetchone()
                        if old and mode=='full':
                            parts=list(cache.execute('SELECT category,area,digest FROM entries WHERE split=? AND method=? AND image_id=? ORDER BY category,area',(split,method,rec['id'])))
                            need(fp(parts)==old[0],'Cached image entry coverage changed')
                            reused+=1;continue
                        dt=v.outside_ignore(v.xyxy_to_det(docs[method]),integral)
                        entries=c.one_image(rec,gt,dt.tolist())
                        store_entries(cache,split,method,rec,entries);new+=1
                    if n in (1,len(rows)) or n%10==0:
                        print(f'COCO {mode} {split}: {n}/{len(rows)}; computed={new}; reused={reused}',flush=True)
                        write(out/'progress.json',{'mode':mode,'split':split,'images_done':n,'images_total':len(rows),'training':False,'gpu':False})
                totals[split]={'images':len(rows),'normal_gt_after_ignore':gt_total}
                if mode=='full' and split=='val':need(gt_total==38599,'VAL GT count differs from frozen ignore preprocessing')
                if mode=='full' and split=='testdev':
                    old=read(HERE/'references/stage18/metrics.json')['methods']['yolo800']['per_class']
                    need(gt_total==sum(x['gt_count_in_reference_denominator'] for x in old),'Test-dev GT count differs from original evaluator')
                point={}
                for method in s['methods']:
                    def loader(cat,area):
                        answer={}
                        for image,h,payload in cache.execute('SELECT image_id,digest,payload FROM entries WHERE split=? AND method=? AND category=? AND area=? ORDER BY image_id',(split,method,cat,area)):
                            d=c.decode(h,payload);need(d['image_id']==image and d['category_id']==cat and d['area_index']==area,'Cache identity differs');answer[image]=d
                        return answer
                    point[method]=c.aggregate([x['id'] for x in rows],loader)[0]
                metrics[split]=point
                seconds=time.perf_counter()-start
                timings[split]={'images_measured':len(rows),'seconds_scoring_and_accumulation':seconds,
                                'rough_full_point_estimate_seconds':seconds*len(s['records'])/len(rows),
                                'estimate_excludes_bootstrap':True,'model_latency_benchmark':False}
        finally:src.close()
    return metrics,timings,totals

def execute(root,out,mode,dataset,env):
    for name,h in read(HERE/'package_sha256.json').items():need(sha(HERE/name)==h,'Package identity differs: '+name)
    import input_inventory
    print('Checking original file bindings and all annotation hashes...',flush=True)
    audit=input_inventory.run(root,dataset,False);write(out/'input_audit.json',audit)
    need(audit['status']=='PREFLIGHT_READY','Input audit needs review; see input_audit.json')
    need(read(HERE/'references/21_onkontrol.json')['status']=='PREFLIGHT_READY','Completed 21 report required')
    sources=source_state(root)
    config={'id':'hycado_coco_point_eval_v1','metric_protocol_sha256':sha(HERE/'protocol.json'),
            'implementation_version':'22_v2_sahi_candidate_validation_repair',
            'code_sha256':{x:sha(HERE/x) for x in ['coco_runner.py','coco_core.py','input_inventory.py','visdrone_preprocess.py','bootstrap_vendor.py','test_coco.py']},
            'environment':env,'source_database_sha256':{k:v['sha256'] for k,v in sources.items()},
            'sampling':'12 evenly spaced images per split for preflight only','bootstrap_in_this_runner':False}
    signature=fp(config)
    if mode=='full':
        gate=read(out/'preflight.json')
        need(gate['status']=='COCO_PREFLIGHT_OK' and gate['signature']==signature,'Current successful preflight required')
    import test_coco
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_coco)
    tests=unittest.TextTestRunner(verbosity=2).run(suite)
    need(tests.wasSuccessful(),'Synthetic/reference tests failed')
    sets=records_and_methods(root,audit)
    metrics,timings,totals=calculate(root,out,sets,mode,signature)
    for stage,folder in [(18,'testdev_frozen_v1_seed42'),(20,'testdev1280_extension_v1_seed42')]:
        p=root/'experiments'/folder/'predictions.sqlite3'
        need((p.stat().st_size,p.stat().st_mtime_ns)==tuple(sources[str(stage)]['stat']),'Source database changed during scoring')
    result={'status':'COCO_PREFLIGHT_OK' if mode=='preflight' else 'COCO_POINT_ESTIMATES_COMPLETE',
            'signature':signature,'config':config,'mode':mode,'totals':totals,'timings':timings,
            'tests_passed':tests.testsRun,'training':False,'gpu':False,'bootstrap_completed':False,
            'confidence_intervals_pending':True,
            'scope':'12-image diagnostic only; NOT paper results' if mode=='preflight' else
                    'Full point metrics; paired bootstrap and new recall-FP diagnostics remain separate stages',
            'metric_label':'COCO-style AP on VisDrone with stated ignore-region preprocessing'}
    result['subset_metrics_not_for_manuscript' if mode=='preflight' else 'metrics']=metrics
    if mode=='full':
        fields=['split','method','maxDets','AP','AP50','AP75','AP_small','AP_medium','AP_large','AR']
        with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for split,methods in metrics.items():
                for method,caps in methods.items():
                    for cap,m in caps.items():writer.writerow({'split':split,'method':method,'maxDets':cap,**{k:m[k] for k in fields[3:]}})
        result['testdev_AP_differences']={cap:{'hybrid1280_minus_yolo1280':metrics['testdev']['hybrid1280'][cap]['AP']-metrics['testdev']['yolo1280'][cap]['AP'],
            'yolo1280_rois800_minus_yolo1280':metrics['testdev']['yolo1280_rois800'][cap]['AP']-metrics['testdev']['yolo1280'][cap]['AP'],
            'hybrid1280_minus_yolo1280_rois800':metrics['testdev']['hybrid1280'][cap]['AP']-metrics['testdev']['yolo1280_rois800'][cap]['AP']} for cap in ['100','500']}
    write(out/('preflight.json' if mode=='preflight' else 'point_results.json'),result)
    return result

def bundle(out,mode,result):
    path=HERE/('22A_coco_onkontrol_rapor.zip' if mode=='preflight' else '22B_coco_nokta_sonuclar_rapor.zip')
    temp=path.with_suffix('.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('result.json',json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
        for name in ['input_audit.json','progress.json']:
            if (out/name).exists():z.write(out/name,name)
        z.write(HERE/'protocol.json','protocol.json')
        z.write(HERE/'repair_record.json','repair_record.json')
        if mode=='full' and (out/'metrics.csv').exists():z.write(out/'metrics.csv','metrics.csv')
    os.replace(temp,path);return path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=Path(r'D:\HyCADO_Temiz_Baslangic'))
    ap.add_argument('--dataset');ap.add_argument('--mode',choices=['preflight','full'],default='preflight')
    a=ap.parse_args();root=a.project.resolve()
    need(root.is_dir(),'Project folder missing')
    out=root/'experiments'/RUN;out.mkdir(parents=True,exist_ok=True)
    try:
        import bootstrap_vendor
        env=bootstrap_vendor.ensure()
        result=execute(root,out,a.mode,a.dataset,env)
    except Exception as exc:
        result={'status':'ERROR','mode':a.mode,'error':str(exc),'traceback':traceback.format_exc(),'training':False,'gpu':False}
        write(out/'error.json',result);traceback.print_exc()
    path=bundle(out,a.mode,result)
    print('SONUC:',result['status']);print('GONDERILECEK RAPOR:',path)
    if result['status']=='COCO_PREFLIGHT_OK':
        for split,v in result['timings'].items():
            print(split,'rough full POINT scoring estimate (bootstrap excluded):',round(v['rough_full_point_estimate_seconds']/60,1),'minutes')
    return 1 if result['status']=='ERROR' else 0

if __name__=='__main__':sys.exit(main())
