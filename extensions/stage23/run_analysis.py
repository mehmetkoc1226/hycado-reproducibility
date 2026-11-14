"""CPU-only paired COCO uncertainty and frozen max500 recall-FP analysis."""
from __future__ import annotations
import argparse,contextlib,csv,hashlib,io,json,os,sqlite3,sys,time,traceback,unittest,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
METHODS=['yolo1280','hybrid1280','yolo1280_rois800']
CONTRASTS=[('hybrid1280_minus_yolo1280',1,0),('yolo1280_rois800_minus_yolo1280',2,0),('hybrid1280_minus_yolo1280_rois800',1,2)]
RUN='coco_uncertainty_recall_v1_seed42'
DRAWS=1000

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def need(ok,msg):
    if not ok:raise ValueError(msg)
def write(path,body):
    temp=path.with_name(path.name+'.partial');temp.write_text(json.dumps(body,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,path)
def readonly(path):
    path=Path(path).resolve();need(path.is_file(),'Source file missing: '+str(path))
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=30);db.execute('PRAGMA query_only=ON');return db
def stat(p):
    s=p.stat();return (s.st_size,s.st_mtime_ns)

def open_state(path,signature):
    db=sqlite3.connect(path,timeout=30)
    try:
        db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL')
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS binding(id INTEGER PRIMARY KEY,signature TEXT)')
            db.execute('INSERT OR IGNORE INTO binding VALUES(1,?)',(signature,))
            need(db.execute('SELECT signature FROM binding WHERE id=1').fetchone()==(signature,),'Analysis cache belongs to different inputs/code; preserve it and report this error')
            db.execute('CREATE TABLE IF NOT EXISTS draws(draw INTEGER PRIMARY KEY,digest TEXT,payload TEXT)')
    except BaseException:
        db.close();raise
    return db

def save_draw(db,i,value):
    payload=json.dumps(value,separators=(',',':'),allow_nan=False)
    h=hashlib.sha256(payload.encode()).hexdigest()
    with db:db.execute('INSERT INTO draws VALUES(?,?,?)',(i,h,payload))

def get_draw(db,i):
    r=db.execute('SELECT digest,payload FROM draws WHERE draw=?',(i,)).fetchone()
    if r is None:return None
    need(hashlib.sha256(r[1].encode()).hexdigest()==r[0],'Bootstrap draw payload changed: '+str(i))
    return json.loads(r[1])

def validate_coco_cache(db,signature,image_ids):
    expected=fp({'signature':signature,'mode':'full'})
    need(db.execute('SELECT signature FROM binding WHERE id=1').fetchone()==(expected,),'Completed 22B cache signature differs')
    expected_ids=set(image_ids)
    for method in METHODS:
        rows=db.execute('SELECT image_id,marker FROM done WHERE split=? AND method=?',('testdev',method)).fetchall()
        need({r[0] for r in rows}==expected_ids and len(rows)==len(image_ids),'Completed 22B image coverage differs: '+method)
        for image,marker in rows:
            parts=list(db.execute('SELECT category,area,digest FROM entries WHERE split=? AND method=? AND image_id=? ORDER BY category,area',('testdev',method,image)))
            need(fp(parts)==marker,'22B cache image-entry coverage changed: '+method+'/'+str(image))

def load_category(db,method,category,image_ids):
    import coco_core as c
    allowed=set(image_ids);out={}
    for image,digest,payload in db.execute('SELECT image_id,digest,payload FROM entries WHERE split=? AND method=? AND category=? AND area=0 ORDER BY image_id',('testdev',method,category)):
        need(image in allowed,'Unknown image in COCO cache')
        d=c.decode(digest,payload)
        need(d['image_id']==image and d['category_id']==category and d['area_index']==0,'COCO cache payload identity differs')
        need(image not in out,'Duplicate COCO cache entry');out[image]=d
    return out

def compile_coco(db,image_ids):
    import uncertainty_core as u
    engine=[]
    for method in METHODS:
        blocks=[]
        for cat in range(1,11):
            blocks.append(u.compile_category(load_category(db,method,cat,image_ids),image_ids))
        engine.append(blocks);print('COCO cache compiled:',method,flush=True)
    return engine

def compile_recall(db,records):
    import numpy as np
    import recall_core as r
    labels={x['stem']:x['sha256'] for x in read(HERE/'references/annotation_index.json')['records']}
    expected=read(HERE/'references/stage20_metrics.json')['methods']
    chunks={m:{'scores':[],'flags':[],'images':[]} for m in METHODS};counts=[]
    for i,rec in enumerate(records):
        row=db.execute('SELECT ann_sha,digest,payload FROM matching WHERE stem=?',(rec['stem'],)).fetchone()
        need(row is not None and row[0]==labels[rec['stem']] and hashlib.sha256(row[2]).hexdigest()==row[1],'Frozen stage20 matching identity differs: '+rec['stem'])
        with np.load(io.BytesIO(row[2]),allow_pickle=False) as z:
            g=z['gt_denominators'];need(g.shape==(10,) and np.issubdtype(g.dtype,np.integer) and np.all(g>=0),'Frozen GT count invalid')
            counts.append(int(g.sum()))
            for method in METHODS:
                scores=z[method+'_scores'];flags=z[method+'_flags'];classes=z[method+'_classes']
                need(scores.ndim==1 and len(scores)<=500 and flags.shape==(len(scores),10) and classes.shape==scores.shape,'Frozen max500 match dimensions differ')
                need(np.isfinite(scores).all() and np.all((scores>=0)&(scores<=1)) and np.all(np.diff(scores)<=0),'Frozen score order/range differs')
                need(np.isin(flags,[-1,0,1]).all() and np.isin(classes,range(1,11)).all(),'Frozen match flags/classes differ')
                chunks[method]['scores'].append(scores.copy());chunks[method]['flags'].append(flags[:,[0,5]].astype(np.int8))
                chunks[method]['images'].append(np.full(len(scores),i,dtype=np.int32))
        if i in (0,len(records)-1) or (i+1)%250==0:print('Frozen recall matches:',i+1,'/',len(records),flush=True)
    engines=[]
    need(sum(counts)==74707,'Frozen recall GT denominator differs from completed 22B')
    for method in METHODS:
        a=chunks[method];scores=np.concatenate(a['scores']);flags=np.concatenate(a['flags']);images=np.concatenate(a['images'])
        for t,key in enumerate(['0.5','0.75']):
            observed={'TP':int(np.count_nonzero(flags[:,t]==1)),'FP':int(np.count_nonzero(flags[:,t]==0)),
                      'ignored_detections':int(np.count_nonzero(flags[:,t]==-1))}
            need(observed==expected[method]['counts_at_fixed_max500'][key],'Recall TP/FP point parity differs: '+method+'/'+key)
        engines.append(r.compile_curve(scores,flags,images,counts))
    return engines

def describe(point,samples):
    import numpy as np
    low,high=np.quantile(samples,[.025,.975],axis=0,method='linear')
    return {'point':float(point),'lower95':float(low),'upper95':float(high),'bootstrap_SE':float(np.std(samples,ddof=1))}

def finish(point,recall_point,coco,recall,curves,out):
    import numpy as np
    import uncertainty_core as u
    import recall_core as r
    need(coco.shape==(DRAWS,3,4) and recall.shape==(DRAWS,3,2,5),'Bootstrap coverage/shape differs')
    need(np.isfinite(coco).all() and np.isfinite(recall).all(),'Undefined bootstrap replicate; report required, no silent omission')
    ci={'replicates_requested':DRAWS,'valid_replicates':DRAWS,'proxy_cluster_count':61,
        'interval':'paired proxy-cluster percentile, numpy linear 2.5/97.5 quantiles',
        'primary':'hybrid1280_minus_yolo1280 / COCO AP maxDets100 area all',
        'scope':'fixed models; previously examined exploratory test-dev; filename/hash groups are not verified independent scenes; no training-seed uncertainty',
        'multiplicity':'marginal intervals, no multiple-comparison correction; secondary/recall intervals descriptive',
        'cap500':'point sensitivity already in stage22; no cap500 bootstrap in this bounded analysis',
        'area_AP':'stage22 point metrics retained; no size-specific bootstrap here',
        'methods':{m:{k:describe(point[j,kid],coco[:,j,kid]) for kid,k in enumerate(u.KEYS)} for j,m in enumerate(METHODS)},
        'differences_pp':{name:{k:describe(point[a,kid]-point[b,kid],coco[:,a,kid]-coco[:,b,kid]) for kid,k in enumerate(u.KEYS)} for name,a,b in CONTRASTS},
        'recall_at_FP':{}}
    for tid,t in enumerate(r.IOU):
        tkey=str(t);ci['recall_at_FP'][tkey]={}
        for bid,budget in enumerate(r.BUDGETS):
            ci['recall_at_FP'][tkey][str(budget)]={
                'methods':{m:describe(recall_point[j,tid,bid],recall[:,j,tid,bid]) for j,m in enumerate(METHODS)},
                'differences_pp':{name:describe(recall_point[a,tid,bid]-recall_point[b,tid,bid],recall[:,a,tid,bid]-recall[:,b,tid,bid]) for name,a,b in CONTRASTS}}
    write(out/'intervals.json',ci)
    with (out/'recall_curves.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['method','IoU','score_threshold','TP','FP','FP_per_image','recall_percent','normal_GT','images'])
        for j,m in enumerate(METHODS):
            for tid,t in enumerate(r.IOU):
                c=curves[j][tid]
                for n,score in enumerate(c['scores']):w.writerow([m,t,score,c['TP'][n],c['FP'][n],c['FP_per_image'][n],c['recall_percent'][n],c['normal_GT'],c['images']])
    with (out/'recall_budgets.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['IoU','FP_per_image_budget','method_or_contrast','point','lower95','upper95'])
        for t,bybudget in ci['recall_at_FP'].items():
            for budget,item in bybudget.items():
                for family in ['methods','differences_pp']:
                    for name,value in item[family].items():w.writerow([t,budget,name,value['point'],value['lower95'],value['upper95']])
    with (out/'bootstrap_draws.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['draw','method']+u.KEYS+['recall_iou'+str(t)+'_fp'+str(b) for t in r.IOU for b in r.BUDGETS])
        for i in range(DRAWS):
            for j,m in enumerate(METHODS):w.writerow([i,m,*coco[i,j].tolist(),*recall[i,j].ravel().tolist()])
    return ci

def execute(root,out,env):
    import numpy as np
    import uncertainty_core as u
    import recall_core as r
    import resampling as b
    for name,h in read(HERE/'package_sha256.json').items():need(sha(HERE/name)==h,'Package file changed: '+name)
    import test_analysis
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_analysis))
    need(tests.wasSuccessful(),'Reference/regression tests failed')
    expected=read(HERE/'references/22_point_results.json')
    need(expected['status']=='COCO_POINT_ESTIMATES_COMPLETE','Completed 22B reference required')
    need(env==expected['config']['environment'],'Evaluator/NumPy environment differs from completed 22B')
    source22=root/'experiments/supplementary_coco_v2_seed42'
    need(read(source22/'point_results.json')==expected,'Local completed 22B result differs from supplied report')
    cache=source22/'evaluation_cache.sqlite3'
    stage20=root/'experiments/testdev1280_extension_v1_seed42/predictions.sqlite3'
    source_stats={str(p):stat(p) for p in [cache,stage20,source22/'point_results.json']}
    print('Checking full source cache/database SHA256...',flush=True)
    cache_sha=sha(cache);stage20_sha=sha(stage20)
    need(stage20_sha==expected['config']['source_database_sha256']['20'],'Frozen source20 database changed')
    records=sorted(read(HERE/'references/dataset_manifest.json')['records'],key=lambda x:x['image'])
    need(len(records)==1610 and len({x['stem'] for x in records})==1610,'Test-dev record identity differs')
    groups=read(HERE/'references/bootstrap_groups.json')
    need(groups==b.make_groups(records) and len(groups['labels'])==61,'Previously frozen proxy groups differ')
    config={'protocol_sha256':sha(HERE/'protocol.json'),'environment':env,
            'source22_result_fingerprint':fp(expected),'source22_cache_sha256':cache_sha,
            'source20_database_sha256':stage20_sha,
            'package_sha256':sha(HERE/'package_sha256.json')}
    signature=fp(config);write(out/'source_bindings.json',config)
    ids=list(range(1,len(records)+1))
    with contextlib.closing(readonly(cache)) as db:
        validate_coco_cache(db,expected['signature'],ids)
        engine=compile_coco(db,ids)
    frequency=np.ones(len(records),dtype=np.int64)
    point=u.evaluate(engine,frequency)
    for j,method in enumerate(METHODS):
        m=expected['metrics']['testdev'][method]['100']
        need(np.allclose(point[j],[m['AP'],m['AP50'],m['AP75'],m['AR']],rtol=0,atol=1e-8),'COCO full-dataset point parity failed: '+method)
    den=np.stack([x['counts'] for x in engine[0]])
    need(int(den.sum())==74707,'COCO cached GT denominator differs')
    for method in engine[1:]:need(np.array_equal(den,np.stack([x['counts'] for x in method])),'COCO cached GT counts differ across methods')
    print('Full 1610-image COCO point parity: PASS',flush=True)
    with contextlib.closing(readonly(stage20)) as db:recall_engine=compile_recall(db,records)
    need(np.array_equal(den.sum(axis=0),recall_engine[0]['gt_counts']),'COCO/recall normal GT counts differ by image')
    recall_values=[r.curve_values(x,frequency,full=True) for x in recall_engine]
    recall_point=np.stack([x[0] for x in recall_values]);curves=[x[1] for x in recall_values]
    write(out/'point.json',{'methods':METHODS,'coco_keys':u.KEYS,'coco':point.tolist(),
                          'recall_budgets':r.BUDGETS,'recall_iou':r.IOU,'recall_at_FP':recall_point.tolist(),
                          'full_COCO_parity_22B':True,'frozen_recall_TP_FP_parity_stage20':True,
                          'normal_GT':74707,'images':1610})
    values=[];start=time.perf_counter();new=0;reused=0
    with contextlib.closing(open_state(out/'bootstrap.sqlite3',signature)) as state:
        for i in range(DRAWS):
            item=get_draw(state,i)
            if item is None:
                freq=b.frequencies(groups,'cluster',i,seed=20261102)
                cp=u.evaluate(engine,freq)
                rp=np.stack([r.curve_values(x,freq)[0] for x in recall_engine])
                need(np.isfinite(cp).all() and np.isfinite(rp).all(),'Undefined bootstrap draw: '+str(i))
                item={'coco':cp.tolist(),'recall':rp.tolist()};save_draw(state,i,item);new+=1
            else:reused+=1
            cp=np.asarray(item['coco']);rp=np.asarray(item['recall'])
            need(cp.shape==(3,4) and rp.shape==(3,2,5) and np.isfinite(cp).all() and np.isfinite(rp).all(),'Stored draw shape/value differs')
            values.append(item)
            if i in (0,4,DRAWS-1) or (i+1)%10==0:
                seconds=time.perf_counter()-start
                remaining=(DRAWS-i-1)*seconds/new if new else None
                print('23 BOOTSTRAP '+str(i+1)+'/'+str(DRAWS)+'; new='+str(new)+'; reused='+str(reused)+
                      ('; rough remaining ~'+str(round(remaining/60,1))+' min' if remaining is not None else ''),flush=True)
                write(out/'progress.json',{'draws_completed':i+1,'draws_total':DRAWS,'new':new,'reused':reused,
                                          'rough_remaining_seconds':remaining,'training':False,'gpu':False})
    cp=np.asarray([v['coco'] for v in values]);rp=np.asarray([v['recall'] for v in values])
    ci=finish(point,recall_point,cp,rp,curves,out)
    for p,old in source_stats.items():need(stat(Path(p))==old,'Source cache/database changed while analysis was active')
    result={'status':'COCO_UNCERTAINTY_RECALL_COMPLETE','signature':signature,'source_bindings':config,
            'images':1610,'normal_GT':74707,'methods':METHODS,'tests_passed':tests.testsRun,
            'replicates_completed':DRAWS,'proxy_groups':61,'COCO_point_parity_22B':True,
            'recall_TP_FP_parity_stage20':True,'training':False,'gpu':False,
            'primary_COCO_AP_difference':ci['differences_pp']['hybrid1280_minus_yolo1280']['AP'],
            'recall_matching':'existing toolkit-aligned global-max500 matches; complete pooled score ties',
            'COCO_bootstrap':'maxDets100 area all; canonical image-order replication',
            'interpretation':'conditional on fixed models and previously examined test-dev; no independent confirmation or training-seed CI',
            'bootstrap_seconds_this_invocation':time.perf_counter()-start,'source_files_unchanged':True}
    write(out/'result.json',result);return result

def bundle(out,result):
    path=HERE/'23_coco_guven_recall_rapor.zip';temp=path.with_suffix('.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('result.json',json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
        for name in ['source_bindings.json','point.json','intervals.json','recall_curves.csv','recall_budgets.csv','bootstrap_draws.csv','progress.json']:
            if out is not None and (out/name).exists():z.write(out/name,name)
        z.write(HERE/'protocol.json','protocol.json');z.write(HERE/'references/bootstrap_groups.json','bootstrap_groups.json')
    os.replace(temp,path);return path

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path(r'D:\HyCADO_Temiz_Baslangic'))
    args=p.parse_args();out=None
    try:
        root=args.project.resolve();need(root.is_dir(),'Project directory missing: '+str(root))
        out=root/'experiments'/RUN;out.mkdir(parents=True,exist_ok=True)
        import bootstrap_vendor
        env=bootstrap_vendor.ensure()
        result=execute(root,out,env)
    except KeyboardInterrupt:
        result={'status':'INTERRUPTED','message':'Completed draws preserved; rerun the same CMD','training':False,'gpu':False}
    except Exception as exc:
        traceback.print_exc();result={'status':'ERROR','error':str(exc),'traceback':traceback.format_exc(),'training':False,'gpu':False}
    path=bundle(out,result)
    print('SONUC:',result['status']);print('GONDERILECEK RAPOR:',path)
    return 0 if result['status']=='COCO_UNCERTAINTY_RECALL_COMPLETE' else 1

if __name__=='__main__':sys.exit(main())
