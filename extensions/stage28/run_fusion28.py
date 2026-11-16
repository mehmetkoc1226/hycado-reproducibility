"""Stage 28: exploratory fixed YOLO-800/1280 fusion, from frozen outputs only."""
from __future__ import annotations
import argparse, contextlib, csv, hashlib, json, os, sys, time, traceback, unittest, zipfile
from pathlib import Path
from fusion28 import (METHOD,need,fp,sha,read,write,readonly,run_lock,validate_rows,
                      fuse,state,pack,unpack,source_body,save_entries,entries_done,load_entries)

HERE=Path(__file__).resolve().parent
RUN='yolo800_1280_fusion_v1_seed42'
BASE=['yolo1280','hybrid1280']
ALL=[METHOD]+BASE
KEYS=['AP','AP50','AP75','AR100']
REF=HERE/'references'

def records():
    rows=sorted(read(REF/'stage18/dataset_manifest.json')['records'],key=lambda r:r['image'])
    labels={r['stem']:r['sha256'] for r in read(REF/'stage18/annotation_index.json')['records']}
    need(len(rows)==1610 and len(labels)==1610,'Frozen 1610-image coverage differs')
    ans=[]
    for i,r in enumerate(rows,1):
        need(Path(r['annotation']).name==r['annotation'] and Path(r['image']).stem==r['stem'],'Unsafe annotation/image name')
        ans.append(dict(r,id=i,annotation_sha256=labels[r['stem']]))
    return ans

class Sources:
    def __init__(self,root):
        self.stack=contextlib.ExitStack()
        try:
            self.dbs={s:self.stack.enter_context(contextlib.closing(readonly(root/'experiments'/f/'predictions.sqlite3')))
                      for s,f in [(18,'testdev_frozen_v1_seed42'),(20,'testdev1280_extension_v1_seed42')]}
            self.indices={s:{Path(r['image']).stem:r for r in read(REF/f'stage{s}/prediction_index.json')['records']} for s in (18,20)}
        except BaseException:self.stack.close();raise
    def close(self):self.stack.close()
    def get(self,stage,method,rec):
        return source_body(self.dbs[stage],method,rec,self.indices[stage][rec['stem']]['variant_sha256'][method])

def groundtruth(folder,rec):
    import numpy as np
    import visdrone_preprocess as v
    path=folder/rec['annotation'];need(sha(path)==rec['annotation_sha256'],'Annotation SHA256 differs: '+rec['stem'])
    raw=v.raw_annotation(path.read_text(encoding='utf-8-sig'))
    need(not np.any((raw[:,2]<=0)|(raw[:,3]<=0)),'Nonpositive test-dev GT geometry')
    need(not np.any((raw[:,5]>=1)&(raw[:,5]<=10)&(raw[:,4]==0)),'Unreviewed class-specific ignored GT')
    integral=v.ignore_integral(raw,rec['width'],rec['height'])
    g=v.prepare_groundtruth(raw,integral)
    g=g[(g[:,5]>=1)&(g[:,5]<=10)&(g[:,4]==0)]
    return np.column_stack((g[:,:4],g[:,5])).tolist(),integral

def progress(out,phase,i,n,new,start):
    elapsed=time.perf_counter()-start
    rem=(n-i)*elapsed/new if new else None
    suffix=f'; rough remaining ~{rem/60:.1f} min' if rem is not None else '; reused'
    print(f'28 {phase}: {i}/{n}; new={new}'+suffix,flush=True)
    write(out/'progress.json',{'phase':phase,'done':i,'total':n,'new_this_invocation':new,
                             'rough_remaining_seconds_this_phase':rem,'training':False,'gpu_inference':False})

def frozen_predictions(db,src,rows,signature,out):
    index=[];new=0;start=time.perf_counter()
    for i,rec in enumerate(rows,1):
        y8=src.get(18,'yolo800',rec);y12=src.get(18,'yolo1280',rec)
        need(y12==src.get(20,'yolo1280',rec),'Source18/20 YOLO-1280 rows differ')
        source_digests={m:src.indices[18][rec['stem']]['variant_sha256'][m] for m in ['yolo800','yolo1280']}
        binding={'signature':signature,'image_id':rec['id'],'image_sha256':rec['image_sha256'],'source_digests':source_digests}
        saved=db.execute('SELECT digest,payload FROM predictions WHERE image_id=?',(rec['id'],)).fetchone()
        if saved:
            doc=unpack(*saved);need(doc['binding']==binding,'Fused prediction binding differs')
            validate_rows(doc['detections'],rec)
        else:
            doc={'binding':binding,'detections':fuse(y8,y12,rec['width'],rec['height'])}
            h,p=pack(doc)
            with db:db.execute('INSERT INTO predictions VALUES(?,?,?)',(rec['id'],h,p))
            new+=1
        digest=db.execute('SELECT digest FROM predictions WHERE image_id=?',(rec['id'],)).fetchone()[0]
        index.append({'id':rec['id'],'image':rec['image'],'image_sha256':rec['image_sha256'],
                      'annotation_sha256':rec['annotation_sha256'],'detections':len(doc['detections']),
                      'fused_payload_sha256':digest,'source_digests':source_digests})
        if i in (1,5,len(rows)) or i%50==0:progress(out,'FUSION FREEZE',i,len(rows),new,start)
    freeze={'signature':signature,'method':METHOD,'records':index,'training':False,'gpu_inference':False,
            'construction_uses_annotation_rows':False}
    write(out/'prediction_index.json',freeze)
    return fp(freeze)

def measure(db,src,rows,folder,out):
    import coco_core as c
    import visdrone_preprocess as v
    start=time.perf_counter();new=0;gt_count=0
    for i,rec in enumerate(rows,1):
        gt,integral=groundtruth(folder,rec);gt_count+=len(gt)
        for m in ALL:
            if entries_done(db,m,rec['id']):continue
            if m==METHOD:
                doc=unpack(*db.execute('SELECT digest,payload FROM predictions WHERE image_id=?',(rec['id'],)).fetchone())
                preds=doc['detections']
            else:preds=src.get(20,m,rec)
            dt=v.outside_ignore(v.xyxy_to_det(preds),integral)
            save_entries(db,m,rec,c.one_image(rec,gt,dt.tolist()));new+=1
        if i in (1,5,len(rows)) or i%25==0:
            progress(out,'CPU MATCHING',i,len(rows),new/len(ALL),start)
    return gt_count

def points(db,rows,out):
    import numpy as np
    import coco_core as c
    allowed={r['id'] for r in rows};ids=[r['id'] for r in rows]
    result={m:c.aggregate(ids,lambda cat,area:load_entries(db,m,cat,area,allowed))[0] for m in ALL}
    expected=read(REF/'22_point_results.json')['metrics']['testdev']
    parity={}
    def difference(a,b):
        if a is None or b is None:
            need(a is None and b is None,'Baseline undefined-metric status differs')
            return 0.
        return abs(a-b)
    for m in BASE:
        diffs=[]
        for cap in ('100','500'):
            for k in ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR']:
                diffs.append(difference(result[m][cap][k],expected[m][cap][k]))
            for a,b in zip(result[m][cap]['per_class'],expected[m][cap]['per_class']):
                for k in ['AP','AP50','AP75']:diffs.append(difference(a[k],b[k]))
        parity[m]=max(diffs);need(parity[m]<1e-8,'Recomputed baseline COCO metrics differ: '+m)
    result['sahi640']=read(REF/'24_metrics.json')['sahi640_coco']
    write(out/'point_results.json',{'metrics':result,'baseline_max_abs_difference_pp':parity,
          'sahi_point_values_reused_from_completed24':True,'metric':'COCO-style AP with pinned VisDrone ignore preprocessing'})
    fields=['method','maxDets','AP','AP50','AP75','AP_small','AP_medium','AP_large','AR']
    with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for m,caps in result.items():
            for cap,vals in caps.items():w.writerow({'method':m,'maxDets':cap,**{k:vals[k] for k in fields[2:]}})
    return result

def archived_draws():
    import numpy as np
    with (REF/'24_bootstrap_draws.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    answer={}
    for m in ['yolo1280','hybrid1280','sahi640']:
        selected=[r for r in rows if r['method']==m];by={int(r['draw']):r for r in selected}
        need(len(selected)==len(by)==1000 and set(by)==set(range(1000)),'Archived paired-draw coverage differs')
        answer[m]=np.asarray([[float(by[i][k]) for k in KEYS] for i in range(1000)])
        need(np.isfinite(answer[m]).all(),'Invalid archived draw')
    # Existing 24 comparators must be the exact retained 23 draws, not independent resamples.
    with (REF/'23_bootstrap_draws.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            m=r['method']
            if m in ('yolo1280','hybrid1280'):
                need(np.array_equal(answer[m][int(r['draw'])],np.asarray([float(r[k]) for k in KEYS])),'23/24 paired draws differ')
    return answer

def bootstrap(db,rows,point,out):
    import numpy as np
    import uncertainty_core as u
    import resampling as b
    ids=[r['id'] for r in rows];allowed=set(ids)
    groups=read(REF/'bootstrap_groups.json');need(b.make_groups(rows)==groups,'Retained proxy groups differ')
    engine=[[u.compile_category(load_entries(db,METHOD,cat,0,allowed),ids) for cat in range(1,11)]]
    def vector(m):return np.asarray([point[m]['100'][k] for k in ['AP','AP50','AP75','AR']])
    need(np.allclose(u.evaluate(engine,np.ones(len(rows),dtype=np.int64))[0],vector(METHOD),rtol=0,atol=1e-8),'Fusion point/bootstrap accumulator parity failed')
    old=archived_draws();new=0;start=time.perf_counter();values=[]
    for i in range(1000):
        saved=db.execute('SELECT digest,payload FROM draws WHERE draw=?',(i,)).fetchone()
        if saved:
            need(hashlib.sha256(saved[1].encode()).hexdigest()==saved[0],'Bootstrap draw digest differs')
            val=np.asarray(json.loads(saved[1]),dtype=np.float64)
        else:
            val=u.evaluate(engine,b.frequencies(groups,'cluster',i,seed=20261102))[0]
            need(val.shape==(4,) and np.isfinite(val).all(),'Undefined bootstrap replicate')
            payload=json.dumps(val.tolist(),separators=(',',':'),allow_nan=False)
            with db:db.execute('INSERT INTO draws VALUES(?,?,?)',(i,hashlib.sha256(payload.encode()).hexdigest(),payload))
            new+=1
        need(val.shape==(4,) and np.isfinite(val).all(),'Invalid cached bootstrap replicate');values.append(val)
        if i in (0,4,999) or (i+1)%25==0:progress(out,'PAIRED BOOTSTRAP',i+1,1000,new,start)
    old[METHOD]=np.asarray(values);contrasts={}
    for a,bb in [('hybrid1280',METHOD),(METHOD,'yolo1280'),(METHOD,'sahi640')]:
        label=a+'_minus_'+bb;contrasts[label]={}
        for j,k in enumerate(KEYS):
            delta=old[a][:,j]-old[bb][:,j];lo,hi=np.quantile(delta,[.025,.975],method='linear')
            contrasts[label][k]={'point':float(vector(a)[j]-vector(bb)[j]),'lower95':float(lo),'upper95':float(hi),'bootstrap_SE':float(delta.std(ddof=1))}
    result={'replicates':1000,'seed':20261102,'proxy_clusters':len(groups['labels']),
            'primary_contrast':'hybrid1280_minus_'+METHOD+' / AP maxDets100',
            'differences_pp':contrasts,'interval':'paired percentile; same retained cluster draws as stages23/24',
            'scope':'exploratory, previously examined test-dev; fixed weights; no training-seed uncertainty; proxy groups are not verified scenes',
            'secondary_multiplicity_corrected':False,'area_specific_bootstrap':False}
    write(out/'intervals.json',result)
    with (out/'bootstrap_draws.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['draw','method']+KEYS)
        for i in range(1000):
            for m in ['yolo1280','hybrid1280','sahi640',METHOD]:w.writerow([i,m,*old[m][i].tolist()])
    return result

def execute(root,dataset,out,env):
    for name,h in read(HERE/'package_sha256.json').items():need(sha(HERE/name)==h,'Package file changed: '+name)
    import test_fusion28
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_fusion28))
    need(tests.wasSuccessful() and tests.testsRun>0,'Self-tests failed')
    rows=records();expected=read(REF/'22_point_results.json')
    need(expected['status']=='COCO_POINT_ESTIMATES_COMPLETE','Reference22 not complete')
    need(read(REF/'24_result.json')['status']=='SAHI640_TESTDEV_COMPLETE','Reference24 not complete')
    manifest=read(REF/'stage18/dataset_manifest.json')
    if dataset:
        target=Path(dataset).expanduser().resolve()
        if (target/'VisDrone2019-DET-test-dev/annotations').is_dir():target=target/'VisDrone2019-DET-test-dev'
    else:target=root/'experiments/testdev_frozen_v1_seed42'/('dataset_'+manifest['official_zip_sha256'][:16])
    folder=target/'annotations';need(folder.is_dir(),'Test-dev annotations missing; pass dataset directory as CMD second argument')
    sources={}
    print('Checking full SHA256 of the two frozen prediction databases...',flush=True)
    for stage,run in [(18,'testdev_frozen_v1_seed42'),(20,'testdev1280_extension_v1_seed42')]:
        p=root/'experiments'/run/'predictions.sqlite3';h=sha(p)
        need(h==expected['config']['source_database_sha256'][str(stage)],'Frozen source database differs: '+str(p))
        sources[str(stage)]={'path':str(p),'sha256':h,'stat':[p.stat().st_size,p.stat().st_mtime_ns]}
    config={'protocol_sha256':sha(HERE/'protocol.json'),'package_manifest_sha256':sha(HERE/'package_sha256.json'),
            'environment':env,'source_databases':{k:v['sha256'] for k,v in sources.items()}}
    signature=fp(config);write(out/'source_bindings.json',{'signature':signature,'config':config,'sources':sources})
    with contextlib.closing(state(out/'fusion_and_metrics.sqlite3',signature)) as db:
        src=Sources(root)
        try:
            freeze=frozen_predictions(db,src,rows,signature,out)
            gt_count=measure(db,src,rows,folder,out);need(gt_count==74707,'Test-dev GT count differs')
            point=points(db,rows,out)
            intervals=bootstrap(db,rows,point,out)
        finally:src.close()
    for source in sources.values():
        p=Path(source['path']);need([p.stat().st_size,p.stat().st_mtime_ns]==source['stat'] and sha(p)==source['sha256'],'Source database changed during analysis')
    result={'status':'FUSION28_COMPLETE','signature':signature,'images':1610,'normal_GT':gt_count,
            'tests_passed':tests.testsRun,'prediction_freeze_sha256':freeze,'training':False,'gpu_inference':False,
            'source_databases_unchanged':True,'hyperparameter_search':False,'fusion_COCO_AP100':point[METHOD]['100']['AP'],
            'primary_COCO_difference':intervals['differences_pp']['hybrid1280_minus_'+METHOD]['AP'],
            'interpretation':'Fixed multiresolution late-fusion control; exploratory test-dev, not an untouched confirmation set',
            'timing_scope':'CPU analysis only; no inference latency or equal-compute claim'}
    write(out/'result.json',result);return result

def bundle(out,result):
    path=HERE/'28_cift_cozunurluk_fuzyon_rapor.zip';temp=path.with_suffix('.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('result.json',json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
        for n in ['source_bindings.json','prediction_index.json','point_results.json','metrics.csv','intervals.json','bootstrap_draws.csv','progress.json']:
            if (out/n).is_file():z.write(out/n,n)
        for n in ['protocol.json','package_sha256.json','run_fusion28.py','fusion28.py','test_fusion28.py']:
            z.write(HERE/n,'code/'+n)
    os.replace(temp,path);return path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=Path(r'D:\HyCADO_Temiz_Baslangic'))
    ap.add_argument('--dataset',type=Path);a=ap.parse_args();root=a.project.resolve()
    out=root/'experiments'/RUN
    try:
        need(root.is_dir(),'Project directory missing: '+str(root));out.mkdir(parents=True,exist_ok=True)
        with run_lock(out/'run.lock'):
            import bootstrap_vendor
            env=bootstrap_vendor.ensure()
            result=execute(root,a.dataset,out,env)
    except KeyboardInterrupt:
        result={'status':'INTERRUPTED','training':False,'gpu_inference':False,'resume':'Run the same CMD; preserve the experiments directory'}
    except Exception as exc:
        result={'status':'ERROR','error':str(exc),'traceback':traceback.format_exc(),'training':False,'gpu_inference':False}
        traceback.print_exc()
    path=bundle(out,result)
    print('SONUC:',result['status'],flush=True);print('GONDERILECEK RAPOR:',path,flush=True)
    return 0 if result['status']=='FUSION28_COMPLETE' else 1

if __name__=='__main__':sys.exit(main())
