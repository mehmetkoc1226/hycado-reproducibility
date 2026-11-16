"""Post-hoc descriptive alpha sensitivity. Frozen weights, crops and raw queries."""
from __future__ import annotations
import argparse,contextlib,csv,hashlib,json,os,sys,time,traceback,unittest,zipfile,zlib
from pathlib import Path
import numpy as np
import frozen_cpu as f
import trace29 as t
import alpha30_core as a

HERE=Path(__file__).resolve().parent
REF=HERE/'references'
RUN='alpha_sensitivity_v1_seed42'

def records():
    rows=sorted(f.read(REF/'stage18/dataset_manifest.json')['records'],key=lambda x:x['image'])
    labels={r['stem']:r['sha256'] for r in f.read(REF/'stage18/annotation_index.json')['records']}
    f.need(len(rows)==len(labels)==1610,'Frozen image/annotation coverage differs')
    return [dict(r,id=i,annotation_sha256=labels[r['stem']]) for i,r in enumerate(rows,1)]

def source_doc(db,method,rec,index):
    row=db.execute('SELECT digest,payload FROM prediction WHERE method=? AND stem=?',(method,rec['stem'])).fetchone()
    f.need(row is not None,'Missing frozen output: '+method+'/'+rec['stem'])
    raw=zlib.decompress(row[1]);f.need(hashlib.sha256(raw).hexdigest()==row[0]==index[rec['stem']]['variant_sha256'][method],'Frozen prediction digest differs')
    d=json.loads(raw);f.need(d['image_sha256']==rec['image_sha256'],'Frozen image binding differs');f.validate_rows(d['detections'],rec)
    return d

def progress(out,phase,i,n,new,start):
    rem=(n-i)*(time.perf_counter()-start)/new if new else None
    print(f'30 {phase}: {i}/{n}; new={new}; reused={i-new}'+(f'; remaining ~{rem/60:.1f} min' if rem is not None else ''),flush=True)
    f.write(out/'progress.json',{'phase':phase,'done':i,'total':n,'new':new,'reused':i-new,'rough_remaining_seconds_this_phase':rem,'training':False,'gpu_inference':False})

def freeze(source,cache,rows,index,signature,out):
    evidence=[];new=0;start=time.perf_counter()
    for i,rec in enumerate(rows,1):
        prior={m:a.load_prediction(cache,m,rec,signature) for m,_ in a.GRID}
        if not all(v is not None for v in prior.values()):
            docs={m:source_doc(source,m,rec,index) for m in ['yolo1280','hybrid1280','hybrid800_reference']}
            r8=t.plan(docs['hybrid800_reference']['rects'],rec);r12=t.plan(docs['hybrid1280']['rects'],rec)
            union=[list(r) for r in dict.fromkeys(tuple(r) for r in r8+r12)]
            raw=t.raw_queries(source,rec,union,index[rec['stem']]['raw_sha256'])
            answers=a.variants(raw,union,r12,docs['yolo1280']['detections'],rec)
            f.need(answers['a025']['detections']==docs['hybrid1280']['detections'],'Alpha0.25 does not reproduce frozen HyCADO-1280')
            f.need(answers['a000']['detections']==docs['yolo1280']['detections'],'Global-only endpoint changed')
            with cache:
                for m,value in answers.items():
                    value.update({'signature':signature,'image_id':rec['id'],'image_sha256':rec['image_sha256'],'raw_archive_sha256':index[rec['stem']]['raw_sha256']})
                    if prior[m] is not None:f.need(value==prior[m],'Cached alpha prediction differs from replay')
                    else:
                        h,p=f.pack(value);cache.execute('INSERT INTO predictions VALUES(?,?,?,?)',(m,rec['id'],h,p))
            new+=1
        digests={m:cache.execute('SELECT digest FROM predictions WHERE method=? AND image_id=?',(m,rec['id'])).fetchone()[0] for m,_ in a.GRID}
        evidence.append({'id':rec['id'],'image':rec['image'],'image_sha256':rec['image_sha256'],'raw_sha256':index[rec['stem']]['raw_sha256'],'variant_payload_sha256':digests})
        if i in (1,5,len(rows)) or i%25==0:progress(out,'FROZEN ALPHA OUTPUTS',i,len(rows),new,start)
    result={'signature':signature,'alphas':a.GRID,'records':evidence,'construction_uses_GT':False}
    f.write(out/'prediction_index.json',result);return f.fp(result)

def groundtruth(folder,rec):
    import visdrone_preprocess as v
    p=folder/rec['annotation'];f.need(f.sha(p)==rec['annotation_sha256'],'GT SHA256 differs: '+rec['stem'])
    raw=v.raw_annotation(p.read_text(encoding='utf-8-sig'))
    f.need(not np.any((raw[:,2]<=0)|(raw[:,3]<=0)),'Nonpositive test-dev GT geometry')
    f.need(not np.any((raw[:,5]>=1)&(raw[:,5]<=10)&(raw[:,4]==0)),'Unreviewed class-specific ignored GT')
    integral=v.ignore_integral(raw,rec['width'],rec['height']);g=v.prepare_groundtruth(raw,integral)
    g=g[(g[:,5]>=1)&(g[:,5]<=10)&(g[:,4]==0)]
    return np.column_stack((g[:,:4],g[:,5])).tolist(),integral

def measure(cache,rows,folder,signature,out):
    import coco_core as c
    start=time.perf_counter();new=0;gt_count=0
    for i,rec in enumerate(rows,1):
        gt,integral=groundtruth(folder,rec);gt_count+=len(gt);changed=False
        for m,alpha in a.GRID:
            if alpha==0 or a.done(cache,m,rec['id']):continue
            doc=a.load_prediction(cache,m,rec,signature);f.need(doc is not None,'Missing frozen alpha output')
            dt,source_ids=a.filtered_detections(doc,integral)
            entries=c.one_image(rec,gt,dt.tolist())
            diag=a.geometry_diagnostic(doc,rec,entries,dt,source_ids) if m=='a025' else None
            if diag is not None:diag['signature']=signature
            a.save_evaluation(cache,m,rec,entries,diag);changed=True
        new+=int(changed)
        if i in (1,5,len(rows)) or i%25==0:progress(out,'CPU COCO MATCHING',i,len(rows),new,start)
    return gt_count

def summarize_metrics(cache,rows,out,expected=None):
    import coco_core as c
    expected=expected or f.read(REF/'22_point_results.json')['metrics']['testdev']
    ids=[r['id'] for r in rows];allowed=set(ids);metrics={'a000':expected['yolo1280']}
    for m,alpha in a.GRID:
        if alpha==0:continue
        for rec in rows:f.need(a.done(cache,m,rec['id']),'Incomplete COCO coverage')
        metrics[m]=c.aggregate(ids,lambda cat,area:a.load_entries(cache,m,cat,area,allowed))[0]
    differences=[]
    def delta(x,y):
        if x is None or y is None:f.need(x is None and y is None,'Undefined-metric status differs');return 0.
        return abs(x-y)
    for cap in ['100','500']:
        for k in ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR']:
            differences.append(delta(metrics['a025'][cap][k],expected['hybrid1280'][cap][k]))
        for x,y in zip(metrics['a025'][cap]['per_class'],expected['hybrid1280'][cap]['per_class']):
            for k in ['AP','AP50','AP75']:differences.append(delta(x[k],y[k]))
    maxdiff=max(differences);f.need(maxdiff<1e-8,'Alpha0.25 COCO metrics differ from completed22')
    result={'metrics':metrics,'grid':a.GRID,'alpha0_global_only_points_reused_from22':True,'alpha025_max_abs_difference_pp':maxdiff,
            'confidence_intervals_computed':False,'interpretation':'Descriptive post-hoc sensitivity; all grid values reported; no final alpha selected using this test-dev curve'}
    f.write(out/'point_results.json',result)
    fields=['alpha','method','maxDets','AP','AP50','AP75','AP_small','AP_medium','AP_large','AR']
    with (out/'alpha_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
        for m,alpha in a.GRID:
            for cap,vals in metrics[m].items():w.writerow({'alpha':alpha,'method':m,'maxDets':cap,**{k:vals[k] for k in fields[3:]}})
    with (out/'class_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.writer(stream);w.writerow(['alpha','maxDets','category_id','class_name','AP','AP50','AP75'])
        for m,alpha in a.GRID:
            for cap,vals in metrics[m].items():
                for cl in vals['per_class']:w.writerow([alpha,cap,cl['category_id'],t.CLASS_NAMES[cl['category_id']-1],cl['AP'],cl['AP50'],cl['AP75']])
    return result

def summarize_geometry(cache,rows,signature,out):
    totals={};examples=[];coverage=[]
    for rec in rows:
        saved=cache.execute('SELECT digest,payload FROM diagnostics WHERE image_id=?',(rec['id'],)).fetchone()
        f.need(saved is not None,'Missing alpha0.25 geometry diagnostic')
        d=f.unpack(*saved);f.need(d['signature']==signature and d['image_id']==rec['id'] and d['image_sha256']==rec['image_sha256'],'Geometry cache binding differs')
        coverage.append({'image':rec['image'],'diagnostic_sha256':saved[0]});examples.extend(d['examples_display_crop_filling'])
        for iou,groups in d['counts'].items():
            for name,vals in groups.items():
                target=totals.setdefault(iou,{}).setdefault(name,{'TP':0,'FP':0,'ignored':0})
                for k,v in vals.items():target[k]+=v
    for groups in totals.values():
        for name,vals in groups.items():
            den=vals['TP']+vals['FP'];vals['precision_in_this_diagnostic']=vals['TP']/den if den else None
        for prefix in ['crop','display_crop']:
            for k in ['TP','FP','ignored']:
                f.need(groups[prefix+'_all'][k]==groups[prefix+'_ratio_ge_0_95'][k]+groups[prefix+'_ratio_lt_0_95'][k],'Geometry diagnostic partition totals differ')
    selected={}
    for label in ['TP','FP','ignored']:
        selected[label]=sorted([x for x in examples if x['COCO_IoU_status']['0.5']==label],
                              key=lambda x:(-x['box_crop_area_ratio'],-x['scaled_score'],x['image'],x['crop_index'],x['query_id']))[:30]
    result={'images':len(rows),'alpha':.25,'counts':totals,'examples_display_crop_filling_by_IoU05_status':selected,
            'diagnostic_index_sha256':f.fp(coverage),'metric_scope':'COCO area all, class-aware greedy IoU0.5/0.75 matches; per-class top100 after pinned ignore preprocessing; display subgroup score>=0.05',
            'different_from29':'29 geometry included all frozen rows without ignore filtering or COCO caps; counts here need not equal29',
            'interpretation':'Association between box/crop area ratio and matched TP/FP status; not a causal effect, AP decomposition, or proof all crop-filling boxes are invalid',
            'examples':'fixed top8 per image per IoU0.5 status by area ratio/score; global top30 per status; no success-only selection'}
    f.write(out/'geometry_matching.json',result);return result

def execute(root,dataset,out,env):
    for name,h in f.read(HERE/'package_sha256.json').items():f.need(f.sha(HERE/name)==h,'Package identity differs: '+name)
    import test_alpha30
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_alpha30))
    f.need(tests.wasSuccessful() and tests.testsRun>0,'Self-tests failed')
    gate=f.read(REF/'29_result.json')
    f.need(gate['status']=='RAW29_AUDIT_COMPLETE' and gate['alpha_sensitivity_possible_without_inference'],'Completed raw29 audit required')
    f.need(f.read(root/'experiments/raw_crop_audit_v1_seed42/result.json')==gate,'Local completed29 result differs from supplied report')
    rows=records();index={Path(r['image']).stem:r for r in f.read(REF/'stage20/prediction_index.json')['records']}
    manifest=f.read(REF/'stage18/dataset_manifest.json')
    target=Path(dataset).expanduser().resolve() if dataset else root/'experiments/testdev_frozen_v1_seed42'/('dataset_'+manifest['official_zip_sha256'][:16])
    if (target/'VisDrone2019-DET-test-dev/annotations').is_dir():target=target/'VisDrone2019-DET-test-dev'
    folder=target/'annotations';f.need(folder.is_dir(),'Test-dev annotations missing; pass dataset root as CMD second argument')
    source_path=root/'experiments/testdev1280_extension_v1_seed42/predictions.sqlite3'
    print('Checking full SHA256 of frozen raw-query database...',flush=True)
    h=f.sha(source_path);expected=f.read(REF/'22_point_results.json')['config']['source_database_sha256']['20']
    f.need(h==expected==f.read(REF/'29_source_bindings.json')['sources']['20']['sha256'],'Frozen raw-query database differs')
    source_stat=[source_path.stat().st_size,source_path.stat().st_mtime_ns]
    config={'protocol_sha256':f.sha(HERE/'protocol.json'),'package_manifest_sha256':f.sha(HERE/'package_sha256.json'),
            'source20_sha256':h,'completed29_signature':gate['signature'],'environment':env}
    signature=f.fp(config);f.write(out/'source_bindings.json',{'signature':signature,'config':config,'source20_stat':source_stat})
    with contextlib.closing(f.readonly(source_path)) as source,contextlib.closing(a.state(out/'alpha_predictions_and_metrics.sqlite3',signature)) as cache:
        freeze_sha=freeze(source,cache,rows,index,signature,out)
        gt=measure(cache,rows,folder,signature,out);f.need(gt==74707,'Normal test-dev GT count differs')
        point=summarize_metrics(cache,rows,out);geometry=summarize_geometry(cache,rows,signature,out)
    f.need([source_path.stat().st_size,source_path.stat().st_mtime_ns]==source_stat and f.sha(source_path)==h,'Original raw-query database changed during scoring')
    comparisons={'yolo1280':f.read(REF/'22_point_results.json')['metrics']['testdev']['yolo1280'],
                 'hybrid1280_original_alpha025':f.read(REF/'22_point_results.json')['metrics']['testdev']['hybrid1280'],
                 'sahi640':f.read(REF/'24_metrics.json')['sahi640_coco'],
                 'yolo800_1280_fusion':f.read(REF/'28_point_results.json')['metrics']['yolo800_1280_fusion']}
    f.write(out/'fixed_comparators.json',comparisons)
    result={'status':'ALPHA30_COMPLETE','signature':signature,'images':1610,'normal_GT':gt,'tests_passed':tests.testsRun,
            'alpha_grid':[v for _,v in a.GRID],'AP100_by_alpha':{str(alpha):point['metrics'][m]['100']['AP'] for m,alpha in a.GRID},
            'alpha025_max_abs_difference_pp':point['alpha025_max_abs_difference_pp'],'prediction_freeze_sha256':freeze_sha,
            'original_database_unchanged':True,'training':False,'gpu_inference':False,'new_score_or_geometry_filter':False,
            'confidence_intervals_computed':False,'final_alpha_changed':False,
            'interpretation':'Descriptive sensitivity on previously examined test-dev; existing alpha0.25 result retained; no held-out optimal-alpha claim'}
    f.write(out/'result.json',result);return result

def bundle(out,result):
    path=HERE/'30_alpha_duyarlilik_rapor.zip';temp=path.with_suffix('.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('result.json',json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
        for n in ['source_bindings.json','prediction_index.json','point_results.json','alpha_metrics.csv','class_metrics.csv','geometry_matching.json','fixed_comparators.json','progress.json']:
            if (out/n).is_file():z.write(out/n,n)
        for n in ['run_alpha30.py','alpha30_core.py','trace29.py','frozen_cpu.py','test_alpha30.py','protocol.json','package_sha256.json']:z.write(HERE/n,'code/'+n)
    os.replace(temp,path);return path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=Path(r'D:\HyCADO_Temiz_Baslangic'));ap.add_argument('--dataset',type=Path);args=ap.parse_args()
    root=args.project.resolve();out=root/'experiments'/RUN
    try:
        f.need(root.is_dir(),'Project directory missing');out.mkdir(parents=True,exist_ok=True)
        with f.run_lock(out/'run.lock'):
            import bootstrap_vendor
            env=bootstrap_vendor.ensure();result=execute(root,args.dataset,out,env)
    except KeyboardInterrupt:
        result={'status':'INTERRUPTED','resume':'Run same CMD and preserve the new experiments folder','training':False,'gpu_inference':False}
    except Exception as exc:
        result={'status':'ERROR','error':str(exc),'traceback':traceback.format_exc(),'training':False,'gpu_inference':False};traceback.print_exc()
    report=bundle(out,result);print('SONUC:',result['status'],flush=True);print('GONDERILECEK RAPOR:',report,flush=True)
    return 0 if result['status']=='ALPHA30_COMPLETE' else 1

if __name__=='__main__':sys.exit(main())
