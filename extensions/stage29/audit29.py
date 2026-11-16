"""Read-only raw crop audit and geometric diagnostics. No GT, model or AP imports."""
from __future__ import annotations
import argparse,contextlib,csv,hashlib,json,os,sqlite3,sys,time,traceback,unittest,zipfile,zlib
from pathlib import Path
import numpy as np
import frozen_cpu as f
import trace29 as t

HERE=Path(__file__).resolve().parent
REF=HERE/'references'
RUN='raw_crop_audit_v1_seed42'

def source_doc(db,method,rec,index):
    item=db.execute('SELECT digest,payload FROM prediction WHERE method=? AND stem=?',(method,rec['stem'])).fetchone()
    f.need(item is not None,'Missing frozen prediction: '+method+'/'+rec['stem'])
    plain=zlib.decompress(item[1])
    f.need(hashlib.sha256(plain).hexdigest()==item[0]==index[rec['stem']]['variant_sha256'][method],'Frozen prediction hash differs')
    doc=json.loads(plain);f.need(doc['image_sha256']==rec['image_sha256'],'Image binding differs')
    f.validate_rows(doc['detections'],rec)
    f.need(f.canonical(doc['detections'],rec['width'],rec['height'])==doc['detections'],'Frozen candidate order differs')
    return doc

def open_state(path,signature):
    db=sqlite3.connect(path,timeout=30)
    try:
        db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL')
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS binding(id INTEGER PRIMARY KEY,signature TEXT)')
            db.execute('INSERT OR IGNORE INTO binding VALUES(1,?)',(signature,))
            f.need(db.execute('SELECT signature FROM binding WHERE id=1').fetchone()==(signature,),'Audit cache belongs to different inputs/code; preserve it and report this error')
            db.execute('CREATE TABLE IF NOT EXISTS summaries(stem TEXT PRIMARY KEY,digest TEXT,payload BLOB)')
        return db
    except BaseException:db.close();raise

def audit_one(rec,db18,db20,index18,index20,signature):
    source={m:source_doc(db18,m,rec,index18) for m in ['yolo800','yolo1280','hybrid_a025']}
    target={m:source_doc(db20,m,rec,index20) for m in ['yolo1280','hybrid800_reference','hybrid1280','yolo1280_rois800']}
    f.need(target['yolo1280']['detections']==source['yolo1280']['detections'],'Source18/20 baseline rows differ')
    f.need(target['hybrid800_reference']['detections']==source['hybrid_a025']['detections'],'Original HyCADO-800 replay source differs')
    r800=t.plan(target['hybrid800_reference']['rects'],rec);r1280=t.plan(target['hybrid1280']['rects'],rec)
    f.need(source['hybrid_a025']['rects']==r800 and target['yolo1280_rois800']['rects']==r800,'Frozen plan reuse differs')
    f.need(len(r800)==index20[rec['stem']]['crops800'] and len(r1280)==index20[rec['stem']]['crops1280'],'Crop counts differ from signed index')
    union=[list(r) for r in dict.fromkeys(tuple(r) for r in r800+r1280)]
    raw=t.raw_queries(db20,rec,union,index20[rec['stem']]['raw_sha256'])
    f.need(t.replay(raw,union,r800,source['yolo800']['detections'],rec)==source['hybrid_a025']['detections'],
           'Raw replay differs from original HyCADO-800: '+rec['stem'])
    f.need(t.replay(raw,union,r800,source['yolo1280']['detections'],rec)==target['yolo1280_rois800']['detections'],
           'Raw replay differs from mixed-resolution control: '+rec['stem'])
    result=t.image_summary(rec,raw,union,r1280,source['yolo1280']['detections'],target['hybrid1280']['detections'])
    result.update({'signature':signature,'image_sha256':rec['image_sha256'],'raw_archive_sha256':index20[rec['stem']]['raw_sha256'],
                   'exact_HyCADO800_replay':True,'exact_mixed_resolution_replay':True,'raw_queries_all_300_retained':True})
    return result

def aggregate(summaries,out):
    from collections import Counter
    totals=Counter();groups={};examples=[];visible_examples=[]
    for item in summaries:
        totals.update(item['counts']);examples.extend(item['largest_retained_box_crop_ratios']);visible_examples.extend(item['largest_display_box_crop_ratios'])
        for k,v in item['distributions'].items():
            if k not in groups:
                groups[k]={x:0 for x in ['count','area_ratio_sum','ratio_ge_0_5','ratio_ge_0_8','ratio_ge_0_95','touches_at_least_two_crop_edges','touches_all_four_crop_edges']}
                groups[k].update({'area_ratio_histogram':[0]*(len(t.BINS)-1),'class_counts':[0]*10,'max_scaled_score':None})
            g=groups[k]
            for key in ['count','area_ratio_sum','ratio_ge_0_5','ratio_ge_0_8','ratio_ge_0_95','touches_at_least_two_crop_edges','touches_all_four_crop_edges']:g[key]+=v[key]
            g['area_ratio_histogram']=[a+b for a,b in zip(g['area_ratio_histogram'],v['area_ratio_histogram'])]
            g['class_counts']=[a+b for a,b in zip(g['class_counts'],v['class_counts'])]
            if v['max_scaled_score'] is not None:g['max_scaled_score']=max(g['max_scaled_score'] or 0,v['max_scaled_score'])
    for g in groups.values():
        f.need(sum(g['area_ratio_histogram'])==sum(g['class_counts'])==g['count'],'Histogram/class total differs')
        g['mean_box_crop_area_ratio']=g['area_ratio_sum']/g['count'] if g['count'] else None
        g['fraction_ratio_ge_0_95']=g['ratio_ge_0_95']/g['count'] if g['count'] else None
    ordered=sorted(examples,key=lambda x:(-x['box_crop_area_ratio'],-x['scaled_score'],x['image'],x['crop_index'],x['query_id']))
    # Prespecified extreme-geometry examples, not an accuracy-ranked sample.
    extreme=ordered[:40]
    visible=sorted(visible_examples,key=lambda x:(-x['box_crop_area_ratio'],-x['scaled_score'],x['image'],x['crop_index'],x['query_id']))[:40]
    doc={'method':'HyCADO-1280','alpha':.25,'images':len(summaries),'total_selected_crops':sum(s['crop_count'] for s in summaries),
         'total_union_crops':sum(s['raw_union_crop_count'] for s in summaries),'total_union_queries':sum(s['full_queries_in_union'] for s in summaries),
         'counts':dict(totals),'distributions':groups,'area_ratio_histogram_edges':t.BINS.tolist(),
         'class_names_zero_based':t.CLASS_NAMES,'geometry_examples_all_scores':extreme,'geometry_examples_display_ge_0_05':visible,
         'interpretation':'Descriptive box/crop geometry and exact NMS source attribution. A crop-filling box is not automatically an artefact or false positive; no GT or AP is evaluated.',
         'example_selection':'top12 retained crop boxes per image by area ratio then score; separate top12 after display-score filtering; merge each set and retain top40; fixed before inspecting actual values'}
    # Cached summaries use canonical JSON key order; exports must remain
    # byte-identical whether they came from a fresh calculation or resume.
    doc=json.loads(json.dumps(doc,sort_keys=True,allow_nan=False))
    f.write(out/'geometry_summary.json',doc)
    with (out/'per_image.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['image','crops1280','raw_union_crops','pre_nms_crop_candidates','retained_crop','retained_crop_display_ge_0_05','retained_crop_ratio_ge_0_95','global_retained','global_suppressed_by_crop']
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
        for s in summaries:
            d=s['distributions'];w.writerow({'image':s['image'],'crops1280':s['crop_count'],'raw_union_crops':s['raw_union_crop_count'],
                'pre_nms_crop_candidates':d['pre_nms_crop']['count'],'retained_crop':d['retained_crop']['count'],
                'retained_crop_display_ge_0_05':d['retained_crop_display_ge_0_05']['count'],
                'retained_crop_ratio_ge_0_95':d['retained_crop']['ratio_ge_0_95'],'global_retained':s['counts']['global_retained'],
                'global_suppressed_by_crop':s['counts'].get('global_suppressed_by_crop',0)})
    return doc

def perform(rows,db18,db20,index18,index20,cache,signature,out):
    summaries=[];new=0;start=time.perf_counter()
    for i,rec in enumerate(rows,1):
        saved=cache.execute('SELECT digest,payload FROM summaries WHERE stem=?',(rec['stem'],)).fetchone()
        if saved:
            value=f.unpack(*saved)
            f.need(value['signature']==signature and value['image']==rec['image'] and value['image_sha256']==rec['image_sha256'] and
                   value['raw_archive_sha256']==index20[rec['stem']]['raw_sha256'],'Cached audit binding differs')
            f.need(all(value[k] for k in ['exact_HyCADO1280_replay','exact_HyCADO800_replay','exact_mixed_resolution_replay']),'Cached replay not complete')
        else:
            value=audit_one(rec,db18,db20,index18,index20,signature);h,p=f.pack(value)
            with cache:cache.execute('INSERT INTO summaries VALUES(?,?,?)',(rec['stem'],h,p))
            new+=1
        summaries.append(value)
        if i in (1,5,len(rows)) or i%25==0:
            rem=(len(rows)-i)*(time.perf_counter()-start)/new if new else None
            print(f'29 RAW + EXACT REPLAY: {i}/{len(rows)}; new={new}; reused={i-new}'+(f'; remaining ~{rem/60:.1f} min' if rem is not None else ''),flush=True)
            f.write(out/'progress.json',{'phase':'RAW_AUDIT_AND_EXACT_REPLAY','done':i,'total':len(rows),'new':new,'reused':i-new,
                                       'rough_remaining_seconds':rem,'training':False,'gpu_inference':False})
    return summaries

def execute(root,out):
    for name,h in f.read(HERE/'package_sha256.json').items():f.need(f.sha(HERE/name)==h,'Package file changed: '+name)
    import test_audit29
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_audit29))
    f.need(tests.wasSuccessful() and tests.testsRun>0,'Self-tests failed')
    rows=sorted(f.read(REF/'stage18/dataset_manifest.json')['records'],key=lambda r:r['image'])
    f.need(len(rows)==len({r['stem'] for r in rows})==1610,'Frozen image coverage differs')
    indices={s:{Path(r['image']).stem:r for r in f.read(REF/f'stage{s}/prediction_index.json')['records']} for s in (18,20)}
    expected=f.read(REF/'22_point_results.json')['config']['source_database_sha256']
    paths={18:root/'experiments/testdev_frozen_v1_seed42/predictions.sqlite3',20:root/'experiments/testdev1280_extension_v1_seed42/predictions.sqlite3'}
    sources={}
    for s,p in paths.items():
        print('Checking frozen source SHA256:',p,flush=True)
        h=f.sha(p);f.need(h==expected[str(s)],'Frozen source database differs: '+str(p))
        sources[str(s)]={'path':str(p),'sha256':h,'stat':[p.stat().st_size,p.stat().st_mtime_ns]}
    config={'protocol_sha256':f.sha(HERE/'protocol.json'),'package_manifest_sha256':f.sha(HERE/'package_sha256.json'),
            'source_database_sha256':{s:v['sha256'] for s,v in sources.items()},'numpy':np.__version__,'python':list(sys.version_info[:3])}
    signature=f.fp(config);f.write(out/'source_bindings.json',{'signature':signature,'config':config,'sources':sources})
    with contextlib.ExitStack() as stack:
        dbs={s:stack.enter_context(contextlib.closing(f.readonly(p))) for s,p in paths.items()}
        schemas={s:[list(x) for x in db.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name")] for s,db in dbs.items()}
        f.write(out/'source_schemas.json',schemas)
        f.need(any(x[0]=='crop_raw' for x in schemas[20]),'Source20 lacks crop_raw; alpha sensitivity is not established')
        count=dbs[20].execute('SELECT count(*) FROM crop_raw').fetchone()[0]
        f.need(count==1610,'Raw crop archive image coverage differs')
        state=stack.enter_context(contextlib.closing(open_state(out/'audit.sqlite3',signature)))
        summaries=perform(rows,dbs[18],dbs[20],indices[18],indices[20],state,signature,out)
        geometry=aggregate(summaries,out)
    for s,p in paths.items():
        v=sources[str(s)];f.need([p.stat().st_size,p.stat().st_mtime_ns]==v['stat'] and f.sha(p)==v['sha256'],'Source database changed during audit')
    index={'signature':signature,'records':[{'image':s['image'],'image_sha256':s['image_sha256'],'raw_archive_sha256':s['raw_archive_sha256'],
           'exact_HyCADO800_replay':True,'exact_HyCADO1280_replay':True,'exact_mixed_resolution_replay':True} for s in summaries]}
    f.write(out/'verified_raw_index.json',index)
    result={'status':'RAW29_AUDIT_COMPLETE','signature':signature,'images':1610,'raw_archives_sha256_verified':1610,
            'tests_passed':tests.testsRun,'exact_replays':{'hybrid800':1610,'hybrid1280':1610,'global1280_regions800':1610},
            'alpha_sensitivity_possible_without_inference':True,'reason':'All300 raw queries per crop, exact ordered geometry, fixed baselines and alpha=.25 replay verified',
            'alpha_sweep_or_AP_computed_in_this_audit':False,'training':False,'gpu_inference':False,'GT_read':False,
            'original_databases_unchanged':True,'verified_raw_index_sha256':f.fp(index),
            'crop_filling_boxes_are_not_automatically_false_positives':True,
            'retained_crop_boxes':geometry['distributions']['retained_crop']['count'],
            'retained_display_crop_boxes_ratio_ge_0_95':geometry['distributions']['retained_crop_display_ge_0_05']['ratio_ge_0_95']}
    f.write(out/'result.json',result);return result

def bundle(out,result):
    path=HERE/'29_ham_crop_denetim_rapor.zip';tmp=path.with_suffix('.partial')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('result.json',json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
        for n in ['source_bindings.json','source_schemas.json','verified_raw_index.json','geometry_summary.json','per_image.csv','progress.json']:
            if (out/n).is_file():z.write(out/n,n)
        for n in ['audit29.py','trace29.py','frozen_cpu.py','test_audit29.py','protocol.json','package_sha256.json']:
            z.write(HERE/n,'code/'+n)
    os.replace(tmp,path);return path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=Path(r'D:\HyCADO_Temiz_Baslangic'));args=ap.parse_args()
    root=args.project.resolve();out=root/'experiments'/RUN
    try:
        f.need(root.is_dir(),'Project directory missing');out.mkdir(parents=True,exist_ok=True)
        with f.run_lock(out/'run.lock'):result=execute(root,out)
    except KeyboardInterrupt:
        result={'status':'INTERRUPTED','resume':'Run the same CMD; preserve the new experiments directory','training':False,'gpu_inference':False}
    except Exception as exc:
        result={'status':'ERROR','error':str(exc),'traceback':traceback.format_exc(),'training':False,'gpu_inference':False};traceback.print_exc()
    report=bundle(out,result);print('SONUC:',result['status'],flush=True);print('GONDERILECEK RAPOR:',report,flush=True)
    return 0 if result['status']=='RAW29_AUDIT_COMPLETE' else 1

if __name__=='__main__':sys.exit(main())
