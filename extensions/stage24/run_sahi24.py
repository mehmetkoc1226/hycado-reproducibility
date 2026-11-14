"""Frozen SAHI640 test-dev inference and paired comparison. No training."""
from __future__ import annotations
import argparse,contextlib,csv,hashlib,importlib.metadata,io,json,os,sqlite3,subprocess,sys,time,traceback,unittest,zipfile
from pathlib import Path
from common24 import need,read,sha,fp,write,readonly,state,infer,load_prediction
HERE=Path(__file__).resolve().parent
RUN='sahi640_testdev_v1_seed42'
WEIGHT_SHA='3b61a12a89a448b849bc965e44f00afa92042a5fc0928b4c4f801e0182f84149'

def inspect_images(dataset,records):
    from PIL import Image
    need((dataset/'images').is_dir() and (dataset/'annotations').is_dir(),'Opened test-dev images/annotations directory required')
    for i,rec in enumerate(records,1):
        image=dataset/'images'/(rec['stem']+'.jpg')
        need(sha(image)==rec['image_sha256'],'Image hash differs: '+rec['stem'])
        with Image.open(image) as im:need(im.size==(rec['width'],rec['height']) and im.format=='JPEG','Image dimensions/format differ')
        need((dataset/'annotations'/rec['annotation']).is_file(),'Annotation missing')
        if i in (1,len(records)) or i%400==0:print('Test-dev image verification:',i,'/',len(records),flush=True)

def prepare_env(root,weight):
    os.environ['YOLO_AUTOINSTALL']='false'
    vendor=root/'experiments/p2_strong_baselines_v1_seed42/sahi_vendor'
    need(vendor.is_dir(),'Existing P2 sahi_vendor directory missing; no package installation attempted')
    sys.path[:0]=[str(vendor),str(root)]
    import torch,sahi
    import sahi.models.ultralytics as adapter
    import sahi.predict as predict
    import sahi.postprocess.combine as combine
    need(Path(sahi.__file__).resolve().is_relative_to(vendor.resolve()),'Expected isolated P2 SAHI vendor not imported')
    versions={k:importlib.metadata.version(k) for k in ['torch','torchvision','ultralytics','sahi','numpy','shapely','tqdm']}
    for k,v in {'torch':'2.6.0+cu124','ultralytics':'8.3.203','sahi':'0.11.35','numpy':'1.26.4'}.items():need(versions[k]==v,'Pinned version differs: '+k)
    need(torch.cuda.is_available(),'CUDA required for this inference step')
    need(sha(weight)==WEIGHT_SHA,'Frozen YOLO checkpoint differs')
    try:
        query=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=False)
        driver=query.stdout.strip() if query.returncode==0 else 'unavailable'
    except (OSError,subprocess.TimeoutExpired):driver='unavailable'
    env={'versions':versions,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
         'driver_inventory':driver,
         'sahi_source_sha256':{name:sha(Path(module.__file__)) for name,module in [('adapter',adapter),('predict',predict),('combine',combine)]},
         'backend_defaults':{'cudnn_benchmark':bool(torch.backends.cudnn.benchmark),'cudnn_deterministic':bool(torch.backends.cudnn.deterministic),
                             'matmul_tf32':bool(torch.backends.cuda.matmul.allow_tf32),'cudnn_tf32':bool(torch.backends.cudnn.allow_tf32)}}
    return env

def smoke(root,model,out):
    from p2_sahi_exact import sahi_predict
    from proposal_geometry import canonical_detections
    p2=root/'experiments/p2_strong_baselines_v1_seed42'
    for actual,ref in [('official_val_protocol.json','p2_protocol.json'),('official_val_prediction_index.json','p2_prediction_index.json')]:
        need(sha(p2/actual)==sha(HERE/'references'/ref),'Original P2 metadata differs: '+actual)
    records=sorted(read(HERE/'references/val_images.json')['records'],key=lambda x:x['source_image'])
    index={x['source_image']:x for x in read(HERE/'references/p2_prediction_index.json')['methods']['sahi640']}
    special={'images/val/0000327_04201_d_0000732.jpg','images/val/0000001_04527_d_0000008.jpg'}
    selected={records[i]['source_image'] for i in [0,len(records)//2,len(records)-1]}|special
    report=[]
    for rec in records:
        if rec['source_image'] not in selected:continue
        path=root/'data/prepared'/rec['source_image'];need(sha(path)==rec['sha256'],'VAL smoke image differs')
        item=index[rec['source_image']];source=p2/item['path'];need(sha(source)==item['sha256'],'Frozen SAHI VAL payload differs')
        old=read(source);w,h,rows,dropped=sahi_predict(model,path,640)
        clean,bad=canonical_detections(rows,w,h)
        need((w,h)==(rec['width'],rec['height']) and clean==old['detections'],'SAHI VAL exact numeric replay differs: '+rec['source_image'])
        report.append({'image':rec['source_image'],'exact_canonical_numeric_parity':True,'detections':len(clean),'dropped_zero_area':dropped+bad})
        print('SAHI frozen VAL replay:',len(report),'/',len(selected),'PASS',flush=True)
    need(len(report)==len(selected),'VAL smoke selection coverage differs')
    write(out/'val_smoke.json',{'status':'PASS','images':report,'metric_computed':False,'gt_read':False,'canonical_order':'same original P2 sorting; exact values, no tolerance'})

def pack_part(part):
    import numpy as np
    f=io.BytesIO();np.savez_compressed(f,**part);data=f.getvalue()
    return hashlib.sha256(data).hexdigest(),data

def load_part(digest,payload):
    import numpy as np
    need(hashlib.sha256(payload).hexdigest()==digest,'Native matching cache digest differs')
    with np.load(io.BytesIO(payload),allow_pickle=False) as z:return {k:z[k] for k in z.files}

def measure(db,records,dataset,signature,root,out):
    import numpy as np
    import visdrone_metrics as v
    import toolkit_core as t
    import coco_core as c
    labels={x['stem']:x['sha256'] for x in read(HERE/'references/annotation_index.json')['records']}
    parts=[]
    source20=root/'experiments/testdev1280_extension_v1_seed42/predictions.sqlite3'
    with contextlib.closing(readonly(source20)) as old:
        for i,rec in enumerate(records,1):
            ann=dataset/'annotations'/rec['annotation'];need(sha(ann)==labels[rec['stem']],'Test-dev annotation identity differs')
            cached=db.execute('SELECT ann_sha,digest,payload,coco_marker FROM matches WHERE stem=?',(rec['stem'],)).fetchone()
            if cached:
                need(cached[0]==labels[rec['stem']],'Cached annotation differs')
                part=load_part(cached[1],cached[2])
                rows=list(db.execute('SELECT category,area,digest FROM coco WHERE image_id=? ORDER BY category,area',(i,)))
                need(fp(rows)==cached[3],'COCO cached entry coverage differs')
            else:
                raw=v.raw_annotation(ann.read_text(encoding='utf-8-sig'))
                need(not np.any((raw[:,5]>=1)&(raw[:,5]<=10)&(raw[:,4]==0)),'Class-specific ignored targets need protocol review')
                need(not np.any(raw[:,2:4]<=0),'Nonpositive GT geometry')
                integral=v.ignore_integral(raw,rec['width'],rec['height']);gt=v.prepare_groundtruth(raw,integral)
                body=load_prediction(db,rec,signature)
                det=v.outside_ignore(v.xyxy_to_det(body['detections']),integral)
                capped=det[:500]
                part={'gt_denominators':np.array([np.count_nonzero(gt[:,5]==cat) for cat in range(1,11)],dtype=np.int64)}
                part['presence']=(part['gt_denominators']>0).astype(np.int64)
                part['sahi640_scores']=capped[:,4].copy();part['sahi640_classes']=capped[:,5].astype(np.int8)
                part['sahi640_flags']=np.column_stack([t.fast_flags(capped,gt,float(thr)) for thr in t.THRESHOLDS])
                normal=gt[(gt[:,4]==0)&(gt[:,5]>=1)&(gt[:,5]<=10)]
                native=np.column_stack((normal[:,:4],normal[:,5])).tolist()
                entries=c.one_image({'id':i,'width':rec['width'],'height':rec['height']},native,det.tolist())
                digest,payload=pack_part(part)
                with db:
                    db.execute('DELETE FROM coco WHERE image_id=?',(i,))
                    for e in entries:
                        row,h,data=c.encode(e);db.execute('INSERT INTO coco VALUES(?,?,?,?,?)',(i,row['category_id'],row['area_index'],h,data))
                    mark=list(db.execute('SELECT category,area,digest FROM coco WHERE image_id=? ORDER BY category,area',(i,)))
                    db.execute('INSERT INTO matches VALUES(?,?,?,?,?)',(rec['stem'],labels[rec['stem']],digest,payload,fp(mark)))
            # Whole original database SHA is already checked; still bind GT counts per image.
            original=old.execute('SELECT ann_sha,digest,payload FROM matching WHERE stem=?',(rec['stem'],)).fetchone()
            need(original is not None and original[0]==labels[rec['stem']],'Frozen comparator matching missing')
            ref=load_part(original[1],original[2])
            need(np.array_equal(ref['gt_denominators'],part['gt_denominators']),'GT denominators differ from frozen comparators')
            parts.append(part)
            if i in (1,len(records)) or i%100==0:
                print('24 CPU matching:',i,'/',len(records),flush=True)
                write(out/'progress.json',{'phase':'MATCHING','done':i,'total':len(records),'training':False})
    need(sum(int(x['gt_denominators'].sum()) for x in parts)==74707,'Normal GT total differs')
    toolkit=t.aggregate(parts,'sahi640')
    def loader(cat,area):
        answer={}
        for image,h,payload in db.execute('SELECT image_id,digest,payload FROM coco WHERE category=? AND area=? ORDER BY image_id',(cat,area)):
            e=c.decode(h,payload);need((e['image_id'],e['category_id'],e['area_index'])==(image,cat,area),'COCO match identity differs');answer[image]=e
        return answer
    coco=c.aggregate(list(range(1,len(records)+1)),loader)[0]
    return toolkit,coco,loader

def baseline_draws():
    import numpy as np
    with (HERE/'references/23_bootstrap_draws.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    answer=[]
    for method in ['yolo1280','hybrid1280']:
        r={int(x['draw']):x for x in rows if x['method']==method}
        need(set(r)==set(range(1000)),'Frozen paired baseline bootstrap coverage differs')
        answer.append([[float(r[i][k]) for k in ['AP','AP50','AP75','AR100']] for i in range(1000)])
    return np.asarray(answer).transpose(1,0,2)

def uncertainty(db,records,coco,loader,out):
    import numpy as np
    import uncertainty_core as u
    import resampling as b
    groups=read(HERE/'references/bootstrap_groups.json');need(b.make_groups(records)==groups,'Frozen resampling groups differ')
    engine=[[u.compile_category(loader(cat,0),list(range(1,len(records)+1))) for cat in range(1,11)]]
    point=u.evaluate(engine,np.ones(len(records),dtype=np.int64))[0]
    need(np.allclose(point,[coco['100'][k] for k in ['AP','AP50','AP75','AR']],rtol=0,atol=1e-8),'New SAHI COCO point/CI engine parity failed')
    old=baseline_draws();samples=[];new=0;start=time.perf_counter()
    for i in range(1000):
        saved=db.execute('SELECT digest,payload FROM bootstrap WHERE draw=?',(i,)).fetchone()
        if saved:
            need(hashlib.sha256(saved[1].encode()).hexdigest()==saved[0],'Bootstrap draw hash differs')
            val=np.asarray(json.loads(saved[1]),dtype=float)
        else:
            val=u.evaluate(engine,b.frequencies(groups,'cluster',i,seed=20261102))[0]
            need(np.isfinite(val).all(),'Undefined SAHI bootstrap replicate')
            payload=json.dumps(val.tolist(),separators=(',',':'),allow_nan=False)
            with db:db.execute('INSERT INTO bootstrap VALUES(?,?,?)',(i,hashlib.sha256(payload.encode()).hexdigest(),payload))
            new+=1
        need(val.shape==(4,) and np.isfinite(val).all(),'Bootstrap sample shape differs');samples.append(val)
        if i in (0,4,999) or (i+1)%25==0:
            rem=(999-i)*(time.perf_counter()-start)/new if new else None
            print('24 COCO bootstrap '+str(i+1)+'/1000'+(' ; rough remaining ~'+str(round(rem/60,1))+' min' if rem is not None else ''),flush=True)
            write(out/'progress.json',{'phase':'BOOTSTRAP','done':i+1,'total':1000,'training':False})
    all_samples=np.concatenate([old,np.asarray(samples)[:,None,:]],axis=1)
    parent=read(HERE/'references/23_point.json');points=np.vstack([np.asarray(parent['coco'])[:2],point])
    contrasts={}
    for name,a,bb in [('hybrid1280_minus_sahi640',1,2),('sahi640_minus_yolo1280',2,0)]:
        contrasts[name]={}
        for j,k in enumerate(u.KEYS):
            delta=all_samples[:,a,j]-all_samples[:,bb,j];lo,hi=np.quantile(delta,[.025,.975],method='linear')
            contrasts[name][k]={'point':float(points[a,j]-points[bb,j]),'lower95':float(lo),'upper95':float(hi),'bootstrap_SE':float(delta.std(ddof=1))}
    result={'replicates':1000,'proxy_clusters':61,'primary_contrast':'hybrid1280_minus_sahi640 / COCO AP cap100 area all',
            'differences_pp':contrasts,'interval':'paired percentile using retained seed/groups; 22B/23 fixed comparator outputs reused',
            'scope':'exploratory previously examined test-dev; fixed trained weights; no training-seed uncertainty; unverified proxy-group independence',
            'secondary_multiplicity_corrected':False,'toolkit_confidence_intervals_computed':False}
    write(out/'intervals.json',result)
    with (out/'bootstrap_draws.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['draw','method']+u.KEYS)
        for i in range(1000):
            for j,m in enumerate(['yolo1280','hybrid1280','sahi640']):w.writerow([i,m,*all_samples[i,j].tolist()])
    return result

def execute(root,dataset,out):
    import numpy as np
    for name,h in read(HERE/'package_sha256.json').items():need(sha(HERE/name)==h,'Package file differs: '+name)
    import bootstrap_vendor
    coco_env=bootstrap_vendor.ensure()
    expected22=read(HERE/'references/22_point_results.json')
    need(coco_env==expected22['config']['environment'],'COCO/NumPy environment differs from completed 22B')
    need(read(root/'experiments/supplementary_coco_v2_seed42/point_results.json')==expected22,'Completed 22B comparator record differs')
    expected23=read(HERE/'references/23_result.json')
    actual23=read(root/'experiments/coco_uncertainty_recall_v1_seed42/result.json')
    need(actual23['status']=='COCO_UNCERTAINTY_RECALL_COMPLETE' and actual23['signature']==expected23['signature'],'Completed 23 paired comparator bootstrap differs')
    need(read(root/'experiments/coco_uncertainty_recall_v1_seed42/point.json')==read(HERE/'references/23_point.json'),'23 comparator points differ')
    import test_sahi24
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_sahi24))
    need(tests.wasSuccessful(),'Regression/reference tests failed')
    records=sorted(read(HERE/'references/dataset_manifest.json')['records'],key=lambda x:x['image'])
    need(len(records)==1610 and len({x['stem'] for x in records})==1610,'Frozen test-dev coverage differs')
    manifest=read(HERE/'references/dataset_manifest.json')
    if dataset is None:dataset=root/'experiments/testdev_frozen_v1_seed42'/('dataset_'+manifest['official_zip_sha256'][:16])
    dataset=dataset.resolve();inspect_images(dataset,records)
    weight=root/'release_stage/checkpoints/yolo_epoch_0080.pt'
    old20=root/'experiments/testdev1280_extension_v1_seed42/predictions.sqlite3'
    need(sha(old20)==expected22['config']['source_database_sha256']['20'],'Source20 comparator database differs')
    source_stats={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in [weight,old20]}
    env=prepare_env(root,weight)
    config={'scientific_protocol_sha256':sha(HERE/'protocol.json'),'package_sha256':sha(HERE/'package_sha256.json'),
            'checkpoint_sha256':WEIGHT_SHA,'environment':env,'coco_environment':coco_env,'dataset_path':str(dataset),
            'reference22_fingerprint':fp(expected22),'reference23_signature':expected23['signature']}
    signature=fp(config);write(out/'runtime_bindings.json',config)
    from p2_sahi_exact import sahi_factory,sahi_predict
    model=sahi_factory(weight)
    smoke(root,model,out)
    try:
        with contextlib.closing(state(out/'predictions.sqlite3',signature)) as db:
            freeze=infer(db,records,dataset,signature,out,lambda:lambda image:sahi_predict(model,image,640))
    finally:
        del model
        import torch,gc
        gc.collect();torch.cuda.empty_cache()
    # All new predictions have been committed before any new annotation row is opened.
    print('All 1610 SAHI outputs frozen. Starting CPU metrics.',flush=True)
    with contextlib.closing(state(out/'predictions.sqlite3',signature)) as db:
        toolkit,coco,loader=measure(db,records,dataset,signature,root,out)
        reference_toolkit=read(HERE/'references/stage20_metrics.json')['methods']
        metric={'images':1610,'normal_GT':74707,'sahi640_coco':coco,'sahi640_toolkit':toolkit,
                'comparators_coco':{m:expected22['metrics']['testdev'][m] for m in ['yolo1280','hybrid1280']},
                'comparators_toolkit':{m:reference_toolkit[m] for m in ['yolo1280','hybrid1280']},
                'new_inference_method':'sahi640','input_candidate_truncation_before_metric':False}
        write(out/'metrics.json',metric)
        with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['metric','method','maxDets','AP','AP50','AP75','AP_small','AP_medium','AP_large'])
            for m,caps in {**metric['comparators_coco'],'sahi640':coco}.items():
                for cap,vals in caps.items():w.writerow(['COCO-style',m,cap,*[vals[k] for k in ['AP','AP50','AP75','AP_small','AP_medium','AP_large']]])
            for m,vals in {**metric['comparators_toolkit'],'sahi640':toolkit}.items():w.writerow(['toolkit',m,'global500',vals['AP'],vals['AP50'],vals['AP75'],'','',''])
        ci=uncertainty(db,records,coco,loader,out)
    for p,before in source_stats.items():need((p.stat().st_size,p.stat().st_mtime_ns)==before,'Original checkpoint/database changed during run')
    result={'status':'SAHI640_TESTDEV_COMPLETE','images':1610,'tests_passed':tests.testsRun,'signature':signature,
            'training':False,'parameter_search':False,'new_inference':'SAHI640 only, frozen P2 settings',
            'original_sources_unchanged':True,'prediction_freeze_sha256':fp(freeze),'runtime_bindings':config,
            'sahi640_COCO_AP100':coco['100']['AP'],'sahi640_toolkit_AP':toolkit['AP'],
            'primary_COCO_difference':ci['differences_pp']['hybrid1280_minus_sahi640']['AP'],
            'runtime_note':'progress estimates only; no FPS or latency benchmark'}
    write(out/'result.json',result);return result

def bundle(out,result):
    path=HERE/'24_sahi640_testdev_rapor.zip';temp=path.with_suffix('.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('result.json',json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
        z.write(HERE/'protocol.json','protocol.json')
        for name in ['runtime_bindings.json','val_smoke.json','prediction_index.json','metrics.json','metrics.csv','intervals.json','bootstrap_draws.csv','progress.json']:
            if out is not None and (out/name).exists():z.write(out/name,name)
    os.replace(temp,path);return path

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path(r'D:\HyCADO_Temiz_Baslangic'));p.add_argument('--dataset',type=Path)
    args=p.parse_args();out=None
    try:
        root=args.project.resolve();need(root.is_dir(),'Project directory missing')
        out=root/'experiments'/RUN;out.mkdir(parents=True,exist_ok=True)
        result=execute(root,args.dataset,out)
    except KeyboardInterrupt:
        result={'status':'INTERRUPTED','training':False,'message':'Committed outputs preserved; rerun same CMD'}
    except Exception as e:
        traceback.print_exc();result={'status':'ERROR','error':str(e),'traceback':traceback.format_exc(),'training':False}
    path=bundle(out,result);print('SONUC:',result['status']);print('GONDERILECEK RAPOR:',path)
    return 0 if result['status']=='SAHI640_TESTDEV_COMPLETE' else 1

if __name__=='__main__':sys.exit(main())
