"""Post-hoc fixed-grid control on the same crop-trained EMA and fixed fusion.

Grid size uses TRAIN-development geometry only. Two fresh paired runtime
passes precede all VAL annotation parsing. This is not an independent test,
an optimized SAHI baseline, or proof isolating selection from crop scale.
"""
from __future__ import annotations
import contextlib,csv,math,os,shutil,sys,tempfile,time,traceback,zipfile
from pathlib import Path
import numpy as np
from .common import ROOT,sha256,write_json,utc_now,system_info
from .stage1 import fingerprint,training_lock
from .crop_loader import safe_path,read_json
from . import joint_runtime as j,score_ablation as scoring
from .dev_metrics import raw_annotation,ignore_integral,prepare_groundtruth,outside_ignore,xyxy_to_det
r=j.r
OUT='fixed_grid_comparison_v1_seed42'
GRID='fixed_grid_a025'
HYBRID='hybrid_a025'
PHASES=[(0,HYBRID),(0,GRID),(1,GRID),(1,HYBRID)]
SIDES=list(range(256,1025,32))
FILES20={'protocol.json':'8ce08b5fdd1777898211d92832763f8ff2c0e35a74bf61f0bd831277b83a6967',
 'result.json':'5b225219a89554c5ce6317865ce75a8c5b19d1bbc673a15506a4c72065e80eee',
 'summary.json':'9655e305e5fac403cd9da53a7d7349e1d6c1a87a8548e7e77e16297568f6b9fd'}


def starts(length,side):
    if type(length) is not int or type(side) is not int or min(length,side)<=0:raise ValueError('Pozitif tamsayi boyut gerekli')
    if length<=side:return [0]
    step=max(1,side*4//5)  # fixed 20% nominal overlap; final edge can overlap more
    points=list(range(0,length-side+1,step))
    if points[-1]!=length-side:points.append(length-side)
    return points


def grid_rects(width,height,side):
    return [[x,y,min(x+side,width),min(y+side,height)] for y in starts(height,side) for x in starts(width,side)]


def calibrate(records,*,expected=(647,10682)):
    """Receives only development image names/dimensions and observed crop counts."""
    records=sorted(records,key=lambda x:x['source_image'])
    if (len(records)!=expected[0] or len({x['source_image'] for x in records})!=len(records) or
        any(not x['source_image'].startswith('images/train/') or type(x['crop_count']) is not int or x['crop_count']<0 for x in records) or
        sum(x['crop_count'] for x in records)!=expected[1]):raise ValueError('TRAIN gelistirme kapsami/sayisi farkli')
    target=expected[1]/expected[0];candidates=[]
    for side in SIDES:
        counts=[len(grid_rects(x['width'],x['height'],side)) for x in records]
        candidates.append({'side_px':side,'total_tiles':sum(counts),'mean_tiles':sum(counts)/len(counts),
            'median_tiles':float(np.median(counts)),'p95_tiles':float(np.percentile(counts,95)),
            'relative_count_error':abs(sum(counts)-expected[1])/expected[1]})
    chosen=min(candidates,key=lambda x:(x['relative_count_error'],-x['side_px']))
    return {'source':'647 held-out official TRAIN development image dimensions and all 10682 unfiltered HyCADO crop counts',
        'development_records_sha256':fingerprint(records),'development_images':len(records),'hybrid_development_crops':expected[1],
        'target_mean_crops':target,'candidate_sides_px':SIDES,'overlap_rule':'step=floor(0.8*side); final edge aligned; small image dimension clipped',
        'selection':'minimize absolute relative total crop-count error on development; ties choose larger side',
        'candidate_statistics':candidates,'selected_side_px':chosen['side_px'],'selected_relative_count_error':chosen['relative_count_error'],
        'development_AP_used':False,'VAL_geometry_or_metrics_used_for_selection':False,'GT_rows_parsed':False,
        'scope':'approximate mean forward-call budget matching, not per-image count, crop-scale, area or wall-time matching'}


def development_geometry(root):
    source,capture,old,index=scoring.validate_capture(root)
    mapped={x['source_image']:x for x in index['records']};records=[];immutable={}
    for folder,names in [(source,['protocol.json','prediction_index.json','result.json']),
                         (capture,['protocol.json','capture_index.json','capture_summary.json','result.json'])]:
        immutable.update({str(folder/n):sha256(folder/n) for n in names})
    for item in old['records']:
        entry=mapped[item['source_image']];path=safe_path(capture,entry['path'])
        if sha256(path)!=entry['sha256']:raise ValueError('11C gelistirme metadata degismis')
        doc=read_json(path)
        if (doc['content_sha256']!=fingerprint({k:v for k,v in doc.items() if k!='content_sha256'}) or
            doc['protocol_sha256']!=scoring.CAPTURE_SIGNATURE or doc['source_image']!=item['source_image'] or
            doc['old11_prediction_sha256']!=item['sha256']):raise ValueError('11C kaynak baglantisi farkli')
        records.append({'source_image':item['source_image'],'width':item['width'],'height':item['height'],'crop_count':doc['crop_count']})
        immutable[str(path)]=entry['sha256']
    return calibrate(records),immutable


def source(root):
    c12,base,folder,w1,w2,full,official,immutable=j.source(root)
    old20=root/'experiments'/j.OUT
    for name,digest in FILES20.items():
        if sha256(old20/name)!=digest:raise ValueError('Tamamlanan 20 kaydi farkli: '+name)
        immutable[str(old20/name)]=digest
    c20=read_json(old20/'protocol.json')
    if c20['code_sha256']!=sha256(j.__file__) or c20['records']!=c12['records']:raise ValueError('20 kod/kaynak kimligi farkli')
    phases=[]
    for p,v in j.PHASES:
        path=old20/f'pass_{p+1}_{v}.json';phases.append(j.validate_phase(read_json(path),c20,p,v));immutable[str(path)]=sha256(path)
    if j.summarize(phases,read_json(folder/'metrics.json'))!=read_json(old20/'summary.json'):raise ValueError('20 ozeti farkli')
    return c12,base,folder,w1,w2,official,immutable


def fuse(raw,rects,baseline,width,height,nms):
    q=np.asarray(raw)
    if q.dtype!=np.float32 or q.shape!=(len(rects),300,14) or not np.isfinite(q).all():raise ValueError('21 ham query bicimi farkli')
    candidates=list(baseline)
    for query,rect in zip(q,rects):
        rows=r.old.map_queries(query,{'source_crop_xyxy':rect})
        for row in rows:row[4]*=.25
        candidates.extend(rows)
    return nms(candidates,width,height)


def run_grid(record,path,predict1,predict2,side,sync,*,before_crops=lambda:None,clock=time.perf_counter):
    from PIL import Image
    sync();start=clock()
    width,height,baseline,_=predict1(path);sync()
    if (width,height)!=(record['width'],record['height']):raise ValueError('21 goruntu boyutu farkli')
    rects=grid_rects(width,height,side);before_crops();raw=[];candidates=list(baseline)
    with Image.open(path) as image:
        if image.size!=(width,height):raise ValueError('21 RGB goruntu boyutu farkli')
        rgb=image.convert('RGB')
    for rect in rects:
        query=predict2(np.array(rgb.crop(tuple(rect))))
        if query.shape!=(300,14) or query.dtype!=np.float32:raise ValueError('21 ham query bicimi farkli')
        raw.append(query)
        rows=r.old.map_queries(query,{'source_crop_xyxy':rect})
        for row in rows:row[4]*=.25
        candidates.extend(rows)
    sync();result=j.optimized.n.fast_nms(candidates,width,height);sync()
    elapsed=(clock()-start)*1000
    # Packing/compression and replay checks occur after the measured interval.
    return {'total_ms':elapsed,'crop_count':len(rects)},baseline,result,rects,np.stack(raw)


def save_prediction(folder,record,cfg,baseline,result,rects,raw):
    if rects!=grid_rects(record['width'],record['height'],cfg['calibration']['selected_side_px']):raise ValueError('21 grid geometrisi farkli')
    # Independent original NumPy NMS checks the accelerated implementation on new candidates.
    if fuse(raw,rects,baseline,record['width'],record['height'],r.old.classwise_nms)!=result:raise ValueError('21 NMS/decoder esligi farkli')
    folder.parent.mkdir(parents=True,exist_ok=True)
    staging=folder.parent.parent/'staging';staging.mkdir(exist_ok=True)
    tmp=Path(tempfile.mkdtemp(dir=staging,prefix=folder.name+'_building_'))
    try:
        np.savez_compressed(tmp/'queries.npz',raw_queries=raw,crop_rects=np.asarray(rects,dtype=np.int32))
        r.val.signed_json(tmp/'prediction.json',{'protocol_sha256':fingerprint(cfg),'source_image':record['image'],
            'image_sha256':record['image_sha256'],'width':record['width'],'height':record['height'],
            'baseline':baseline,GRID:result,'crop_rects':rects,'crop_count':len(rects),'independent_NMS_replay_exact':True,
            'raw_archive_sha256':sha256(tmp/'queries.npz')})
        if folder.exists():raise FileExistsError('21 tamamlanmis goruntunun uzerine yazilmadi')
        tmp.rename(folder)
    finally:
        if tmp.exists():shutil.rmtree(tmp)


def checked_prediction(folder,record,cfg):
    doc=r.val.checked_json(folder/'prediction.json',fingerprint(cfg),record)
    rects=grid_rects(record['width'],record['height'],cfg['calibration']['selected_side_px'])
    if (doc['crop_rects']!=rects or doc['crop_count']!=len(rects) or doc.get('independent_NMS_replay_exact') is not True or
        doc['raw_archive_sha256']!=sha256(folder/'queries.npz')):raise ValueError('21 grid/ham kayit farkli')
    with np.load(folder/'queries.npz',allow_pickle=False) as data:
        if data['crop_rects'].tolist()!=rects:raise ValueError('21 arsiv crop geometrisi farkli')
        rows=fuse(data['raw_queries'],rects,doc['baseline'],record['width'],record['height'],j.optimized.n.fast_nms)
    if rows!=doc[GRID]:raise ValueError('21 decoder arsiv esligi farkli')
    return doc


def validate_phase(doc,cfg,p,variant):
    if ((p,variant) not in PHASES or doc.get('status')!='COMPLETED' or doc.get('pass_id')!=p or doc.get('variant')!=variant or
        doc.get('protocol_sha256')!=fingerprint(cfg) or doc.get('content_sha256')!=fingerprint({k:v for k,v in doc.items() if k!='content_sha256'})):
        raise ValueError('21 faz kimligi/hash farkli')
    order=r.pass_order(cfg['records'],p)
    if [x['source_image'] for x in doc['rows']]!=[x['image'] for x in order]:raise ValueError('21 faz kapsami/sirasi farkli')
    for row,rec in zip(doc['rows'],order):
        count=cfg['hybrid_crop_counts'][rec['image']] if variant==HYBRID else len(grid_rects(rec['width'],rec['height'],cfg['calibration']['selected_side_px']))
        if row.get('exact_reference_output') is not True or row['crop_count']!=count:raise ValueError('21 cikti/crop esligi basarisiz')
        if any(not math.isfinite(row[k]) or row[k]<=0 for k in ['total_ms','peak_allocated_mib','peak_reserved_mib']):raise ValueError('21 gecersiz sure/bellek')
        if row['peak_reserved_mib']+.01<row['peak_allocated_mib']:raise ValueError('21 bellek sayaclari tutarsiz')
    return doc


def worker(job_path):
    job=read_json(job_path);output=Path(job['output']);cfg=read_json(output/'protocol.json');p=job['pass_id'];variant=job['variant']
    try:
        if (p,variant) not in PHASES or job['protocol_sha256']!=fingerprint(cfg) or cfg['code_sha256']!=sha256(__file__):raise ValueError('21 is/kod/protokol farkli')
        c12,base,_,w1,w2,official,immutable=source(ROOT)
        import torch
        env=system_info(include_torch=True)
        if not torch.cuda.is_available() or env['gpu']!=cfg['gpu'] or env['nvidia_smi'].split(',')[-1].strip()!=cfg['driver']:raise ValueError('21 CUDA/GPU/surucu farkli')
        loading=time.perf_counter()
        predict1=r.proposals.make_predictor(r.proposals.load_ema(w1,r.val.P1),{'inference':c12['stage1_inference']},output)
        state1=r.backend_state();model=r.crop_capture.load_model(w2,official,'cuda');state2=r.backend_state();load_seconds=time.perf_counter()-loading
        def fresh(rec):
            r.restore_backend(state1);path=safe_path(ROOT/'data/prepared',rec['image'])
            before=lambda:r.restore_backend(state2);predict2=lambda rgb:r.direct_queries(model,rgb)
            if variant==GRID:return run_grid(rec,path,predict1,predict2,cfg['calibration']['selected_side_px'],torch.cuda.synchronize,before_crops=before)
            return (*j.optimized.run_image(rec,path,predict1,predict2,c12['crop_policy'],True,torch.cuda.synchronize,before_crops=before),None)
        order=r.pass_order(cfg['records'],p)
        for rec in order:
            if sha256(safe_path(ROOT/'data/prepared',rec['image']))!=rec['image_sha256']:raise ValueError('21 kaynak goruntu degismis')
        mapping={x['source_image']:x for x in read_json(base/'prediction_index.json')['records']}
        warm=time.perf_counter()
        for rec in order[:cfg['warmup_images']]:fresh(rec)
        torch.cuda.synchronize();warm_seconds=time.perf_counter()-warm;rows=[];started=time.perf_counter()
        for i,rec in enumerate(order,1):
            expected=r.references(base,rec,c12,mapping);folder=output/'predictions'/Path(rec['image']).stem;reference=None
            if variant==GRID and folder.exists():reference=checked_prediction(folder,rec,cfg)
            if variant==GRID and p>0 and reference is None:raise ValueError('21 ilk grid turu tamamlanmamis')
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            times,baseline,result,rects,raw=fresh(rec)
            if baseline!=expected['baseline']:raise ValueError('21 Stage1, sabit 12 ile birebir farkli')
            if variant==HYBRID:
                if result!=expected[HYBRID] or rects!=expected['crop_rects']:raise ValueError('21 HyCADO, sabit 12 ile birebir farkli')
            elif reference is None:save_prediction(folder,rec,cfg,baseline,result,rects,raw)
            elif result!=reference[GRID] or baseline!=reference['baseline']:raise ValueError('21 tekrar ciktisi ilk grid turuyla farkli')
            rows.append({'source_image':rec['image'],**times,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'exact_reference_output':True})
            if i==1 or i%10==0 or i==len(order):
                print(f'21 tur {p+1}/2 {variant}: {i}/{len(order)}; gecen={(time.perf_counter()-started)/60:.1f} dk',flush=True)
                write_json(output/'progress.json',{'status':'MEASURING','pass_id':p,'variant':variant,'images_done':i})
        for rec in order:
            if sha256(safe_path(ROOT/'data/prepared',rec['image']))!=rec['image_sha256']:raise ValueError('21 olcum sirasinda goruntu degismis')
        for path,digest in immutable.items():
            if sha256(path)!=digest:raise ValueError('21 eski kaynak/agirlik degismis')
        body={'status':'COMPLETED','protocol_sha256':fingerprint(cfg),'pass_id':p,'variant':variant,'rows':rows,
            'created_utc':utc_now(),'environment':env,'model_load_seconds_excluded':load_seconds,'warmup_seconds_excluded':warm_seconds}
        body['content_sha256']=fingerprint(body);validate_phase(body,cfg,p,variant);write_json(output/f'pass_{p+1}_{variant}.json',body);return 0
    except (Exception,KeyboardInterrupt) as exc:
        traceback.print_exc();write_json(output/'phase_error.json',{'pass_id':p,'variant':variant,'error':str(exc)});return 1


def measure(records,prepared,base,output,cfg,folder19):
    # Validate every prediction and the complete set BEFORE parsing any annotation rows.
    if sorted(x.name for x in (output/'predictions').iterdir() if x.is_dir())!=sorted(Path(x['image']).stem for x in records):raise ValueError('21 548 grid tahmini tamamlanmamis')
    signature=fingerprint(cfg);index=[];parts=[];oldmap={x['source_image']:x for x in read_json(base/'prediction_index.json')['records']};immutable={}
    for rec in records:
        folder=output/'predictions'/Path(rec['image']).stem;checked_prediction(folder,rec,cfg)
        index.append({'source_image':rec['image'],'path':(folder/'prediction.json').relative_to(output).as_posix(),
            'sha256':sha256(folder/'prediction.json'),'raw_archive_sha256':sha256(folder/'queries.npz')})
    write_json(output/'prediction_index.json',{'protocol_sha256':signature,'records':index})
    for i,(rec,item) in enumerate(zip(records,index),1):
        path=safe_path(output,item['path']);new=r.val.checked_json(path,signature,rec)
        entry=oldmap[rec['image']];priorpath=safe_path(base,entry['path']);prior=r.references(base,rec,cfg,oldmap)
        annotation=prepared/'visdrone_annotations/val'/(Path(rec['image']).stem+'.txt')
        if sha256(annotation)!=rec['annotation_sha256'] or sha256(path)!=item['sha256']:raise ValueError('21 metrik kaynagi degismis')
        if prior['baseline']!=new['baseline']:raise ValueError('21 metrik Stage1 kontrolu farkli')
        immutable.update({str(path):item['sha256'],str(priorpath):entry['sha256'],str(annotation):rec['annotation_sha256']})
        raw=raw_annotation(annotation.read_text(encoding='utf-8-sig'));integral=ignore_integral(raw,rec['width'],rec['height']);gt=prepare_groundtruth(raw,integral)
        data={'gt_denominators':np.array([np.count_nonzero(gt[:,5]==c) for c in range(1,11)],dtype=np.int64)};data['presence']=(data['gt_denominators']>0).astype(np.int64)
        for name,rows in [('baseline',prior['baseline']),(HYBRID,prior[HYBRID]),(GRID,new[GRID])]:
            det=outside_ignore(xyxy_to_det(rows),integral)[:500]
            data[name+'_scores']=det[:,4].copy();data[name+'_classes']=det[:,5].astype(np.int8)
            data[name+'_flags']=np.column_stack([scoring.fast_flags(det,gt,float(t)) for t in scoring.THRESHOLDS])
        parts.append(data)
        if i==1 or i%25==0 or i==len(records):print(f'21 ayni VAL metrikleri: {i}/{len(records)}',flush=True)
    variants={v:scoring.aggregate(parts,v) for v in ['baseline',HYBRID,GRID]};frozen=read_json(folder19/'metrics.json')
    for v in ['baseline',HYBRID]:
        scoring.compare_metrics(variants[v],frozen['variants'][v])
        if variants[v]['counts_at_fixed_max500']!=frozen['variants'][v]['counts_at_fixed_max500']:raise ValueError('21 eski TP/FP kontrolu farkli')
    for path,digest in immutable.items():
        if sha256(path)!=digest:raise ValueError('21 metrik sirasinda kaynak degismis')
    report={'images':len(records),'variants':variants,'frozen_baseline_and_hybrid_metrics_equal':True,
        'delta_hybrid_minus_grid_pp':{k:variants[HYBRID][k]-variants[GRID][k] for k in ['AP','AP50','AP75','AR500']},
        'official_binary_parity_verified':False,'metric_scope':'unchanged provisional reference-aligned Python metric, percent units',
        'independent_test':False,'multiple_training_seeds':False,'source_full_rtdetr_metrics_unchanged':frozen['variants']['rtdetr_full960']}
    write_json(output/'metrics.json',report)
    for filename,rows in [('metrics.csv',[{'variant':v,**{k:variants[v][k] for k in ['AP','AP50','AP75','AR500']}} for v in variants]),
        ('per_class.csv',[{'variant':v,**row} for v in variants for row in variants[v]['per_class']])]:
        with (output/filename).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return report


def summarize(phases,metrics,cfg):
    if len(phases)!=len(PHASES) or {(x['pass_id'],x['variant']) for x in phases}!=set(PHASES):raise ValueError('21 dort tam faz gerekli')
    for doc in phases:validate_phase(doc,cfg,doc['pass_id'],doc['variant'])
    variants={}
    for variant in [HYBRID,GRID]:
        rows=[row for doc in phases if doc['variant']==variant for row in doc['rows']];a=np.array([x['total_ms'] for x in rows])
        variants[variant]={'AP':metrics['variants'][variant]['AP'],'AP50':metrics['variants'][variant]['AP50'],
            'mean_ms':float(a.mean()),'median_ms':float(np.median(a)),'p95_ms':float(np.percentile(a,95)),
            'throughput_images_per_second':float(1000/a.mean()),'measured_image_calls':len(rows),
            'mean_crops':sum(x['crop_count'] for x in rows)/len(rows),'peak_allocated_mib':max(x['peak_allocated_mib'] for x in rows),
            'peak_reserved_mib':max(x['peak_reserved_mib'] for x in rows)}
    paired=[]
    for p in range(2):
        means={v:float(np.mean([x['total_ms'] for doc in phases if doc['pass_id']==p and doc['variant']==v for x in doc['rows']])) for v in [HYBRID,GRID]}
        paired.append({'pass_id':p,**means,'hybrid_to_grid_latency_ratio':means[HYBRID]/means[GRID]})
    return {'variants':variants,'paired_passes':paired,'selected_side_px':cfg['calibration']['selected_side_px'],
        'timing_scope':'two paired fresh-process passes, both models resident; batch1 path-to-CPU final boxes; model load/warmup, replay and archive I/O excluded',
        'memory_scope':'Torch allocated/reserved only, resident weights and warm allocator included, not total device use',
        'control_scope':'same crop-trained EMA80, YOLO, input preprocessing, alpha .25, NMS .7 and max1000; different crop geometry',
        'limitations':['development mean crop count approximately matched, not per-image count, area, scale, or measured time',
            'crop model was trained on HyCADO crops; this is a geometry control, not fully tuned tiling or SAHI benchmark',
            'post-hoc extension on previously examined VAL; one training seed; two runtime repetitions are not independent training runs'],
        'source20_timing_reused':False,'training_started':False}


def bundle(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=output,suffix='.zip.tmp');os.close(fd)
    target=output/'21_sabit_dilimleme_rapor.zip'
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(output.glob('*.json'))+sorted(output.glob('*.csv'))+sorted((output/'predictions').glob('*/prediction.json')):z.write(path,path.relative_to(output).as_posix())
            if (output/'console.log').exists():
                with (output/'console.log').open('rb') as f:f.seek(max(0,(output/'console.log').stat().st_size-90000));z.writestr('console_tail.log',f.read())
        os.replace(tmp,target)
    finally:Path(tmp).unlink(missing_ok=True)
    return target


def backup(root,output):
    """Small new experiment only, including its raw queries; no model/data recopy."""
    target=root.parent/'HyCADO_Yedekleri/21_Sabit_Dilimleme/HyCADO_Temiz_Baslangic'
    paths=[root/'hycado_start/fixed_grid_comparison.py']+[root/n for n in ['21_SABIT_DILIMLEME_KARSILASTIR.cmd','21_RAPOR_AL.cmd','21_ONCE_OKUYUN.md','SHA256SUMS_21.txt'] if (root/n).exists()]
    paths += [p for p in output.rglob('*') if p.is_file() and (p.parent==output or p.parent.parent.name=='predictions') and p.suffix in ['.json','.csv','.zip','.npz']]
    for path in paths:
        dest=target/path.relative_to(root);digest=sha256(path);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and sha256(dest)==digest:continue
        fd,tmp=tempfile.mkstemp(dir=dest.parent,suffix='.tmp');os.close(fd)
        try:
            shutil.copy2(path,tmp)
            if sha256(tmp)!=digest:raise ValueError('21 yedek dogrulamasi basarisiz')
            os.replace(tmp,dest)
        finally:Path(tmp).unlink(missing_ok=True)
    write_json(output/'backup_receipt.json',{'verified_utc':utc_now(),'files':len(paths),'destination':str(target),
        'scope':'21 code/config/reports and raw predictions only; not all project/models; same D volume is not independent media'})


def run():
    from .__main__ import _Tee
    output=ROOT/'experiments'/OUT;output.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        for lock in [ROOT/'.backup.lock']+[ROOT/'experiments'/folder/name for folder,name in [(OUT,'.benchmark.lock'),
            (j.OUT,'.benchmark.lock'),(j.e.OUT,'.evaluation.lock'),(j.e.training.OUT,'.training.lock'),
            ('stage1_v1_seed42','.training.lock'),('rtdetr_crop_v1_seed42','.training.lock'),(r.val.OUT,'.evaluation.lock'),
            ('development_eval_v1_seed42','.evaluation.lock'),('raw_crop_capture_v1_seed42','.capture.lock'),
            (r.OUT,'.benchmark.lock'),(j.optimized.OUT,'.benchmark.lock')]]:stack.enter_context(training_lock(lock))
        with (output/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    write_json(output/'result.json',{'status':'RUNNING'});print('21 SABIT DILIMLEME KONTROLU:',utc_now(),flush=True)
                    c12,base,folder19,*_,immutable=source(ROOT);calibration,devhashes=development_geometry(ROOT);immutable.update(devhashes)
                    env=system_info(include_torch=True)
                    if not env.get('cuda_available'):raise RuntimeError('21 icin CUDA gerekli')
                    cfg={'id':'hycado_fixed_grid_control_v1','date':'2026-11-02','calibration':calibration,'records':c12['records'],
                        'hybrid_crop_counts':{x['source_image']:x['crops'] for x in read_json(base/'prediction_index.json')['records']},
                        'phase_order':PHASES,'warmup_images':5,'weights':{'YOLO':c12['stage1_weights_sha256'],'crop':c12['crop_weights_sha256']},
                        'runtime_versions':c12['runtime_versions'],'gpu':env['gpu'],'driver':env['nvidia_smi'].split(',')[-1].strip(),
                        'fusion':c12['fusion'],'source20_metadata_sha256':FILES20,'development_metadata_sha256':devhashes,
                        'source12_protocol':r.P12,'crop_inference':c12['crop_inference'],'code_sha256':sha256(__file__),
                        'freeze_before_new_VAL_inference':True,'VAL_used_to_select_grid':False,'post_hoc_extension':True,'training_started':False}
                    # JSON normalizes tuples; compare stable fingerprints rather than Python tuple/list equality.
                    if (output/'protocol.json').exists() and fingerprint(read_json(output/'protocol.json'))!=fingerprint(cfg):raise ValueError('21 protokol/kod/ortam farkli; eski kayitlar karistirilmadi')
                    write_json(output/'protocol.json',cfg);write_json(output/'calibration.json',calibration);write_json(output/'environment.json',env)
                    # Only uncommitted experiment-owned staging is disposable after a hard restart.
                    if (output/'staging').exists():shutil.rmtree(output/'staging')
                    for name in ['metrics.json','metrics.csv','per_class.csv','summary.json','summary.csv']:(output/name).unlink(missing_ok=True)
                    print('GELISTIRMEDE SABITLENDI: tile=',calibration['selected_side_px'],'; ortalama crop farki=',round(100*calibration['selected_relative_count_error'],2),'%',flush=True)
                    backup(ROOT,output);phases=[];reused=0
                    for p,v in PHASES:
                        path=output/f'pass_{p+1}_{v}.json'
                        if path.exists():reused+=1;print('TAMAMLANAN FAZ KORUNUYOR:',path.name,flush=True)
                        else:
                            (output/'phase_error.json').unlink(missing_ok=True)
                            write_json(output/'worker_job.json',{'output':str(output),'pass_id':p,'variant':v,'protocol_sha256':fingerprint(cfg)})
                            code=r.child_process([sys.executable,'-X','faulthandler','-u','-m','hycado_start.fixed_grid_comparison','--worker',str(output/'worker_job.json')],ROOT)
                            if code!=0:raise RuntimeError('21 alt islem durdu: '+str(code))
                        phases.append(validate_phase(read_json(path),cfg,p,v));backup(ROOT,output)
                    metrics=measure(cfg['records'],ROOT/'data/prepared',base,output,cfg,folder19);summary=summarize(phases,metrics,cfg);write_json(output/'summary.json',summary)
                    with (output/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
                        w=csv.DictWriter(f,fieldnames=['variant']+list(summary['variants'][HYBRID]));w.writeheader()
                        for v,vals in summary['variants'].items():w.writerow({'variant':v,**vals})
                    for path,digest in immutable.items():
                        if sha256(path)!=digest:raise ValueError('21 onceki kaynak/agirlik degismis')
                    backup(ROOT,output)
                    result={'status':'COMPLETED','images':len(cfg['records']),'phases':4,'measured_image_calls':4*len(cfg['records']),
                        'original_weights_and_results_unchanged':True,'hybrid_exact_frozen12':True,'grid_repeats_exact':True,
                        'grid_independent_NMS_replay_exact':True,'frozen_baseline_and_hybrid_metrics_equal':True,
                        'backup_verified':True,'training_started':False,'VAL_used_to_select_grid':False,'reused_completed_phases':reused,'official_binary_parity_verified':False}
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();result={'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR','error':str(exc),
                        'resume':'rerun21; complete phases reused; partial phase retimed from start; completed first-grid image predictions retained'}
                write_json(output/'result.json',result);write_json(output/'progress.json',result);print('SONUC:',result['status'],flush=True)
        report=bundle(output)
        try:backup(ROOT,output)
        except Exception as exc:
            print('21 SON RAPOR YEDEGI ALINAMADI:',str(exc),flush=True)
            if result['status']=='COMPLETED':result.update(status='ERROR',backup_verified=False,error='Final report backup failed: '+str(exc));write_json(output/'result.json',result);bundle(output)
        print('GONDERILECEK RAPOR:',report,flush=True);return result['status']=='COMPLETED'


def main():
    if '--worker' in sys.argv:return worker(Path(sys.argv[sys.argv.index('--worker')+1]))
    if '--report-only' in sys.argv:print(bundle(ROOT/'experiments'/OUT),flush=True);return 0
    return 0 if run() else 1


if __name__=='__main__':raise SystemExit(main())
