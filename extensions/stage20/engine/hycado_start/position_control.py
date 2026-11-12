"""Conditional crop-location control, fixed counts and ordered dimensions.

Three predeclared location seeds, no training, tuning or runtime claim.
The same crop-trained EMA and cached YOLO detections are used throughout.
All geometry is frozen before new inference; GT parsing follows every seed.
"""
from __future__ import annotations
import contextlib,csv,os,shutil,sys,tempfile,time,traceback,zipfile
from pathlib import Path
import numpy as np
from .common import ROOT,sha256,write_json,utc_now,system_info
from .stage1 import fingerprint,training_lock
from .crop_loader import safe_path,read_json
from . import fixed_grid_comparison as g
from .dev_metrics import raw_annotation,ignore_integral,prepare_groundtruth,outside_ignore,xyxy_to_det
r=g.r
OUT='crop_position_control_v1_seed42'
SEEDS=[0,1,2]
FILES21={'protocol.json':'c71a9d8f4329dfac7efdd51849dc84e6023845b9c2985a64cc9ccc286bb6c96c',
 'result.json':'446de60d38bfa1364e4aa30352b04e4a197d60a20f9e98705b8c6efba14fae2a',
 'summary.json':'2ab432160f2eec0d74c8959c4eb35d2c3aff94d6a46791f3676ddcc36a07fa56',
 'metrics.json':'3722ea76a96ed4cc5aa0ab0e5c1ea08dd9df71d50c0d32dd9abfe24285d81328',
 'prediction_index.json':'b85ef1a717ad2316d1fe0a6a8e8a4c9a038c76dabfd18d20667184fea01c1751'}


def variant(seed):return f'relocated_s{seed}_a025'


def uniform_coordinate(maximum,identity):
    """Unbiased integer on [0, maximum] from SHA256 with rejection sampling."""
    if type(maximum) is not int or maximum<0:raise ValueError('22 koordinat araligi gecersiz')
    n=maximum+1;bound=(1<<256)//n*n;counter=0
    while True:
        value=int(fingerprint({'domain':'hycado22_crop_position_v1','identity':identity,'counter':counter}),16)
        if value<bound:return value%n
        counter+=1


def relocate(rects,width,height,image,seed):
    if seed not in SEEDS or type(seed) is not int or type(width) is not int or type(height) is not int or min(width,height)<=0:raise ValueError('22 seed/boyut gecersiz')
    result=[]
    for i,rect in enumerate(rects):
        if (len(rect)!=4 or any(type(x) is not int for x in rect) or not 0<=rect[0]<rect[2]<=width or not 0<=rect[1]<rect[3]<=height):raise ValueError('22 kaynak crop geometrisi gecersiz')
        w,h=rect[2]-rect[0],rect[3]-rect[1]
        key={'image':image,'position_seed':seed,'crop_index':i}
        x=uniform_coordinate(width-w,{**key,'axis':'x'});y=uniform_coordinate(height-h,{**key,'axis':'y'})
        result.append([x,y,x+w,y+h])
    return result


def make_plan(inputs,*,expected=(548,10057)):
    inputs=sorted(inputs,key=lambda x:x['source_image']);plans=[]
    if (len(inputs)!=expected[0] or len({x['source_image'] for x in inputs})!=len(inputs) or
        any(not x['source_image'].startswith('images/val/') for x in inputs) or sum(len(x['crop_rects']) for x in inputs)!=expected[1]):raise ValueError('22 kaynak goruntu/crop kapsami farkli')
    for item in inputs:
        row={k:item[k] for k in ['source_image','image_sha256','width','height','reference12_sha256']}
        row['original_rects']=item['crop_rects']
        for seed in SEEDS:row[variant(seed)]=relocate(item['crop_rects'],item['width'],item['height'],item['source_image'],seed)
        plans.append(row)
    return {'algorithm':'SHA256 canonical JSON with rejection, independent uniform integer x/y over all in-bounds placements per crop',
        'location_seeds':SEEDS,'images':len(plans),'crops_per_seed':expected[1],'total_new_crop_forwards':expected[1]*len(SEEDS),
        'source_pixel_area_per_seed':sum((b[2]-b[0])*(b[3]-b[1]) for x in plans for b in x['original_rects']),
        'unchanged_position_crops':{variant(s):sum(a==b for x in plans for a,b in zip(x['original_rects'],x[variant(s)])) for s in SEEDS},
        'rules':'ordered widths/heights and count preserved; no clipping/padding, deduplication, retry-to-avoid-overlap or forced movement; coincidences allowed',
        'annotation_or_score_based_placement':False,'records':plans}


def source(root):
    c12,base,folder19,w1,w2,official,immutable=g.source(root);folder21=root/'experiments'/g.OUT
    for name,digest in FILES21.items():
        if sha256(folder21/name)!=digest:raise ValueError('Tamamlanan 21 kaydi farkli: '+name)
        immutable[str(folder21/name)]=digest
    cfg=read_json(folder21/'protocol.json');result=read_json(folder21/'result.json')
    if (cfg['code_sha256']!=sha256(g.__file__) or cfg['records']!=c12['records'] or result.get('status')!='COMPLETED' or
        result.get('images')!=548 or result.get('backup_verified') is not True):raise ValueError('22 icin dogrulanmis 21 gerekli')
    phases=[]
    for p,v in g.PHASES:
        path=folder21/f'pass_{p+1}_{v}.json';phases.append(g.validate_phase(read_json(path),cfg,p,v));immutable[str(path)]=sha256(path)
    if g.summarize(phases,read_json(folder21/'metrics.json'),cfg)!=read_json(folder21/'summary.json'):raise ValueError('21 ozet esligi farkli')
    return c12,base,folder19,w2,official,immutable


def build_plan(c12,base):
    mapping={x['source_image']:x for x in read_json(base/'prediction_index.json')['records']};inputs=[];immutable={}
    for rec in c12['records']:
        item=mapping[rec['image']];doc=r.references(base,rec,c12,mapping);path=safe_path(base,item['path'])
        if len(doc['crop_rects'])!=item['crops']:raise ValueError('12 crop sayaci farkli')
        inputs.append({'source_image':rec['image'],'image_sha256':rec['image_sha256'],'width':rec['width'],'height':rec['height'],
            'reference12_sha256':item['sha256'],'crop_rects':doc['crop_rects']});immutable[str(path)]=item['sha256']
    return make_plan(inputs),immutable


def save_prediction(folder,rec,cfg,plan,seed,baseline,raw):
    rects=plan[variant(seed)];rows=g.fuse(raw,rects,baseline,rec['width'],rec['height'],g.j.optimized.n.fast_nms)
    if rows!=g.fuse(raw,rects,baseline,rec['width'],rec['height'],r.old.classwise_nms):raise ValueError('22 hizli/eski NMS esligi basarisiz')
    folder.parent.mkdir(parents=True,exist_ok=True);staging=folder.parents[2]/'staging';staging.mkdir(exist_ok=True)
    tmp=Path(tempfile.mkdtemp(dir=staging,prefix=f's{seed}_{folder.name}_'))
    try:
        np.savez_compressed(tmp/'queries.npz',raw_queries=raw,crop_rects=np.asarray(rects,dtype=np.int32).reshape(-1,4))
        r.val.signed_json(tmp/'prediction.json',{'protocol_sha256':fingerprint(cfg),'source_image':rec['image'],
            'image_sha256':rec['image_sha256'],'width':rec['width'],'height':rec['height'],'position_seed':seed,'variant':variant(seed),
            'reference12_sha256':plan['reference12_sha256'],'plan_record_sha256':fingerprint(plan),'crop_count':len(rects),
            'crop_rects':rects,'detections':rows,'independent_NMS_replay_exact':True,'raw_archive_sha256':sha256(tmp/'queries.npz')})
        if folder.exists():raise FileExistsError('22 tamamlanmis tahmin uzerine yazilmadi')
        tmp.rename(folder)
    finally:
        if tmp.exists():shutil.rmtree(tmp)


def checked_prediction(folder,rec,cfg,plan,seed,baseline):
    doc=r.val.checked_json(folder/'prediction.json',fingerprint(cfg),rec);rects=plan[variant(seed)]
    if (doc['position_seed']!=seed or doc['variant']!=variant(seed) or doc['reference12_sha256']!=plan['reference12_sha256'] or
        doc['plan_record_sha256']!=fingerprint(plan) or doc['crop_rects']!=rects or doc['crop_count']!=len(rects) or
        doc.get('independent_NMS_replay_exact') is not True or sha256(folder/'queries.npz')!=doc['raw_archive_sha256']):raise ValueError('22 tahmin/plan/arsiv kimligi farkli')
    with np.load(folder/'queries.npz',allow_pickle=False) as data:
        if data['crop_rects'].tolist()!=rects:raise ValueError('22 ham geometri farkli')
        rows=g.fuse(data['raw_queries'],rects,baseline,rec['width'],rec['height'],g.j.optimized.n.fast_nms)
    if rows!=doc['detections']:raise ValueError('22 ham decoder esligi farkli')
    return doc


def infer_image(model,path,rec,rects):
    from PIL import Image
    with Image.open(path) as image:
        if image.size!=(rec['width'],rec['height']):raise ValueError('22 goruntu boyutu farkli')
        rgb=image.convert('RGB')
    raw=[r.direct_queries(model,np.array(rgb.crop(tuple(rect)))) for rect in rects]
    return np.stack(raw) if raw else np.empty((0,300,14),dtype=np.float32)


def copy_verified(root,paths):
    target=root.parent/'HyCADO_Yedekleri/22_Konum_Kontrolu/HyCADO_Temiz_Baslangic'
    for path in paths:
        digest=sha256(path);dest=target/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and sha256(dest)==digest:continue
        fd,tmp=tempfile.mkstemp(dir=dest.parent,suffix='.tmp');os.close(fd)
        try:
            shutil.copy2(path,tmp)
            if sha256(tmp)!=digest:raise ValueError('22 yedek dogrulamasi basarisiz')
            os.replace(tmp,dest)
        finally:Path(tmp).unlink(missing_ok=True)
    return target


def validate_seed(doc,cfg,plan,seed):
    if (seed not in SEEDS or doc.get('status')!='COMPLETED' or doc.get('position_seed')!=seed or
        doc.get('protocol_sha256')!=fingerprint(cfg) or doc.get('content_sha256')!=fingerprint({k:v for k,v in doc.items() if k!='content_sha256'})):
        raise ValueError('22 seed kimligi/hash farkli')
    records=doc['records'];mapped={x['source_image']:x for x in plan['records']}
    if [x['source_image'] for x in records]!=[x['image'] for x in cfg['records']]:raise ValueError('22 seed goruntu kapsami/sirasi farkli')
    if any(x['crops']!=len(mapped[x['source_image']]['original_rects']) for x in records) or sum(x['crops'] for x in records)!=plan['crops_per_seed']:raise ValueError('22 seed crop kapsami farkli')
    return doc


def worker(job_path):
    job=read_json(job_path);output=Path(job['output']);cfg=read_json(output/'protocol.json');plan=read_json(output/'geometry_plan.json');seed=job['position_seed']
    try:
        if seed not in SEEDS or job['protocol_sha256']!=fingerprint(cfg) or fingerprint(plan)!=cfg['geometry_plan_sha256'] or sha256(__file__)!=cfg['code_sha256']:raise ValueError('22 is/protokol/kod farkli')
        c12,base,_,weights,official,immutable=source(ROOT)
        import torch
        env=system_info(include_torch=True)
        if not torch.cuda.is_available() or env['gpu']!=cfg['gpu'] or env['nvidia_smi'].split(',')[-1].strip()!=cfg['driver']:raise ValueError('22 CUDA/GPU/surucu farkli')
        mapping={x['source_image']:x for x in read_json(base/'prediction_index.json')['records']};plans={x['source_image']:x for x in plan['records']}
        model=None;index=[];new=reused=0;started=time.perf_counter()
        for i,rec in enumerate(cfg['records'],1):
            path=safe_path(ROOT/'data/prepared',rec['image']);prior=r.references(base,rec,c12,mapping);p=plans[rec['image']]
            if sha256(path)!=rec['image_sha256'] or mapping[rec['image']]['sha256']!=p['reference12_sha256'] or p['original_rects']!=prior['crop_rects']:raise ValueError('22 kaynak goruntu/geometri degismis')
            folder=output/'predictions'/f's{seed}'/Path(rec['image']).stem
            if folder.exists():reused+=1
            else:
                if model is None and p[variant(seed)]:model=r.crop_capture.load_model(weights,official,'cuda')
                raw=infer_image(model,path,rec,p[variant(seed)])
                save_prediction(folder,rec,cfg,p,seed,prior['baseline'],raw);new+=1
            doc=checked_prediction(folder,rec,cfg,p,seed,prior['baseline'])
            copy_verified(ROOT,[folder/'prediction.json',folder/'queries.npz'])
            index.append({'source_image':rec['image'],'path':(folder/'prediction.json').relative_to(output).as_posix(),
                'sha256':sha256(folder/'prediction.json'),'raw_archive_sha256':doc['raw_archive_sha256'],'crops':doc['crop_count']})
            if i==1 or i%10==0 or i==len(cfg['records']):
                print(f'22 konum seed={seed}: {i}/{len(cfg["records"])}; yeni={new}; korunan={reused}; gecen={(time.perf_counter()-started)/60:.1f} dk',flush=True)
                write_json(output/'progress.json',{'status':'INFERENCE','position_seed':seed,'images_done':i,'new':new,'reused':reused})
        for rec in cfg['records']:
            if sha256(safe_path(ROOT/'data/prepared',rec['image']))!=rec['image_sha256']:raise ValueError('22 goruntu cikarim sirasinda degismis')
        for path,digest in immutable.items():
            if sha256(path)!=digest:raise ValueError('22 eski kaynak/agirlik degismis')
        body={'status':'COMPLETED','protocol_sha256':fingerprint(cfg),'position_seed':seed,'records':index,'created_utc':utc_now(),
            'environment':env,'new_images_this_invocation':new,'reused_images_this_invocation':reused,'timing_scope':'progress only, cached Stage1 and resumed images; no FPS or runtime claim'}
        body['content_sha256']=fingerprint(body);validate_seed(body,cfg,plan,seed);write_json(output/f'seed_{seed}.json',body);return 0
    except (Exception,KeyboardInterrupt) as exc:
        traceback.print_exc();write_json(output/'seed_error.json',{'position_seed':seed,'error':str(exc)});return 1


def validate_all(output,cfg,plan,base):
    """Complete all seeds/raw replays before the first new annotation parse."""
    mapped={x['source_image']:x for x in read_json(base/'prediction_index.json')['records']};plans={x['source_image']:x for x in plan['records']};docs={};immutable={}
    for seed in SEEDS:
        cache=validate_seed(read_json(output/f'seed_{seed}.json'),cfg,plan,seed)
        for rec,item in zip(cfg['records'],cache['records']):
            path=safe_path(output,item['path'])
            if sha256(path)!=item['sha256']:raise ValueError('22 tahmin indeksi hash farkli')
            prior=r.references(base,rec,cfg,mapped);p=plans[rec['image']]
            if p['reference12_sha256']!=mapped[rec['image']]['sha256'] or p['original_rects']!=prior['crop_rects']:raise ValueError('22 sabit kaynak/plan farkli')
            doc=checked_prediction(path.parent,rec,cfg,p,seed,prior['baseline'])
            if doc['raw_archive_sha256']!=item['raw_archive_sha256']:raise ValueError('22 ham indeks hash farkli')
            docs[(seed,rec['image'])]=doc;immutable[str(path)]=item['sha256'];immutable[str(path.parent/'queries.npz')]=item['raw_archive_sha256']
    return docs,immutable


def measure(output,cfg,plan,base,folder19):
    docs,immutable=validate_all(output,cfg,plan,base);parts=[];mapping={x['source_image']:x for x in read_json(base/'prediction_index.json')['records']}
    variants=['baseline',g.HYBRID]+[variant(s) for s in SEEDS]
    for i,rec in enumerate(cfg['records'],1):
        prior=r.references(base,rec,cfg,mapping);annotation=ROOT/'data/prepared/visdrone_annotations/val'/(Path(rec['image']).stem+'.txt')
        if sha256(annotation)!=rec['annotation_sha256']:raise ValueError('22 annotation kaynagi degismis')
        immutable[str(annotation)]=rec['annotation_sha256'];immutable[str(safe_path(base,mapping[rec['image']]['path']))]=mapping[rec['image']]['sha256']
        raw=raw_annotation(annotation.read_text(encoding='utf-8-sig'));integral=ignore_integral(raw,rec['width'],rec['height']);gt=prepare_groundtruth(raw,integral)
        data={'gt_denominators':np.array([np.count_nonzero(gt[:,5]==c) for c in range(1,11)],dtype=np.int64)};data['presence']=(data['gt_denominators']>0).astype(np.int64)
        rows={'baseline':prior['baseline'],g.HYBRID:prior[g.HYBRID],**{variant(s):docs[(s,rec['image'])]['detections'] for s in SEEDS}}
        for name in variants:
            det=outside_ignore(xyxy_to_det(rows[name]),integral)[:500];data[name+'_scores']=det[:,4].copy();data[name+'_classes']=det[:,5].astype(np.int8)
            data[name+'_flags']=np.column_stack([g.scoring.fast_flags(det,gt,float(t)) for t in g.scoring.THRESHOLDS])
        parts.append(data)
        if i==1 or i%25==0 or i==len(cfg['records']):print(f'22 ayni VAL metrikleri: {i}/{len(cfg["records"])}',flush=True)
    metrics={v:g.scoring.aggregate(parts,v) for v in variants};frozen=read_json(folder19/'metrics.json')['variants']
    for name in ['baseline',g.HYBRID]:
        g.scoring.compare_metrics(metrics[name],frozen[name])
        if metrics[name]['counts_at_fixed_max500']!=frozen[name]['counts_at_fixed_max500']:raise ValueError('22 eski TP/FP kontrolu farkli')
    for path,digest in immutable.items():
        if sha256(path)!=digest:raise ValueError('22 kaynak metrik sirasinda degismis')
    report={'images':len(cfg['records']),'variants':metrics,'frozen_baseline_and_hybrid_metrics_equal':True,
        'official_binary_parity_verified':False,'metric_scope':'unchanged provisional reference-aligned Python metric, percent units',
        'independent_new_test':False,'multiple_training_seeds':False,'timing_claim':False}
    write_json(output/'metrics.json',report)
    for filename,rows in [('metrics.csv',[{'variant':v,**{k:metrics[v][k] for k in ['AP','AP50','AP75','AR500']}} for v in variants]),
        ('per_class.csv',[{'variant':v,**row} for v in variants for row in metrics[v]['per_class']])]:
        with (output/filename).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return report


def summarize(metrics,plan):
    keys=['AP','AP50','AP75','AR500'];variants=metrics['variants'];control=np.array([[variants[variant(s)][k] for k in keys] for s in SEEDS])
    return {'original_hybrid':{k:variants[g.HYBRID][k] for k in keys},'relocated_seeds':{variant(s):{k:variants[variant(s)][k] for k in keys} for s in SEEDS},
        'relocated_mean':dict(zip(keys,map(float,control.mean(axis=0)))),'relocated_min':dict(zip(keys,map(float,control.min(axis=0)))),
        'relocated_max':dict(zip(keys,map(float,control.max(axis=0)))),'relocated_sample_SD':dict(zip(keys,map(float,control.std(axis=0,ddof=1)))),
        'delta_hybrid_minus_each_seed_pp':{variant(s):{k:variants[g.HYBRID][k]-variants[variant(s)][k] for k in keys} for s in SEEDS},
        'delta_hybrid_minus_relocated_mean_pp':{k:variants[g.HYBRID][k]-float(control[:,i].mean()) for i,k in enumerate(keys)},
        'crops_per_seed':plan['crops_per_seed'],'new_crop_forwards':plan['total_new_crop_forwards'],'source_pixel_area_per_seed':plan['source_pixel_area_per_seed'],
        'unchanged_position_crops':plan['unchanged_position_crops'],'seed_interpretation':'three location randomizations conditional on one trained seed42 model pair; not training uncertainty or a significance test',
        'control_scope':'same ordered crop dimensions/count/source pixel area, EMA80, cached YOLO, preprocessing and fusion; only integer x/y placement changes',
        'limitations':['crop shapes/counts remain conditional on HyCADO proposals, not a detector-independent baseline',
            'relocation changes content, overlaps and union coverage; summed crop area is preserved, union area is not',
            'crop model trained on HyCADO crops; conditional inference control, not random-crop retraining comparison',
            'post-hoc extension on previously examined VAL; no independent test or verified reference binary parity'],
        'training_started':False,'parameter_search':False,'FPS_claim':False,'seed_selection_by_AP':False}


def bundle(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=output,suffix='.zip.tmp');os.close(fd);target=output/'22_konum_kontrolu_rapor.zip'
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.glob('*.json'))+sorted(output.glob('*.csv'))+sorted((output/'predictions').glob('s*/*/prediction.json')):z.write(p,p.relative_to(output).as_posix())
            if (output/'console.log').exists():
                with (output/'console.log').open('rb') as f:f.seek(max(0,(output/'console.log').stat().st_size-90000));z.writestr('console_tail.log',f.read())
        os.replace(tmp,target)
    finally:Path(tmp).unlink(missing_ok=True)
    return target


def backup(root,output):
    paths=[root/'hycado_start/position_control.py']+[root/n for n in ['22_KONUM_KONTROLU.cmd','22_RAPOR_AL.cmd','22_ONCE_OKUYUN.md','SHA256SUMS_22.txt','tests/test_position_control.py'] if (root/n).exists()]
    paths += [p for p in output.rglob('*') if p.is_file() and (p.parent==output or p.relative_to(output).parts[0]=='predictions') and p.suffix in ['.json','.csv','.npz','.zip']]
    target=copy_verified(root,paths);write_json(output/'backup_receipt.json',{'verified_utc':utc_now(),'files':len(paths),'destination':str(target),
        'scope':'22 code/geometry/reports and raw queries only; not all project/models; same D volume is not independent media'})


def run():
    from .__main__ import _Tee
    output=ROOT/'experiments'/OUT;output.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        for lock in [ROOT/'.backup.lock']+[ROOT/'experiments'/folder/name for folder,name in [(OUT,'.evaluation.lock'),
            (g.OUT,'.benchmark.lock'),(g.j.OUT,'.benchmark.lock'),(g.j.e.OUT,'.evaluation.lock'),(g.j.e.training.OUT,'.training.lock'),
            ('stage1_v1_seed42','.training.lock'),('rtdetr_crop_v1_seed42','.training.lock'),(r.val.OUT,'.evaluation.lock'),
            (r.OUT,'.benchmark.lock'),(g.j.optimized.OUT,'.benchmark.lock')]]:stack.enter_context(training_lock(lock))
        with (output/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    write_json(output/'result.json',{'status':'RUNNING'});print('22 CROP KONUM KONTROLU:',utc_now(),flush=True)
                    c12,base,folder19,_,_,immutable=source(ROOT);plan,refs=build_plan(c12,base);immutable.update(refs);env=system_info(include_torch=True)
                    if not env.get('cuda_available'):raise RuntimeError('22 icin CUDA gerekli')
                    cfg={'id':'hycado_crop_position_control_v1','date':'2026-11-02','records':c12['records'],'location_seeds':SEEDS,
                        'geometry_plan_sha256':fingerprint(plan),'source21_metadata_sha256':FILES21,'source12_protocol_sha256':r.P12,
                        'weights':{'YOLO':c12['stage1_weights_sha256'],'crop':c12['crop_weights_sha256']},'fusion':c12['fusion'],
                        'crop_inference':c12['crop_inference'],'runtime_versions':c12['runtime_versions'],'code_sha256':sha256(__file__),
                        'gpu':env['gpu'],'driver':env['nvidia_smi'].split(',')[-1].strip(),'Stage1':'unchanged frozen12 cache; no new YOLO inference',
                        'position_rule':'predeclared uniform integer per-crop placements; no GT/score-based rejection or seed/AP selection',
                        'geometry_frozen_before_new_inference':True,'training_started':False,'timing_claim':False,'post_hoc_extension':True}
                    if (output/'protocol.json').exists() and read_json(output/'protocol.json')!=cfg:raise ValueError('22 protokol/kod/ortam farkli')
                    if (output/'geometry_plan.json').exists() and read_json(output/'geometry_plan.json')!=plan:raise ValueError('22 sabit geometri plani farkli')
                    write_json(output/'protocol.json',cfg);write_json(output/'geometry_plan.json',plan);write_json(output/'environment.json',env)
                    if (output/'staging').exists():shutil.rmtree(output/'staging')
                    for n in ['metrics.json','metrics.csv','per_class.csv','summary.json']:(output/n).unlink(missing_ok=True)
                    print('PLAN SABITLENDI:',plan['images'],'goruntu;',plan['crops_per_seed'],'crop/seed;',plan['total_new_crop_forwards'],'yeni crop cikarimi',flush=True)
                    backup(ROOT,output);reused=0
                    for seed in SEEDS:
                        path=output/f'seed_{seed}.json'
                        if path.exists():validate_seed(read_json(path),cfg,plan,seed);reused+=1;print('TAMAMLANAN SEED KORUNUYOR:',seed,flush=True)
                        else:
                            (output/'seed_error.json').unlink(missing_ok=True)
                            write_json(output/'worker_job.json',{'output':str(output),'position_seed':seed,'protocol_sha256':fingerprint(cfg)})
                            code=r.child_process([sys.executable,'-X','faulthandler','-u','-m','hycado_start.position_control','--worker',str(output/'worker_job.json')],ROOT)
                            if code!=0:raise RuntimeError('22 alt islem durdu: '+str(code))
                            validate_seed(read_json(path),cfg,plan,seed)
                        backup(ROOT,output)
                    metrics=measure(output,cfg,plan,base,folder19);write_json(output/'summary.json',summarize(metrics,plan))
                    for path,digest in immutable.items():
                        if sha256(path)!=digest:raise ValueError('22 eski kaynak/agirlik degismis')
                    backup(ROOT,output)
                    result={'status':'COMPLETED','images':len(cfg['records']),'position_seeds':SEEDS,'crops_per_seed':plan['crops_per_seed'],
                        'new_crop_forwards_in_completed_predictions':plan['total_new_crop_forwards'],'prediction_records':len(cfg['records'])*len(SEEDS),
                        'counts_and_ordered_dimensions_exact':True,'raw_decode_and_independent_NMS_exact':True,'frozen_baseline_and_hybrid_metrics_equal':True,
                        'original_weights_and_results_unchanged':True,'backup_verified':True,'training_started':False,'timing_claim':False,
                        'reused_completed_seeds':reused,'official_binary_parity_verified':False}
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();result={'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR','error':str(exc),
                        'resume':'rerun22; complete seeds and validated per-image predictions reused; partial image replayed'}
                write_json(output/'result.json',result);write_json(output/'progress.json',result);print('SONUC:',result['status'],flush=True)
        report=bundle(output)
        try:backup(ROOT,output)
        except Exception as exc:
            print('22 SON RAPOR YEDEGI ALINAMADI:',str(exc),flush=True)
            if result['status']=='COMPLETED':result.update(status='ERROR',backup_verified=False,error='Final backup failed: '+str(exc));write_json(output/'result.json',result);bundle(output)
        print('GONDERILECEK RAPOR:',report,flush=True);return result['status']=='COMPLETED'


def main():
    if '--worker' in sys.argv:return worker(Path(sys.argv[sys.argv.index('--worker')+1]))
    if '--report-only' in sys.argv:print(bundle(ROOT/'experiments'/OUT),flush=True);return 0
    return 0 if run() else 1


if __name__=='__main__':raise SystemExit(main())
