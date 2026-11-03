"""CPU-only paired uncertainty analysis of frozen 12/19/21/22 outputs.

Proxy-cluster bootstrap is primary; iid-image bootstrap is a sensitivity
analysis. Original records, AP aggregation, ties, ignores and caps are retained.
"""
from __future__ import annotations
import contextlib,csv,os,shutil,sys,tempfile,time,traceback,zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from .common import ROOT,sha256,write_json,system_info,utc_now
from .stage1 import fingerprint,training_lock
from .crop_loader import read_json,safe_path
from . import position_control as p,bootstrap_statistics as stats
from .dev_metrics import raw_annotation,ignore_integral,prepare_groundtruth,outside_ignore,xyxy_to_det
r=p.r
OUT='paired_bootstrap_v1_seed42'
B=2000
BLOCK=25
WORKERS=2
DRAW_SEED=20261102
FILES22={'protocol.json':'cd43edb2b765eec8cb4e150d42ca4a757fafc971965467c963c3b9dd857bfb94',
 'geometry_plan.json':'ae528ddc16806af1aa8ef0ce382b022686266d3106d91b2b20ea9f7dd5afceb7',
 'result.json':'3551e36751f99719fca689166b9d1407a2675c7f8ba2f628abb445bb1fe5e1a9',
 'metrics.json':'9e8fc19d85913cc8d0fa3e019428f290c2b907e4ec0126a844c7e625165c0730',
 'summary.json':'08c3d478a8795131206e242fbe118a3cb405d986df30a973f6af13134d270969'}


def source(root):
    c12,base,f19,_,_,immutable=p.source(root);f21=root/'experiments'/p.g.OUT;f22=root/'experiments'/p.OUT
    for name,digest in FILES22.items():
        if sha256(f22/name)!=digest:raise ValueError('Tamamlanan 22 kaydi farkli: '+name)
        immutable[str(f22/name)]=digest
    c22=read_json(f22/'protocol.json');plan=read_json(f22/'geometry_plan.json');res=read_json(f22/'result.json')
    if c22['records']!=c12['records'] or c22['code_sha256']!=sha256(p.__file__) or fingerprint(plan)!=c22['geometry_plan_sha256'] or res.get('status')!='COMPLETED' or res.get('backup_verified') is not True:raise ValueError('23 icin dogrulanmis 22 gerekli')
    inputs=[{'source_image':x['source_image'],'image_sha256':x['image_sha256'],'width':x['width'],'height':x['height'],
        'reference12_sha256':x['reference12_sha256'],'crop_rects':x['original_rects']} for x in plan['records']]
    if p.make_plan(inputs)!=plan:raise ValueError('22 yerlesim plani farkli')
    seeds={}
    for seed in p.SEEDS:
        path=f22/f'seed_{seed}.json';doc=p.validate_seed(read_json(path),c22,plan,seed);seeds[seed]=doc;immutable[str(path)]=sha256(path)
    frozen19=read_json(f19/'metrics.json')['variants'];frozen21=read_json(f21/'metrics.json')['variants'];frozen22=read_json(f22/'metrics.json')['variants']
    expected={**{v:frozen19[v] for v in ['baseline','hybrid_a025','rtdetr_full960']},'fixed_grid_a025':frozen21['fixed_grid_a025'],
        **{p.variant(s):frozen22[p.variant(s)] for s in p.SEEDS}}
    ctx={'base':base,'f19':f19,'f21':f21,'f22':f22,'c12':c12,'c19':read_json(f19/'protocol.json'),
        'c21':read_json(f21/'protocol.json'),'c22':c22,'plan':{x['source_image']:x for x in plan['records']},'expected':expected,
        'm12':{x['source_image']:x for x in read_json(base/'prediction_index.json')['records']},
        'm19':{x['source_image']:x for x in read_json(f19/'prediction_index.json')['records']},
        'm21':{x['source_image']:x for x in read_json(f21/'prediction_index.json')['records']},
        'm22':{s:{x['source_image']:x for x in doc['records']} for s,doc in seeds.items()}}
    return c12,ctx,immutable


def paths_for(rec,ctx):
    image=rec['image'];result={}
    for key,folder,mapping in [('12',ctx['base'],ctx['m12']),('19',ctx['f19'],ctx['m19']),('21',ctx['f21'],ctx['m21'])]:
        item=mapping[image];result[key]=(safe_path(folder,item['path']),item['sha256'])
    for seed in p.SEEDS:
        item=ctx['m22'][seed][image];result[f'22s{seed}']=(safe_path(ctx['f22'],item['path']),item['sha256'])
    return result


def load_rows(rec,ctx):
    paths=paths_for(rec,ctx);out={}
    for key,(path,digest) in paths.items():
        if sha256(path)!=digest:raise ValueError('23 kaynak tahmin degismis: '+str(path))
        cfg=ctx['c12'] if key=='12' else ctx['c19'] if key=='19' else ctx['c21'] if key=='21' else ctx['c22']
        doc=r.val.checked_json(path,fingerprint(cfg),rec)
        if key=='12':out.update(baseline=doc['baseline'],hybrid_a025=doc['hybrid_a025'])
        elif key=='19':out['rtdetr_full960']=doc['rtdetr_full960']
        elif key=='21':out['fixed_grid_a025']=doc['fixed_grid_a025']
        else:
            seed=int(key[-1]);plan=ctx['plan'][rec['image']];item=ctx['m22'][seed][rec['image']]
            if (doc['position_seed']!=seed or doc['variant']!=p.variant(seed) or doc['reference12_sha256']!=ctx['m12'][rec['image']]['sha256'] or
                doc['plan_record_sha256']!=fingerprint(plan) or doc['crop_rects']!=plan[p.variant(seed)] or doc['raw_archive_sha256']!=item['raw_archive_sha256']):raise ValueError('23 sabit 22 baglantisi farkli')
            out[p.variant(seed)]=doc['detections']
    for name,rows in out.items():
        clean,bad=r.old.canonical_detections(rows,rec['width'],rec['height'])
        if bad or clean!=rows or len(rows)>(300 if name=='rtdetr_full960' else 1000):raise ValueError('23 tahmin bicimi farkli')
    return out


def validate_part(data):
    if (data['gt_denominators'].shape!=(10,) or data['presence'].shape!=(10,) or not np.issubdtype(data['gt_denominators'].dtype,np.integer) or np.any(data['gt_denominators']<0) or
        not np.array_equal(data['presence'],data['gt_denominators']>0)):raise ValueError('23 GT sayaci gecersiz')
    for v in stats.NAMES:
        scores=data[v+'_scores'];classes=data[v+'_classes'];flags=data[v+'_flags']
        if (len(scores)>500 or classes.shape!=scores.shape or flags.shape!=(len(scores),10) or not np.issubdtype(classes.dtype,np.integer) or not np.issubdtype(flags.dtype,np.integer) or not np.isfinite(scores).all() or
            np.any((scores<=0)|(scores>1)) or np.any(np.diff(scores)>0) or not np.isin(classes,np.arange(1,11)).all() or not np.isin(flags,[-1,0,1]).all()):raise ValueError('23 eslesme cache bicimi farkli')
    return data


def save_part(folder,data,bindings,diagnostic):
    folder.parent.mkdir(parents=True,exist_ok=True);staging=folder.parent.parent/'staging';staging.mkdir(exist_ok=True)
    tmp=Path(tempfile.mkdtemp(dir=staging,prefix=folder.name+'_'))
    try:
        np.savez_compressed(tmp/'matches.npz',**data);body={'bindings':bindings,'npz_sha256':sha256(tmp/'matches.npz'),'annotation_diagnostic':diagnostic}
        r.val.signed_json(tmp/'summary.json',body)
        if folder.exists():raise FileExistsError('23 tamamlanmis cache uzerine yazilmadi')
        tmp.rename(folder)
    finally:
        if tmp.exists():shutil.rmtree(tmp)


def read_part(folder,bindings):
    meta=read_json(folder/'summary.json')
    if meta['content_sha256']!=fingerprint({k:v for k,v in meta.items() if k!='content_sha256'}) or meta['bindings']!=bindings or sha256(folder/'matches.npz')!=meta['npz_sha256']:raise ValueError('23 eslesme cache kimligi farkli')
    with np.load(folder/'matches.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    return validate_part(data),meta


def prepare_parts(root,output,cfg,ctx):
    parts=[];index=[];immutable={};started=time.perf_counter()
    for i,rec in enumerate(cfg['records'],1):
        annotation=root/'data/prepared/visdrone_annotations/val'/(Path(rec['image']).stem+'.txt');paths=paths_for(rec,ctx)
        if sha256(annotation)!=rec['annotation_sha256']:raise ValueError('23 ham annotation degismis')
        immutable[str(annotation)]=rec['annotation_sha256']
        for path,digest in paths.values():
            if sha256(path)!=digest:raise ValueError('23 kaynak JSON degismis')
            immutable[str(path)]=digest
        bindings={'protocol_sha256':fingerprint(cfg),'source_image':rec['image'],'annotation_sha256':rec['annotation_sha256'],
            'source_prediction_sha256':{k:d for k,(_,d) in paths.items()}}
        folder=output/'matches'/Path(rec['image']).stem
        if not folder.exists():
            rows=load_rows(rec,ctx);raw=raw_annotation(annotation.read_text(encoding='utf-8-sig'));integral=ignore_integral(raw,rec['width'],rec['height']);gt=prepare_groundtruth(raw,integral)
            data={'gt_denominators':np.array([np.count_nonzero(gt[:,5]==c) for c in range(1,11)],dtype=np.int64)};data['presence']=(data['gt_denominators']>0).astype(np.int64)
            for v in stats.NAMES:
                det=outside_ignore(xyxy_to_det(rows[v]),integral)[:500];data[v+'_scores']=det[:,4].copy();data[v+'_classes']=det[:,5].astype(np.int8)
                data[v+'_flags']=np.column_stack([p.g.scoring.fast_flags(det,gt,float(t)) for t in p.g.scoring.THRESHOLDS])
            validate_part(data);save_part(folder,data,bindings,{'raw_rows':len(raw),'sorted_raw_rows_sha256':fingerprint(sorted(raw.tolist()))})
        data,meta=read_part(folder,bindings);parts.append(data)
        copy_verified(root,[folder/'matches.npz',folder/'summary.json'])
        index.append({'source_image':rec['image'],'path':folder.relative_to(output).as_posix(),'summary_sha256':sha256(folder/'summary.json'),
            'npz_sha256':meta['npz_sha256'],'annotation_diagnostic':meta['annotation_diagnostic']})
        if i==1 or i%25==0 or i==len(cfg['records']):
            print(f'23 CPU eslesme hazirligi: {i}/{len(cfg["records"])}; gecen={(time.perf_counter()-started)/60:.1f} dk',flush=True)
            write_json(output/'progress.json',{'status':'MATCH_CACHE','images_done':i})
    write_json(output/'cache_index.json',{'protocol_sha256':fingerprint(cfg),'records':index});return parts,index,immutable


def check_engine(parts,engine,groups,expected):
    point={name:p.g.scoring.aggregate(parts,name) for name in stats.NAMES}
    for name in stats.NAMES:
        p.g.scoring.compare_metrics(point[name],expected[name])
        if point[name]['counts_at_fixed_max500']!=expected[name]['counts_at_fixed_max500']:raise ValueError('23 sabit TP/FP farkli: '+name)
    checks=[]
    for label,freq in [('original',np.ones(len(parts),dtype=np.int64))]+[(mode,stats.frequencies(groups,mode,0,DRAW_SEED)) for mode in stats.MODES]:
        fast=stats.evaluate(engine,freq);expanded=[part for part,n in zip(parts,freq) for _ in range(int(n))]
        reference=point if label=='original' else {v:p.g.scoring.aggregate(expanded,v) for v in stats.NAMES}
        literal=np.array([[reference[v][k] for k in stats.KEYS] for v in stats.NAMES])
        if not np.allclose(fast,literal,rtol=0,atol=1e-10,equal_nan=True):raise ValueError('23 hizli/literal AP esligi farkli: '+label)
        checks.append({'sample':label,'max_absolute_difference':float(np.nanmax(np.abs(fast-literal)))})
        print('23 HIZLI / LITERAL METRIK ESLIGI:',label,'OK',flush=True)
    return stats.evaluate(engine,np.ones(len(parts),dtype=np.int64)),point,checks


def save_block(folder,values,counts,cfg,mode,start):
    folder.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=folder.parent,suffix='.npz.tmp');os.close(fd)
    try:
        with open(tmp,'wb') as f:np.savez_compressed(f,values=values,counts=counts)
        os.replace(tmp,folder.with_suffix('.npz'))
    finally:Path(tmp).unlink(missing_ok=True)
    body={'protocol_sha256':fingerprint(cfg),'mode':mode,'start':start,'size':len(values),'npz_sha256':sha256(folder.with_suffix('.npz'))}
    r.val.signed_json(folder.with_suffix('.json'),body)


def read_block(folder,cfg,mode,start,groups,size):
    meta=read_json(folder.with_suffix('.json'))
    if (meta['content_sha256']!=fingerprint({k:v for k,v in meta.items() if k!='content_sha256'}) or meta['protocol_sha256']!=fingerprint(cfg) or
        meta['mode']!=mode or meta['start']!=start or meta['size']!=size or meta['npz_sha256']!=sha256(folder.with_suffix('.npz'))):raise ValueError('23 bootstrap blok kimligi farkli')
    with np.load(folder.with_suffix('.npz'),allow_pickle=False) as z:values=z['values'];counts=z['counts']
    expected=np.array([stats.frequencies(groups,mode,i,DRAW_SEED) for i in range(start,start+size)])
    if values.shape!=(size,len(stats.NAMES),4) or counts.shape!=expected.shape or not np.array_equal(counts,expected):raise ValueError('23 bootstrap cekilisi/boyutu farkli')
    if np.isinf(values).any() or np.any((values[np.isfinite(values)]<0)|(values[np.isfinite(values)]>100+1e-9)):raise ValueError('23 bootstrap metrik araligi gecersiz')
    return values,counts


def run_bootstrap(output,cfg,engine,groups):
    values={};summary={};new_done=0;compute_seconds=0.;all_done=0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for mode in stats.MODES:
            blocks=[];frequencies=[]
            for start in range(0,B,BLOCK):
                size=min(BLOCK,B-start);folder=output/'bootstrap'/f'{mode}_{start:04d}'
                if not folder.with_suffix('.json').exists():
                    counts=np.array([stats.frequencies(groups,mode,i,DRAW_SEED) for i in range(start,start+size)],dtype=np.uint16)
                    clock=time.perf_counter();result=np.stack(list(pool.map(lambda f:stats.evaluate(engine,f),counts)))
                    compute_seconds+=time.perf_counter()-clock;new_done+=size;save_block(folder,result,counts,cfg,mode,start)
                result,counts=read_block(folder,cfg,mode,start,groups,size)
                undefined=(counts@engine['presence']).sum(axis=1)==0
                if not np.array_equal(np.isnan(result).all(axis=(1,2)),undefined) or np.any(np.isnan(result).any(axis=(1,2))!=undefined):raise ValueError('23 tanimsiz tekrar/GT kapsami farkli')
                blocks.append(result);frequencies.append(counts);all_done+=size
                copy_verified(ROOT,[folder.with_suffix('.json'),folder.with_suffix('.npz')])
                eta=compute_seconds/max(1,new_done)*(len(stats.MODES)*B-all_done)/60 if new_done else None
                print(f'23 {mode}: {start+size}/{B}; bootstrap kalan ~{eta:.1f} dk' if eta is not None else f'23 {mode}: {start+size}/{B}; kayitlar korunuyor',flush=True)
                write_json(output/'progress.json',{'status':'BOOTSTRAP','mode':mode,'replicates_done':start+size,'new_replicates_this_invocation':new_done,'remaining_minutes_estimate':eta})
            values[mode]=np.concatenate(blocks);all_counts=np.concatenate(frequencies)
            summary[mode]={'replicate_image_count_min':int(all_counts.sum(axis=1).min()),'replicate_image_count_max':int(all_counts.sum(axis=1).max()),
                'target_classes_present_min':int(np.count_nonzero(all_counts@engine['presence'],axis=1).min()),'independent_units_assumed':len(groups['labels']) if mode=='cluster' else len(groups['image_to_group'])}
    return values,summary


def write_results(output,cfg,groups,index,point,metrics,draws,sampling,checks,ctx):
    intervals={mode:stats.intervals(point,draws[mode]) for mode in stats.MODES}
    report={'primary_mode':'cluster','sensitivity_mode':'image','groups':groups,'sampling':sampling,'intervals':intervals,
        'engine_checks':checks,'official_binary_parity_verified':False,'training_uncertainty_measured':False,'post_hoc_exploratory':True,
        'no_GPU_inference':True,'no_training':True,'raw_prediction_replay':'23 uses frozen final JSONs; no raw query decoder or model rerun; prior steps verified raw replay'}
    write_json(output/'summary.json',report);write_json(output/'point_metrics.json',{'variants':metrics})
    with (output/'accuracy_intervals.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['resampling','variant','metric','point','lower95','upper95','bootstrap_SE']);w.writeheader()
        for mode,data in intervals.items():
            for v,rows in data['methods'].items():
                for key,row in rows.items():w.writerow({'resampling':mode,'variant':v,'metric':key,**row})
    with (output/'paired_differences.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['resampling','contrast','metric','point','lower95','upper95','bootstrap_SE']);w.writeheader()
        for mode,data in intervals.items():
            for v,rows in data['differences_pp'].items():
                for key,row in rows.items():w.writerow({'resampling':mode,'contrast':v,'metric':key,**row})
    diagnostics={x['source_image']:x['annotation_diagnostic'] for x in index};audit=[]
    for entry in groups['duplicate_image_hash_groups']:
        audit.append({**entry,'annotation_hashes':{x['image']:x['annotation_sha256'] for x in cfg['records'] if x['image'] in entry['images']},
            'parsed_annotation_diagnostics':{image:diagnostics[image] for image in entry['images']},
            'action':'all original records retained; resampled together in primary proxy-cluster analysis'})
    write_json(output/'duplicate_audit.json',audit)
    # Keep historical runtime sessions separate: do not fabricate one joint four-method benchmark.
    sessions=[('20',ROOT/'experiments'/p.g.j.OUT),('21',ROOT/'experiments'/p.g.OUT)];runtime=[]
    for label,folder in sessions:
        for name,row in read_json(folder/'summary.json')['variants'].items():runtime.append({'source_step':label,'variant':name,**{k:row[k] for k in ['mean_ms','median_ms','p95_ms','throughput_images_per_second','peak_allocated_mib','peak_reserved_mib']}})
    with (output/'historical_runtime_sessions.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(runtime[0]));w.writeheader();w.writerows(runtime)
    text=['# HyCADO frozen results and paired uncertainty','',
        'Exploratory percentile intervals conditional on one trained model pair. Filename/hash clusters are proxies, not verified independent videos.',
        '', '| Method | AP | Cluster 95% interval | AP50 | AR500 |','|---|---:|---:|---:|---:|']
    for name,row in intervals['cluster']['methods'].items():text.append(f'| {name} | {row["AP"]["point"]:.3f} | [{row["AP"]["lower95"]:.3f}, {row["AP"]["upper95"]:.3f}] | {row["AP50"]["point"]:.3f} | {row["AR500"]["point"]:.3f} |')
    text+=['','Primary exploratory AP contrasts (percentage points):']
    for name in intervals['cluster']['primary_exploratory_AP_contrasts']:
        row=intervals['cluster']['differences_pp'][name]['AP'];text.append(f'- {name}: {row["point"]:.3f}, 95% interval [{row["lower95"]:.3f}, {row["upper95"]:.3f}].')
    text+=['','Marginal intervals, no hypothesis-test p-values or multiplicity correction. Class-image presence weights are recomputed in every resample.',
        'The mean of the three relocation metrics is a fixed-randomization summary, not an ensemble detector. Training variation is not estimated.',
        'Official binary parity remains unverified. Previously examined VAL is not an independent new test.',
        'Runtime rows are historical and labelled by session 20/21. No new runtime or GPU inference took place.']
    (output/'results_summary.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def copy_verified(root,paths):
    target=root.parent/'HyCADO_Yedekleri/23_Bootstrap/HyCADO_Temiz_Baslangic'
    for path in paths:
        digest=sha256(path);dest=target/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and sha256(dest)==digest:continue
        fd,tmp=tempfile.mkstemp(dir=dest.parent,suffix='.tmp');os.close(fd)
        try:
            shutil.copy2(path,tmp)
            if sha256(tmp)!=digest:raise ValueError('23 yedek farkli')
            os.replace(tmp,dest)
        finally:Path(tmp).unlink(missing_ok=True)
    return target


def backup(root,output):
    names=['hycado_start/paired_bootstrap.py','hycado_start/bootstrap_statistics.py','tests/test_paired_bootstrap.py',
        '23_ESLI_BOOTSTRAP.cmd','23_RAPOR_AL.cmd','23_ONCE_OKUYUN.md','SHA256SUMS_23.txt']
    paths=[root/n for n in names if (root/n).exists()]+[x for x in output.rglob('*') if x.is_file() and 'staging' not in x.relative_to(output).parts and x.suffix in ['.json','.npz','.csv','.zip','.md']]
    target=copy_verified(root,paths);write_json(output/'backup_receipt.json',{'verified_utc':utc_now(),'files':len(paths),'destination':str(target),'scope':'23 code, match cache, bootstrap blocks and reports only; same disk, not independent media'})


def bundle(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=output,suffix='.zip.tmp');os.close(fd);target=output/'23_esli_bootstrap_rapor.zip'
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(output.rglob('*')):
                if path.is_file() and 'staging' not in path.relative_to(output).parts and path.suffix in ['.json','.csv','.npz','.md']:z.write(path,path.relative_to(output).as_posix())
            if (output/'console.log').exists():
                with (output/'console.log').open('rb') as f:f.seek(max(0,(output/'console.log').stat().st_size-90000));z.writestr('console_tail.log',f.read())
        os.replace(tmp,target)
    finally:Path(tmp).unlink(missing_ok=True)
    return target


def run():
    from .__main__ import _Tee
    output=ROOT/'experiments'/OUT;output.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        for lock in [ROOT/'.backup.lock']+[ROOT/'experiments'/folder/name for folder,name in [(OUT,'.analysis.lock'),(p.OUT,'.evaluation.lock'),
            (p.g.OUT,'.benchmark.lock'),(p.g.j.OUT,'.benchmark.lock'),(p.g.j.e.OUT,'.evaluation.lock'),(r.val.OUT,'.evaluation.lock'),
            ('stage1_v1_seed42','.training.lock'),('rtdetr_crop_v1_seed42','.training.lock'),(p.g.j.e.training.OUT,'.training.lock'),
            (r.OUT,'.benchmark.lock'),(p.g.j.optimized.OUT,'.benchmark.lock')]]:stack.enter_context(training_lock(lock))
        with (output/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    write_json(output/'result.json',{'status':'RUNNING'});print('23 CPU ESLI BOOTSTRAP:',utc_now(),flush=True)
                    c12,ctx,immutable=source(ROOT);groups=stats.make_groups(c12['records'])
                    cfg={'id':'hycado_paired_bootstrap_v1','date':'2026-11-02','records':c12['records'],'names':stats.NAMES,'keys':stats.KEYS,
                        'groups':groups,'replicates_per_mode':B,'modes':stats.MODES,'block_size':BLOCK,'workers':WORKERS,'draw_seed':DRAW_SEED,
                        'source22_metadata_sha256':FILES22,'runtime_versions':c12['runtime_versions'],
                        'code_sha256':{'paired_bootstrap.py':sha256(__file__),'bootstrap_statistics.py':sha256(stats.__file__)},
                        'resampling':'paired frequency draws, canonical original image order; stable ties exactly preserved',
                        'metric':'same literal class-image-weighted AP; dynamic class presence per replicate; global max500 after ignore',
                        'interval':'marginal percentile 95%; no p-values or multiple-testing correction; post-hoc exploratory',
                        'training_started':False,'GPU_inference':False,'training_uncertainty':False,'verified_sequence_groups':False}
                    if (output/'protocol.json').exists() and read_json(output/'protocol.json')!=cfg:raise ValueError('23 protokol/kod degismis')
                    write_json(output/'protocol.json',cfg);write_json(output/'groups.json',groups);write_json(output/'environment.json',system_info(False))
                    if (output/'staging').exists():shutil.rmtree(output/'staging')
                    for n in ['summary.json','accuracy_intervals.csv','paired_differences.csv','results_summary.md']:(output/n).unlink(missing_ok=True)
                    print('KUME SAYISI:',len(groups['labels']),'; goruntu:',len(cfg['records']),'; tekrar:',B,'x 2',flush=True);backup(ROOT,output)
                    parts,index,more=prepare_parts(ROOT,output,cfg,ctx);immutable.update(more);engine=stats.compile_engine(parts)
                    point,metrics,checks=check_engine(parts,engine,groups,ctx['expected']);write_json(output/'engine_checks.json',checks);backup(ROOT,output)
                    draws,sampling=run_bootstrap(output,cfg,engine,groups)
                    write_results(output,cfg,groups,index,point,metrics,draws,sampling,checks,ctx)
                    for path,digest in immutable.items():
                        if sha256(path)!=digest:raise ValueError('23 sabit kaynak degismis')
                    backup(ROOT,output);result={'status':'COMPLETED','images':len(cfg['records']),'variants':len(stats.NAMES),
                        'proxy_clusters':len(groups['labels']),'replicates_per_mode':B,'modes':stats.MODES,'frozen_point_metrics_exact':True,
                        'fast_vs_literal_engine_exact':True,'training_started':False,'GPU_inference':False,'original_results_unchanged':True,'backup_verified':True,
                        'official_binary_parity_verified':False,'post_hoc_exploratory':True}
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();result={'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR','error':str(exc),'resume':'rerun23; complete per-image matches and bootstrap blocks retained'}
                write_json(output/'result.json',result);write_json(output/'progress.json',result);print('SONUC:',result['status'],flush=True)
        report=bundle(output)
        try:backup(ROOT,output)
        except Exception as exc:
            print('23 SON RAPOR YEDEGI ALINAMADI:',str(exc),flush=True)
            if result['status']=='COMPLETED':result.update(status='ERROR',backup_verified=False,error='Final backup failed: '+str(exc));write_json(output/'result.json',result);bundle(output)
        print('GONDERILECEK RAPOR:',report,flush=True);return result['status']=='COMPLETED'


def main():
    if '--report-only' in sys.argv:print(bundle(ROOT/'experiments'/OUT),flush=True);return 0
    return 0 if run() else 1


if __name__=='__main__':raise SystemExit(main())
