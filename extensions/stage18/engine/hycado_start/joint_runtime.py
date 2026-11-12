"""Three-method controlled, batch-one runtime; unchanged frozen predictions."""
from __future__ import annotations
import contextlib,csv,math,sys,time,traceback
from pathlib import Path
import numpy as np
from .common import ROOT,sha256,write_json,utc_now,system_info
from .stage1 import fingerprint,training_lock
from .crop_loader import safe_path,read_json
from . import full_image_evaluation as e,optimized_runtime as optimized
r=e.r
OUT='joint_runtime_v1_seed42'
VARIANTS=['baseline','rtdetr_full960','hybrid_a025']
PHASES=[(p,VARIANTS[(p+i)%3]) for p in range(3) for i in range(3)]
FILES19={'protocol.json':'e0d97a7d3359fb68e237a8be64ea14ee1397c3b4cd5a5ee80cb7422d77af3e94',
 'result.json':'465143cf8defb35396094c1c4454192afab1233b37aa198a6e3e4918c93303a1',
 'prediction_index.json':'51195d72561879853a143bd72c7ad487cbaadd2981bdc66fc33c19408607ddff',
 'metrics.json':'cfde2da47bb80884c788f4c591d92722ee5b68c734cef19dce963aafc3784c93'}


def source(root):
    cfg12,base,w1,w2,official,up=optimized.sources(root)
    _,_,full,_,immutable=e.source(root)
    folder=root/'experiments'/e.OUT
    for name,digest in FILES19.items():
        if sha256(folder/name)!=digest:raise ValueError('Tamamlanan 19 kaydi farkli: '+name)
        immutable[str(folder/name)]=digest
    cfg19=read_json(folder/'protocol.json')
    if (sha256(e.__file__)!=cfg19['code_sha256'] or cfg19['source18_weights_sha256']!=e.W18 or
        read_json(folder/'result.json').get('status')!='COMPLETED'):raise ValueError('Dogrulanmis 19 kodu/sonucu gerekli')
    upstream=read_json(root/'experiments'/e.training.OUT/'protocol.json')['upstream_metadata_sha256']
    p15=root/'experiments'/optimized.OUT/'protocol.json'
    if (sha256(p15)!=upstream[p15.relative_to(root).as_posix()] or sha256(optimized.__file__)!=read_json(p15)['code_sha256']):
        raise ValueError('Dogrulanmis 15 cikarim akisi degismis')
    immutable.update({str(root/k):v for k,v in up.items()})
    immutable.update({str(p):sha256(p) for p in [w1,w2,full,official,p15]})
    return cfg12,base,folder,w1,w2,full,official,immutable


def run_full(record,path,model,sync,clock=time.perf_counter):
    sync();start=clock()
    query=e.predict_path(model,path,record);result=e.decode(query,record['width'],record['height'])
    sync();return {'total_ms':(clock()-start)*1000,'crop_count':0},result


def validate_phase(payload,cfg,pass_id,variant):
    if (payload.get('content_sha256')!=fingerprint({k:v for k,v in payload.items() if k!='content_sha256'}) or
        payload.get('protocol_sha256')!=fingerprint(cfg) or payload.get('status')!='COMPLETED' or
        payload.get('pass_id')!=pass_id or payload.get('variant')!=variant):raise ValueError('20 faz kimligi/imzasi farkli')
    expected=[x['image'] for x in r.pass_order(cfg['records'],pass_id)]
    if [x['source_image'] for x in payload['rows']]!=expected:raise ValueError('20 goruntu sirasi/kapsami farkli')
    for row in payload['rows']:
        if row.get('exact_frozen_output') is not True:raise ValueError('20 sabit cikti esligi basarisiz')
        if any(not math.isfinite(row[k]) or row[k]<=0 for k in ['total_ms','peak_allocated_mib','peak_reserved_mib']):raise ValueError('20 gecersiz olcum')
        if row['peak_reserved_mib']+.01<row['peak_allocated_mib']:raise ValueError('Bellek sayaclari tutarsiz')
        crops=cfg['crop_counts'][row['source_image']] if variant=='hybrid_a025' else 0
        if row['crop_count']!=crops:raise ValueError('20 crop kapsami farkli')
    return payload


def worker(job_path):
    job=read_json(job_path);output=Path(job['output']);cfg=read_json(output/'protocol.json');pass_id=job['pass_id'];variant=job['variant']
    try:
        if (pass_id,variant) not in PHASES or job['protocol_sha256']!=fingerprint(cfg) or sha256(__file__)!=cfg['code_sha256']:
            raise ValueError('20 is/protokol/kod farkli')
        c12,base,folder,w1,w2,full,official,immutable=source(ROOT)
        import torch
        env=system_info(include_torch=True)
        if (not torch.cuda.is_available() or env['gpu']!=cfg['gpu'] or env['nvidia_smi'].split(',')[-1].strip()!=cfg['driver']):raise ValueError('20 CUDA/GPU/surucu farkli')
        loading=time.perf_counter();predict1=crop_model=full_model=None;state2=None
        if variant=='rtdetr_full960':full_model=e.load_model(full,official);state1=r.backend_state()
        else:
            predict1=r.proposals.make_predictor(r.proposals.load_ema(w1,r.val.P1),{'inference':c12['stage1_inference']},output)
            state1=r.backend_state()
            if variant=='hybrid_a025':crop_model=r.crop_capture.load_model(w2,official,'cuda');state2=r.backend_state()
        loading_seconds=time.perf_counter()-loading
        def fresh(rec):
            path=safe_path(ROOT/'data/prepared',rec['image']);r.restore_backend(state1)
            if variant=='rtdetr_full960':
                times,result=run_full(rec,path,full_model,torch.cuda.synchronize);return times,None,result,[]
            return optimized.run_image(rec,path,predict1,lambda rgb:r.direct_queries(crop_model,rgb),c12['crop_policy'],
                variant=='hybrid_a025',torch.cuda.synchronize,before_crops=lambda:r.restore_backend(state2))
        order=r.pass_order(cfg['records'],pass_id)
        for rec in order:
            if sha256(safe_path(ROOT/'data/prepared',rec['image']))!=rec['image_sha256']:raise ValueError('Kaynak goruntu degismis')
        mapping={x['source_image']:x for x in read_json(base/'prediction_index.json')['records']}
        newmap={x['source_image']:x for x in read_json(folder/'prediction_index.json')['records']}
        warm=time.perf_counter()
        for rec in order[:cfg['warmup_images']]:fresh(rec)
        torch.cuda.synchronize();warm_seconds=time.perf_counter()-warm;rows=[];started=time.perf_counter()
        for i,rec in enumerate(order,1):
            if variant=='rtdetr_full960':
                item=newmap[rec['image']];path=safe_path(folder,item['path'])
                if sha256(path)!=item['sha256']:raise ValueError('19 referans tahmini degismis')
                expected=e.checked_prediction(path.parent,rec,fingerprint(read_json(folder/'protocol.json')))
            else:expected=r.references(base,rec,c12,mapping)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            times,baseline,result,rects=fresh(rec)
            if result!=expected[variant]:raise ValueError('20 yeni GPU ciktisi sabit sonucuyla birebir farkli: '+rec['image'])
            if variant=='hybrid_a025' and (baseline!=expected['baseline'] or rects!=expected['crop_rects']):raise ValueError('20 hibrit geometri/baseline farkli')
            rows.append({'source_image':rec['image'],'total_ms':times['total_ms'],'crop_count':times['crop_count'],
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'exact_frozen_output':True})
            if i==1 or i%10==0 or i==len(order):
                print(f'20 tur {pass_id+1}/3 {variant}: {i}/{len(order)}; gecen={(time.perf_counter()-started)/60:.1f} dk',flush=True)
                write_json(output/'progress.json',{'status':'MEASURING','pass_id':pass_id,'variant':variant,'images_done':i})
        for rec in order:
            if sha256(safe_path(ROOT/'data/prepared',rec['image']))!=rec['image_sha256']:raise ValueError('Goruntu olcum sirasinda degismis')
        for path,digest in immutable.items():
            if sha256(path)!=digest:raise ValueError('Olcum kaynagi/agirligi degismis')
        body={'status':'COMPLETED','protocol_sha256':fingerprint(cfg),'pass_id':pass_id,'variant':variant,'created_utc':utc_now(),
            'environment':env,'model_load_seconds_excluded':loading_seconds,'warmup_seconds_excluded':warm_seconds,'rows':rows}
        body['content_sha256']=fingerprint(body);validate_phase(body,cfg,pass_id,variant)
        write_json(output/f'pass_{pass_id+1}_{variant}.json',body);return 0
    except (Exception,KeyboardInterrupt) as exc:
        traceback.print_exc();write_json(output/'phase_error.json',{'error':str(exc),'pass_id':pass_id,'variant':variant});return 1


def summarize(phases,metrics):
    if {(p['pass_id'],p['variant']) for p in phases}!=set(PHASES) or len(phases)!=9:raise ValueError('9 faz gerekli')
    result={'variants':{},'paired_passes':[],'training_started':False,'parameter_search':False,'new_AP_computed':False,
       'scope':'warmed batch1 sequential path-to-CPU detections; OS file cache primed; load, warmup, reference checks and output I/O excluded',
       'memory_scope':'Torch allocated/reserved, resident weights included; allocator warm cache included; not whole-system GPU memory',
       'source_accuracy':'frozen 19 metrics; no new AP selection','resumption_note':'whole phases can span invocations; inspect per-phase timestamps'}
    for v in VARIANTS:
        rows=[x for p in phases if p['variant']==v for x in p['rows']];a=np.array([x['total_ms'] for x in rows])
        result['variants'][v]={'mean_ms':float(a.mean()),'median_ms':float(np.median(a)),'p95_ms':float(np.percentile(a,95)),
            'throughput_images_per_second':float(1000/a.mean()),'measured_image_calls':len(rows),
            'peak_allocated_mib':max(x['peak_allocated_mib'] for x in rows),'peak_reserved_mib':max(x['peak_reserved_mib'] for x in rows),
            **{k:metrics['variants'][v][k] for k in ['AP','AP50','AP75','AR500']}}
    for p in range(3):
        means={v:float(np.mean([x['total_ms'] for phase in phases if phase['pass_id']==p and phase['variant']==v for x in phase['rows']])) for v in VARIANTS}
        result['paired_passes'].append({'pass_id':p,**means,'hybrid_to_YOLO_ratio':means['hybrid_a025']/means['baseline'],
            'hybrid_to_RTDETR_ratio':means['hybrid_a025']/means['rtdetr_full960']})
    return result


def bundle(output):
    import zipfile,os,tempfile
    output=Path(output);output.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=output,suffix='.zip.tmp');os.close(fd)
    target=output/'20_ortak_hiz_bellek_rapor.zip'
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.glob('*.json'))+sorted(output.glob('*.csv')):z.write(p,p.name)
            if (output/'console.log').exists():
                with (output/'console.log').open('rb') as f:
                    f.seek(max(0,(output/'console.log').stat().st_size-100000));z.writestr('console_tail.log',f.read())
        os.replace(tmp,target)
    finally:Path(tmp).unlink(missing_ok=True)
    return target


def backup(root,output):
    import os,shutil,tempfile
    target=root.parent/'HyCADO_Yedekleri/20_Ortak_Olcum'
    paths=[root/'hycado_start/joint_runtime.py']+[root/n for n in ['20_ORTAK_HIZ_BELLEK.cmd','20_RAPOR_AL.cmd','20_ONCE_OKUYUN.md','SHA256SUMS_20.txt'] if (root/n).exists()]
    paths += [p for p in output.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.zip']]
    for path in paths:
        digest=sha256(path);dest=target/digest[:12]/path.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists() or sha256(dest)!=digest:
            fd,tmp=tempfile.mkstemp(dir=dest.parent,suffix='.tmp');os.close(fd)
            try:
                shutil.copy2(path,tmp)
                if sha256(tmp)!=digest:raise ValueError('20 yedek farkli')
                os.replace(tmp,dest)
            finally:Path(tmp).unlink(missing_ok=True)


def run():
    from .__main__ import _Tee
    output=ROOT/'experiments'/OUT;output.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        locks=[ROOT/'.backup.lock']+[ROOT/'experiments'/f/n for f,n in [(OUT,'.benchmark.lock'),
            (e.OUT,'.evaluation.lock'),(e.training.OUT,'.training.lock'),('stage1_v1_seed42','.training.lock'),
            ('rtdetr_crop_v1_seed42','.training.lock'),(r.val.OUT,'.evaluation.lock'),(r.OUT,'.benchmark.lock'),(optimized.OUT,'.benchmark.lock')]]
        for lock in locks:stack.enter_context(training_lock(lock))
        with (output/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    write_json(output/'result.json',{'status':'RUNNING'});c12,base,folder,*_,immutable=source(ROOT)
                    env=system_info(include_torch=True)
                    if not env.get('cuda_available'):raise RuntimeError('CUDA gerekli')
                    cfg={'id':'hycado_three_method_runtime_v1','date':'2026-11-02','source19_metadata_sha256':FILES19,
                        'records':c12['records'],'phase_order':[[p,v] for p,v in PHASES],'warmup_images':5,
                        'crop_counts':{x['source_image']:x['crops'] for x in read_json(base/'prediction_index.json')['records']},
                        'source_protocols':{'12':r.P12,'18':e.P18},'weights':{'YOLO':c12['stage1_weights_sha256'],'crop':c12['crop_weights_sha256'],'full':e.W18},
                        'runtime_versions':c12['runtime_versions'],'gpu':env['gpu'],'driver':env['nvidia_smi'].split(',')[-1].strip(),
                        'methods':'unchanged 15 YOLO/hybrid, unchanged 19 full RTDETR; batch1; no GT or training',
                        'image_order':'same 13 hash order within each pass','phase_order_rule':'cyclic three positions; each method once in each position',
                        'code_sha256':sha256(__file__)}
                    if (output/'protocol.json').exists() and read_json(output/'protocol.json')!=cfg:raise ValueError('20 protokol/ortam/kod farkli')
                    write_json(output/'protocol.json',cfg);write_json(output/'environment.json',env)
                    for name in ['summary.json','summary.csv']:(output/name).unlink(missing_ok=True)
                    backup(ROOT,output);phases=[];reused=0
                    for p,v in PHASES:
                        path=output/f'pass_{p+1}_{v}.json'
                        if not path.exists():
                            (output/'phase_error.json').unlink(missing_ok=True)
                            write_json(output/'worker_job.json',{'output':str(output),'protocol_sha256':fingerprint(cfg),'pass_id':p,'variant':v})
                            code=r.child_process([sys.executable,'-X','faulthandler','-u','-m','hycado_start.joint_runtime','--worker',str(output/'worker_job.json')],ROOT)
                            if code!=0:raise RuntimeError('20 olcum alt islemi durdu: '+str(code))
                        else:reused+=1;print('TAMAMLANAN FAZ KULLANILIYOR:',path.name,flush=True)
                        phases.append(validate_phase(read_json(path),cfg,p,v))
                    for path,digest in immutable.items():
                        if sha256(path)!=digest:raise ValueError('Sabit kaynak/agirlilar degismis')
                    summary=summarize(phases,read_json(folder/'metrics.json'));write_json(output/'summary.json',summary)
                    with (output/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
                        w=csv.DictWriter(f,fieldnames=['variant']+list(summary['variants']['baseline']));w.writeheader()
                        for v in VARIANTS:w.writerow({'variant':v,**summary['variants'][v]})
                    result={'status':'COMPLETED','phases':9,'images_per_phase':548,'measured_image_calls':4932,
                        'all_outputs_exact_frozen_references':True,'original_weights_and_results_unchanged':True,
                        'training_started':False,'parameter_search':False,'reused_completed_phases':reused}
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();result={'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR','error':str(exc),
                        'resume':'rerun 20; whole complete phases reused; partial phase restarts'}
                write_json(output/'result.json',result);write_json(output/'progress.json',result);print('SONUC:',result['status'],flush=True)
        path=bundle(output)
        try:backup(ROOT,output)
        except Exception as exc:print('20 RAPOR YEDEGI ALINAMADI:',str(exc),flush=True)
        print('GONDERILECEK RAPOR:',path,flush=True)
        return result['status']=='COMPLETED'


def main():
    if '--worker' in sys.argv:return worker(Path(sys.argv[sys.argv.index('--worker')+1]))
    if '--report-only' in sys.argv:print(bundle(ROOT/'experiments'/OUT),flush=True);return 0
    return 0 if run() else 1


if __name__=='__main__':raise SystemExit(main())
