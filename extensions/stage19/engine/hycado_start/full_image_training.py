"""Fixed 80-round full-image baseline; reuse unchanged checked 10 engine."""
from __future__ import annotations
import contextlib
from pathlib import Path
import sys
import traceback
from .common import ROOT,sha256,write_json,utc_now,system_info
from .stage1 import fingerprint,training_lock,memory_snapshot
from .crop_loader import read_json
from . import full_image_pilot as p

OUT='rtdetr_full_image_v1_seed42'
FILES17={'protocol.json':'ac5830bb09b8aef765d1aa49afb2aca26923aa9673773a789bef7633497fbc0e',
         'result.json':'778fa0bd49831d849f7cde8371b09de8ab8adad2cb1f5f29c1fc5f54c6c7fee1',
         'history.json':'97d2e30d1dc694a63d8ad3827d2a88a7554145f2709400cf4d6145eadff4f758',
         'training_plan.json':'d65a5005449350f38863ded2ba8449a358aa5a855ed65910ae87e9f400266a8f'}


def calibration(folder):
    for name,digest in FILES17.items():
        if sha256(folder/name)!=digest:raise ValueError('Dogrulanmis 17 kaydi farkli: '+name)
    cfg=read_json(folder/'protocol.json');history=read_json(folder/'history.json');plan=read_json(folder/'training_plan.json')
    if fingerprint(plan)!=cfg['training_plan_sha256'] or len(history)!=3:raise ValueError('17 ornek plani/blok sayisi farkli')
    for name,digest in cfg['code_hashes'].items():
        if Path(name).name!=name or sha256(Path(__file__).with_name(name))!=digest:raise ValueError('17 kaynak kodu degismis')
    for row in history:
        events=row['amp_attempt_events']
        if row['optimizer_attempts']!=32 or row['optimizer_updates']!=32 or len(events)!=32 or not all(
                e['optimizer_update_applied'] and e['unscaled_gradient_norm_finite'] and e['scale_before']==e['scale_after']==16 for e in events):
            raise ValueError('17 AMP kararlilik kaydi farkli')
    return cfg,plan


def prepare(stage_data,records,data,data_cfg,dev_sources):
    wanted=[r for r in records if r['split']=='train' or r['source_image'] in set(dev_sources)]
    if len([r for r in wanted if r['split']=='train'])!=5824 or len(set(dev_sources))!=32:raise ValueError('5824/32 kaynak plani gerekli')
    if (data/'protocol.json').exists() and read_json(data/'protocol.json')!=data_cfg:raise ValueError('18 veri protokolu degismis')
    write_json(data/'protocol.json',data_cfg);groups={'train':[],'val':[]};items={}
    for i,r in enumerate(wanted,1):
        import shutil
        if shutil.disk_usage(data).free<3*2**30:raise OSError('18 veri/checkpoint icin en az 3 GiB bos alan gerekli')
        item=p.materialize(stage_data,data,r,fingerprint(data_cfg));items[r['source_image']]=item
        groups[r['split']].append({'source_image':r['source_image'],'crops':[item]})
        if i==1 or i%50==0 or i==len(wanted):print(f'18 tam goruntu verisi: {i}/{len(wanted)}',flush=True)
    dev=[items[name] for name in dev_sources]
    if any(x['split']!='val' for x in dev):raise ValueError('Gelistirme kaynagi TRAIN egitim rolunde olamaz')
    write_json(data/'sample_index.json',{'protocol_sha256':fingerprint(data_cfg),'records':list(items.values())})
    return groups,dev


def configuration(c17,data_cfg,dev,upstream,env):
    from . import crop_experiment as engine
    keys=['seed','initial_weights_sha256','imgsz','physical_batch','effective_batch','optimizer','lr','weight_decay','betas',
          'gradient_clip_norm','amp','amp_init_scale','amp_growth_factor','amp_backoff_factor','amp_growth_interval','ema_decay','ema_tau','ignore_rule','runtime_versions']
    return {**{k:c17[k] for k in keys},'protocol_id':'hycado_rtdetr_full_image_fixed80_v1','rounds':80,'samples_per_round':5824,
        'lrf':.1,'minimum_update_fraction_per_round':.90,'development_diagnostic_samples':32,
        'source17_metadata_sha256':FILES17,'data_protocol_sha256':fingerprint(data_cfg),'development_plan_sha256':fingerprint(dev),
        'upstream_metadata_sha256':upstream,'initialization':'fresh official COCO, no pilot/crop checkpoint initialization',
        'selection':'80th round EMA; no early stopping or best-AP selection','augmentation':False,
        'sampling':'one complete source image per round; unchanged source_round seed42 order; no pilot stress oversampling',
        'official_val_rows_used_for_training':False,'official_VAL_previously_evaluated':True,'independent_new_test_claim':False,
        'gpu':env['gpu'],'driver':env['nvidia_smi'].split(',')[-1].strip(),
        'code_hashes':{n:sha256(Path(__file__).with_name(n)) for n in ['full_image_training.py','full_image_pilot.py',
             'crop_experiment.py','crop_training_v2.py','crop_loader.py','crop_labels.py','crop_criterion.py','common.py','stage1.py']},
        'training_engine_sha256':sha256(engine.__file__),'determinism':c17['determinism']}


def validate_history(history,cfg,schedule):
    if len(history)!=cfg['rounds']:raise ValueError('Sabit 80 tur tamamlanmadi')
    for i,h in enumerate(history):
        if (h['round']!=i+1 or h['training_samples']!=cfg['samples_per_round'] or
            h['source_round_order_sha256']!=schedule[i]['order_sha256'] or h['learning_rate']!=schedule[i]['learning_rate'] or
            h['optimizer_attempts']!=728 or h['optimizer_updates']/728<.90):raise ValueError('Tur kapsam/sira/LR/AMP farkli')
    return {'status':'COMPLETED','completed_rounds':80,'target_rounds':80,'training_samples':sum(x['training_samples'] for x in history),
        'optimizer_attempts':sum(x['optimizer_attempts'] for x in history),'optimizer_updates':sum(x['optimizer_updates'] for x in history),
        'official_val_rows_used_for_training':False,'pilot_checkpoint_used':False,'weights_selected':'80th round EMA','detection_AP_measured':False}


def bundle(output):
    import os,tempfile,zipfile
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=output,suffix='.zip.tmp');os.close(fd);target=output/'18_tam_egitim_rapor.zip'
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(output.glob('*.json'))+sorted(output.glob('*.csv')):z.write(path,path.name)
            for name in ['protocol.json','sample_index.json']:
                path=output/'data'/name
                if path.exists():z.write(path,'data/'+name)
            for name in ['console.log','optimizer_attempts.jsonl']:
                path=output/name
                if path.exists():
                    with path.open('rb') as f:
                        start=max(0,path.stat().st_size-100000);f.seek(start)
                        if start:f.readline()
                        z.writestr(name.replace('.log','_tail.log').replace('.jsonl','_tail.jsonl'),f.read())
        os.replace(tmp,target)
    finally:Path(tmp).unlink(missing_ok=True)
    return target


def backup(root,output):
    import shutil,os,tempfile
    target=root.parent/'HyCADO_Yedekleri/18_Tam_Egitim'
    paths=[root/'hycado_start/full_image_training.py']+[root/n for n in ['18_RTDETR_TAM_EGITIM.cmd','18_RAPOR_AL.cmd','18_ONCE_OKUYUN.md','SHA256SUMS_18.txt'] if (root/n).exists()]
    paths += [x for x in output.iterdir() if x.is_file() and x.suffix in ['.json','.csv','.zip']]
    if (output/'checkpoint_index.json').exists():
        latest=read_json(output/'checkpoint_index.json')['latest'];checkpoint=p.safe_path(output,latest['path'])
        if sha256(checkpoint)!=latest['sha256']:raise ValueError('Yedeklenecek son checkpoint bozuk')
        paths.append(checkpoint)
    for path in paths:
        digest=sha256(path);dest=target/digest[:12]/path.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists() or sha256(dest)!=digest:
            fd,tmp=tempfile.mkstemp(dir=dest.parent,suffix='.tmp');os.close(fd)
            try:
                shutil.copy2(path,tmp)
                if sha256(tmp)!=digest:raise ValueError('18 kucuk yedek farkli')
                os.replace(tmp,dest)
            finally:Path(tmp).unlink(missing_ok=True)


def run():
    from .__main__ import _Tee
    output=ROOT/'experiments'/OUT;output.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        folders=[(OUT,'.training.lock'),(p.OUT,'.training.lock'),('stage1_v1_seed42','.training.lock'),('rtdetr_crop_v1_seed42','.training.lock'),
            (p.size.OUT,'.analysis.lock'),(p.size.o.OUT,'.benchmark.lock'),(p.size.o.r.OUT,'.benchmark.lock'),
            (p.size.o.n.OUT,'.analysis.lock'),(p.size.o.r.val.OUT,'.evaluation.lock')]
        for lock in [ROOT/'.backup.lock']+[ROOT/'experiments'/f/n for f,n in folders]:stack.enter_context(training_lock(lock))
        with (output/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    print('18 GERCEK TAM GORUNTU RT-DETR EGITIMI:',utc_now(),flush=True)
                    write_json(output/'result.json',{'status':'RUNNING','target_rounds':80})
                    c17,plan17=calibration(ROOT/'experiments'/p.OUT)
                    stage,manifest,official,upstream=p.sources(ROOT)
                    if c17['upstream_metadata_sha256']!=upstream:raise ValueError('17 kaynak baglantilari degismis')
                    for name,digest in FILES17.items():upstream['experiments/'+p.OUT+'/'+name]=digest
                    env=system_info(include_torch=True);write_json(output/'environment.json',env)
                    if not env.get('cuda_available'):raise RuntimeError('CUDA gerekli; CPU egitimine otomatik gecilmedi')
                    if env['gpu']!=c17['gpu'] or env['nvidia_smi'].split(',')[-1].strip()!=c17['driver']:raise ValueError('Kalibrasyondaki GPU/surucu gerekli')
                    backup(ROOT,output)
                    records=p.catalog(stage,manifest,read_json(ROOT/'data/prepared/manifest.json'))
                    dev_sources=[x['source_image'] for x in plan17[0]['development']]
                    dc={'id':'hycado_whole_image_training_data_v1','source_manifest_sha256':sha256(stage/'manifest.json'),
                        'annotation_policy':p.ANNOTATION_POLICY,'source_rectangle':'whole image','train_sources':5824,
                        'diagnostic_development_sources':dev_sources,'png_compress_level':1,'code_sha256':sha256(__file__)}
                    import shutil
                    minimum=(3 if (output/'data/sample_index.json').exists() else 40)*2**30
                    if shutil.disk_usage(output).free<minimum:raise OSError('18 veri ve checkpoint icin bos disk alani yetersiz')
                    groups,dev=prepare(stage,records,output/'data',dc,dev_sources)
                    from . import crop_experiment as engine
                    import torch,cv2
                    from .crop_criterion import criterion_self_check
                    cfg=configuration(c17,dc,dev,upstream,env);schedule=engine.make_schedule(groups,cfg)
                    if (output/'protocol.json').exists() and read_json(output/'protocol.json')!=cfg:raise ValueError('18 protokol/kod/ortam degismis')
                    write_json(output/'protocol.json',cfg);write_json(output/'round_schedule.json',schedule);write_json(output/'development_plan.json',dev)
                    torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
                    torch.use_deterministic_algorithms(True,warn_only=True);write_json(output/'criterion_self_check.json',criterion_self_check('cuda'))
                    history=engine.train(output/'data',output,official,cfg,groups,schedule,dev)
                    result=validate_history(history,cfg,schedule)
                    for path,digest in upstream.items():
                        if sha256(ROOT/path)!=digest:raise ValueError('Onceki deney kayitlari degismis')
                    if sha256(official)!=p.OFFICIAL:raise ValueError('Resmi baslangic agirligi degismis')
                    result['checkpoint']=read_json(output/'checkpoint_index.json')['latest']
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();result={'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR','error':str(exc),
                        'memory':memory_snapshot(),'resume':'rerun 18; continue from own latest completed round; partial round repeats'}
                write_json(output/'result.json',result);write_json(output/'progress.json',result);print('SONUC:',result['status'],flush=True)
        report=bundle(output)
        try:backup(ROOT,output)
        except Exception as exc:print('18 RAPOR KUCUK YEDEGI ALINAMADI:',str(exc),flush=True)
        print('GONDERILECEK RAPOR:',report,flush=True)
        return result['status']=='COMPLETED'


def main():
    if '--report-only' in sys.argv:print(bundle(ROOT/'experiments'/OUT),flush=True);return 0
    return 0 if run() else 1


if __name__=='__main__':raise SystemExit(main())
