"""Independent full-image RT-DETR pilot, initialized from official COCO.

Reuses the unchanged, previously checked optimizer/EMA/recovery engine and
ignore-aware loader. Its 'crop' rectangle is exactly the whole source image.
No existing experimental checkpoint initializes this separate pilot.
"""
from __future__ import annotations
from collections import Counter
import contextlib
import importlib.metadata
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import traceback
from .common import ROOT,sha256,write_json,utc_now,system_info
from .stage1 import fingerprint,training_lock,memory_snapshot
from .crop_loader import safe_path,read_json
from .crop_labels import ANNOTATION_POLICY,canonical_source_targets,crop_annotations,validate_annotation
from . import size_diagnostics as size

OUT='rtdetr_full_image_pilot_v1_seed42'
OFFICIAL='6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f'
FILES16={'protocol.json':'b5425d1a8385ad1ee813eebefc6f9a8a5716d72ca125c007bdd0eaa40d6a75bc',
         'result.json':'777ffadebe066b067abcb2ecd7ed6b355836aa834ff9c51255ed45e878aa65e4',
         'summary.json':'add052bc7c6fc88e3b5a306376e2a4382dd41bd268201cab456ee4986d8af9e9',
         'cache_index.json':'c4a8713c17a16e9841dc8d3e86ecf60fc03f384f3fcbd528a7ac0f36fc95800a'}


def sources(root):
    _,_,upstream=size.source(root)
    folder=root/'experiments'/size.OUT
    for name,digest in FILES16.items():
        if sha256(folder/name)!=digest:raise ValueError('Tamamlanan 16 kaydi farkli: '+name)
        upstream[(folder/name).relative_to(root).as_posix()]=digest
    c16=read_json(folder/'protocol.json');res=read_json(folder/'result.json')
    if (c16['code_sha256']!=sha256(size.__file__) or res.get('status')!='COMPLETED' or
            res.get('images')!=548 or res.get('original12_TP_FP_counts_equal') is not True):
        raise ValueError('Dogrulanmis 16 kodu/sonucu gerekli')
    stage=root/'experiments/stage1_v1_seed42';p1=read_json(stage/'protocol.json')
    manifest=read_json(stage/'data/manifest.json');size.o.r.proposals.validate_manifest(manifest)
    if (fingerprint(p1)!=size.o.r.val.P1 or sha256(stage/'data/manifest.json')!=p1['dataset_manifest_sha256'] or
            manifest['source_manifest_sha256']!=sha256(root/'data/prepared/manifest.json')):
        raise ValueError('Sabit 5824/647 TRAIN ayrimi degismis')
    official=root/'work/fresh_official_pretrained/rtdetr-l.pt'
    if sha256(official)!=OFFICIAL or read_json(official.with_suffix('.provenance.json'))['sha256']!=OFFICIAL:
        raise ValueError('Resmi COCO baslangic agirligi/kaynak kaydi farkli')
    upstream[(stage/'data/manifest.json').relative_to(root).as_posix()]=sha256(stage/'data/manifest.json')
    return stage/'data',manifest,official,upstream


def catalog(stage_data, manifest, prepared_manifest):
    originals={r['image']:r for r in prepared_manifest['records'] if r['split']=='train'}
    records=[]
    for i,r in enumerate(manifest['records'],1):
        original=originals[r['source_image']]
        if (r['source_split']!='train' or not r['source_image'].startswith('images/train/') or r['split'] not in ['train','val'] or
                original['image_sha256']!=r['image_sha256'] or original['annotation_sha256']!=r['annotation_sha256']):
            raise ValueError('Sabit TRAIN kaynak kimligi/rolu farkli')
        stem=Path(r['image']).stem;label=stage_data/'labels'/r['split']/(stem+'.txt');anno=stage_data/'annotations'/r['split']/(stem+'.txt')
        if sha256(label)!=r['label_sha256'] or sha256(anno)!=r['annotation_sha256']:raise ValueError('TRAIN etiket kaynagi degismis')
        records.append({**r,'width':original['width'],'height':original['height']})
        if i==1 or i%1000==0 or i==len(manifest['records']):print(f'17 TRAIN etiket denetimi: {i}/{len(manifest["records"])}',flush=True)
    return records


def select_plan(records,representative=240,stress_count=16,development=32):
    train=sorted((r for r in records if r['split']=='train'),key=lambda r:r['source_image'])
    dev=sorted((r for r in records if r['split']=='val'),key=lambda r:fingerprint({'17_development':r['source_image']}))[:development]
    if len(train)<3*representative or len(dev)!=development or len(train)<stress_count:raise ValueError('Kisa kosu icin yeterli kaynak yok')
    stress=sorted(train,key=lambda r:(-r['gt_count'],r['source_image']))[:stress_count]
    random.Random(42).shuffle(train)
    # These records are selected before pixels, loss or timing are inspected.
    return [{'phase':p+1,'representative':train[p*representative:(p+1)*representative],
             'stress':stress,'development':dev} for p in range(3)]


def materialize(stage_data, data, record, signature):
    from PIL import Image
    stem=Path(record['image']).stem;role=record['split'];folder=data/'items'/role/stem
    image=safe_path(stage_data,record['image']);label=stage_data/'labels'/role/(stem+'.txt');raw=stage_data/'annotations'/role/(stem+'.txt')
    expected={str(p):record[k] for p,k in [(image,'image_sha256'),(label,'label_sha256'),(raw,'annotation_sha256')]}
    if any(sha256(p)!=digest for p,digest in expected.items()):raise ValueError('Tam goruntu kaynak dosyasi degismis')
    bindings={'data_protocol_sha256':signature,'record':record}
    if (folder/'sample.json').exists():
        saved=read_json(folder/'sample.json')
        if saved['bindings']!=bindings or saved['content_sha256']!=fingerprint({k:v for k,v in saved.items() if k!='content_sha256'}):raise ValueError('17 veri kaydi kimligi farkli')
        item=saved['item']
        for field in ['image','annotation']:
            if sha256(safe_path(data,item[field]))!=item[field+'_sha256']:raise ValueError('17 kayitli veri dosyasi bozuk')
        return item
    if folder.exists():raise ValueError('17 tamamlanmamis hedef veri klasoru var')
    annotation=canonical_source_targets(raw.read_text(encoding='utf-8-sig'),record['width'],record['height'],label.read_text(encoding='utf-8'),source=str(raw))
    if len(annotation['targets'])!=record['gt_count']:raise ValueError('Tam goruntu GT sayisi farkli')
    ann=crop_annotations(annotation['targets'],annotation['ignore_regions'],[0,0,record['width'],record['height']])
    ann.update(protocol_sha256=signature,source_image=record['source_image'],split=role,group=record['group'],crop_index=0,sample_kind='whole_image')
    folder.parent.mkdir(parents=True,exist_ok=True);temporary=Path(tempfile.mkdtemp(dir=folder.parent,prefix=stem+'_building_'))
    try:
        with Image.open(image) as im:
            if im.size!=(record['width'],record['height']):raise ValueError('Tam goruntu boyutu farkli')
            im.convert('RGB').save(temporary/'full.png',compress_level=1)
        write_json(temporary/'full.json',ann)
        item={'protocol_sha256':signature,'source_image':record['source_image'],'split':role,'group':record['group'],'crop_index':0,
              'width':record['width'],'height':record['height'],'positive_targets':len(ann['positive_targets']),
              'supervision_state':ann['supervision_state'],'ignored_boxes_by_reason':dict(Counter(r['reason'] for r in ann['ignore_regions'])),
              'image':(folder/'full.png').relative_to(data).as_posix(),'annotation':(folder/'full.json').relative_to(data).as_posix(),
              'image_sha256':sha256(temporary/'full.png'),'annotation_sha256':sha256(temporary/'full.json')}
        body={'bindings':bindings,'item':item};body['content_sha256']=fingerprint(body);write_json(temporary/'sample.json',body)
        if any(sha256(p)!=digest for p,digest in expected.items()):raise ValueError('Veri yazilirken kaynak degismis')
        temporary.rename(folder)
        return item
    finally:
        if temporary.exists():shutil.rmtree(temporary)


def prepare_plan(stage_data,data,records,data_cfg):
    old=read_json(data/'protocol.json') if (data/'protocol.json').exists() else None
    if old is not None and old!=data_cfg:raise ValueError('17 veri protokolu degismis')
    write_json(data/'protocol.json',data_cfg);signature=fingerprint(data_cfg);selection=select_plan(records)
    unique={r['source_image']:r for phase in selection for kind in ['representative','stress','development'] for r in phase[kind]}
    items={}
    for i,(key,record) in enumerate(sorted(unique.items()),1):
        if shutil.disk_usage(data).free<3*2**30:raise OSError('17 PNG/checkpoint icin en az 3 GiB bos disk gerekli')
        items[key]=materialize(stage_data,data,record,signature)
        if i==1 or i%25==0 or i==len(unique):print(f'17 tam goruntu PNG/etiket: {i}/{len(unique)}',flush=True)
    plan=[{'phase':p['phase'],**{kind:[items[r['source_image']] for r in p[kind]] for kind in ['representative','stress','development']}} for p in selection]
    write_json(data/'sample_index.json',{'protocol_sha256':signature,'records':list(items.values())})
    return plan


def configuration(weights,plan,data_cfg,upstream,environment):
    from . import crop_training_v2 as engine
    from .crop_criterion import IGNORE_RULE
    return {'protocol_id':'hycado_rtdetr_full_image_pilot_v1','seed':42,'initial_weights_sha256':sha256(weights),
        'initialization':'fresh official COCO; independent of trained crop model and Stage1',
        'data_protocol_sha256':fingerprint(data_cfg),'training_plan_sha256':fingerprint(plan),'upstream_metadata_sha256':upstream,
        'imgsz':960,'physical_batch':1,'effective_batch':8,'phases':3,'warmup_timing_samples_per_phase':16,
        'optimizer':'AdamW','lr':1e-4,'weight_decay':1e-4,'betas':[.9,.999],'gradient_clip_norm':10.,
        'amp':True,'amp_init_scale':16.,'amp_growth_factor':2.,'amp_backoff_factor':.5,'amp_growth_interval':2000,
        'amp_min_update_fraction':.90,'ema':True,'ema_decay':.9999,'ema_tau':2000,'ignore_rule':IGNORE_RULE,
        'input':'whole source RGB image; lossless native PNG; unchanged cv2.INTER_LINEAR square scale-fill; no augmentation',
        'sampling':'720 source-shuffled representative frames; same 16 densest TRAIN frames per phase; same 32 internal development frames for EMA loss',
        'stress_representative_overlap':'allowed, explicitly pilot oversampling; not final epoch sampling',
        'empty_or_fully_ignored_full_frames':'retained in this full-frame pilot; no crop eligibility filter applied',
        'official_val_rows_used_for_training':False,'official_VAL_previously_evaluated':True,'independent_new_test_claim':False,
        'pilot_weights_used_for_final_experiment':False,'long_training_started':False,'detection_AP_measured':False,
        'candidate_long_budget':{'rounds':80,'samples_per_round':5824,'imgsz':960,'physical_batch':1,'effective_batch':8,
             'lr':1e-4,'lrf':.1,'scheduler':'same fixed round-cosine as 10','selection':'final-round EMA; no AP-based epoch selection',
             'frozen':False,'note':'requires pilot numerical/memory check; restart from official COCO for actual baseline'},
        'gpu':environment['gpu'],'driver':environment['nvidia_smi'].split(',')[-1].strip(),
        'runtime_versions':{name:importlib.metadata.version(name) for name in ['torch','torchvision','ultralytics','numpy','Pillow','opencv-python']},
        'upstream_code_sha256':engine.upstream_hashes(),
        'code_hashes':{name:sha256(Path(__file__).with_name(name)) for name in
              ['full_image_pilot.py','crop_training_v2.py','crop_loader.py','crop_labels.py','crop_criterion.py','stage1.py','common.py']},
        'determinism':'seed/RNG and full-phase checkpoints; CUDA grid_sample backward can remain nondeterministic'}


def projection(history,parents=5824):
    n=sum(h['representative_timed_samples'] for h in history)
    if n<=0:raise ValueError('Zamanlanan temsilci ornek yok')
    seconds=sum(h['mean_seconds_per_training_sample']*h['representative_timed_samples'] for h in history)/n
    return {'timed_representative_samples':n,'mean_training_seconds_per_sample':seconds,
       'candidate_80_round_training_only_hours':seconds*parents*80/3600,
       'training_sources':parents,'candidate_rounds':80,
       'maximum_observed_reserved_gib':max(h['peak_cuda_reserved_gib'] for h in history),
       'maximum_observed_allocated_gib':max(h['peak_cuda_allocated_gib'] for h in history),
       'limits':'projection, not promised duration; excludes dev evaluation, checkpoint I/O and later thermal/runtime changes',
       'all_TRAIN_shapes_fully_tested':False,'densest_16_TRAIN_frames_tested':True,
       'long_training_started':False,'pilot_checkpoint_must_not_initialize_baseline':True}


def small_backup(root,output,report=False):
    target=root.parent/'HyCADO_Yedekleri/17_Tam_Goruntu'
    files=[root/'hycado_start/full_image_pilot.py']
    files += [root/n for n in ['17_RTDETR_TAM_GORUNTU_KISA.cmd','17_RAPOR_AL.cmd','17_ONCE_OKUYUN.md','SHA256SUMS_17.txt'] if (root/n).exists()]
    if report:files += [p for p in output.glob('*') if p.is_file() and p.suffix in ['.json','.zip']]
    for p in files:
        digest=sha256(p);dest=target/digest[:12]/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists() or sha256(dest)!=digest:
            fd,tmp=tempfile.mkstemp(dir=dest.parent,suffix='.tmp');os.close(fd)
            try:
                shutil.copy2(p,tmp)
                if sha256(tmp)!=digest:raise ValueError('17 kucuk yedek kopyasi farkli')
                os.replace(tmp,dest)
            finally:Path(tmp).unlink(missing_ok=True)
    return str(target)


def bundle(output):
    import zipfile
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=output,suffix='.zip.tmp');os.close(fd);target=output/'17_tam_goruntu_rapor.zip'
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.glob('*.json')):
                if p.name!='budget_projection.json' or ((output/'result.json').exists() and read_json(output/'result.json').get('status')=='COMPLETED'):z.write(p,p.name)
            for name in ['protocol.json','sample_index.json']:
                p=output/'data'/name
                if p.exists():z.write(p,'data/'+name)
            if (output/'console.log').exists():
                with (output/'console.log').open('rb') as f:
                    f.seek(max(0,(output/'console.log').stat().st_size-100000));z.writestr('console_tail.log',f.read())
        os.replace(tmp,target)
    finally:Path(tmp).unlink(missing_ok=True)
    return target


def run():
    from .__main__ import _Tee
    output=ROOT/'experiments'/OUT;output.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        locks=[ROOT/'.backup.lock']+[ROOT/'experiments'/folder/name for folder,name in
            [(OUT,'.training.lock'),('stage1_v1_seed42','.training.lock'),('rtdetr_crop_v1_seed42','.training.lock'),
             (size.OUT,'.analysis.lock'),(size.o.OUT,'.benchmark.lock'),(size.o.r.OUT,'.benchmark.lock'),
             (size.o.n.OUT,'.analysis.lock'),(size.o.r.val.OUT,'.evaluation.lock')]]
        for lock in locks:stack.enter_context(training_lock(lock))
        with (output/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    print('17 BAGIMSIZ TAM GORUNTU RT-DETR KISA EGITIM:',utc_now(),flush=True)
                    write_json(output/'result.json',{'status':'RUNNING'});(output/'budget_projection.json').unlink(missing_ok=True)
                    stage_data,manifest,weights,upstream=sources(ROOT)
                    environment=system_info(include_torch=True);write_json(output/'environment.json',environment)
                    if not environment.get('cuda_available'):raise RuntimeError('CUDA gerekli; CPU egitimine otomatik gecilmedi')
                    records=catalog(stage_data,manifest,read_json(ROOT/'data/prepared/manifest.json'))
                    data_cfg={'id':'hycado_full_frame_pilot_data_v1','stage1_manifest_sha256':sha256(stage_data/'manifest.json'),
                        'annotation_policy':ANNOTATION_POLICY,'source_rectangle':'[0,0,source_width,source_height]; no proposals, crop selection or pixel masking',
                        'selection':'720 train shuffled seed42, top16 TRAIN positive-GT counts, 32 source-hash internal development',
                        'png_compress_level':1,'code_sha256':sha256(__file__)}
                    backup_path=small_backup(ROOT,output)
                    plan=prepare_plan(stage_data,output/'data',records,data_cfg)
                    write_json(output/'training_plan.json',plan)
                    import torch,cv2
                    from . import crop_training_v2 as engine
                    from .crop_criterion import criterion_self_check
                    torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
                    torch.use_deterministic_algorithms(True,warn_only=True)
                    cfg=configuration(weights,plan,data_cfg,upstream,environment)
                    if (output/'protocol.json').exists() and read_json(output/'protocol.json')!=cfg:raise ValueError('17 protokolu/ortami/plani degismis')
                    write_json(output/'protocol.json',cfg);write_json(output/'criterion_self_check.json',criterion_self_check('cuda'))
                    history=engine.train_core(output/'data',output,weights,cfg,plan)
                    quality=engine.validate_amp_history(history,.90)
                    if len(history)!=3 or sum(h['training_samples'] for h in history)!=768:raise ValueError('17 tam kisa kosu kapsami farkli')
                    for path,digest in upstream.items():
                        if sha256(ROOT/path)!=digest:raise ValueError('Onceki deney kayitlari degismis')
                    if sha256(weights)!=OFFICIAL:raise ValueError('Resmi baslangic agirligi degismis')
                    write_json(output/'budget_projection.json',projection(history))
                    result={'status':'COMPLETED','phases':3,'training_samples':768,'development_samples_per_phase':32,
                        'optimizer_attempts':quality['attempts'],'optimizer_updates':quality['updates'],'amp_quality':quality,
                        'initial_weights_sha256':OFFICIAL,'initialization':'official COCO, not trained crop model',
                        'official_val_rows_used_for_training':False,'long_training_started':False,'detection_AP_measured':False,
                        'pilot_weights_used_for_final_experiment':False,'small_backup_directory':backup_path}
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();result={'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',
                        'error_type':type(exc).__name__,'error':str(exc),'memory':memory_snapshot(),
                        'resume':'rerun 17; completed full phases reused; partial phase repeats; no resolution/batch fallback'}
                write_json(output/'result.json',result);write_json(output/'progress.json',result);print('SONUC:',result['status'],flush=True)
        path=bundle(output)
        try:small_backup(ROOT,output,True)
        except Exception as exc:print('17 RAPOR KUCUK YEDEGI ALINAMADI:',str(exc),flush=True)
        print('GONDERILECEK RAPOR:',path,flush=True)
        return result['status']=='COMPLETED'


def main():
    if '--report-only' in sys.argv:print(bundle(ROOT/'experiments'/OUT),flush=True);return 0
    return 0 if run() else 1


if __name__=='__main__':raise SystemExit(main())
