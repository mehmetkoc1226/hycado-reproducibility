"""Fixed-budget long crop RT-DETR run with source cycles and round recovery."""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib.metadata
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path

from .common import ROOT,sha256,system_info,utc_now,write_json
from .stage1 import fingerprint,training_lock,memory_snapshot,commit_checkpoint,resume_path
from .crop_loader import load_catalog,source_round,load_sample
from . import crop_training_v2 as calibrated


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def round_lr(cfg,index):
    if not 0<=index<cfg['rounds']:raise ValueError('Tur indeksi gecersiz')
    if cfg['rounds']==1:return cfg['lr']
    return cfg['lr']*(cfg['lrf']+(1-cfg['lrf'])*(1+math.cos(math.pi*index/(cfg['rounds']-1)))/2)


def order_fingerprint(items):
    return fingerprint([(item['source_image'],item['crop_index']) for item in items])


def make_schedule(groups,cfg):
    schedule=[]
    for index in range(cfg['rounds']):
        items=source_round(groups['train'],index,cfg['seed'])
        if len(items)!=cfg['samples_per_round']:raise ValueError('Tur kaynak sayisi sabit butceyle farkli')
        schedule.append({'round':index+1,'samples':len(items),'order_sha256':order_fingerprint(items),
                         'learning_rate':round_lr(cfg,index)})
    return schedule


def validate_calibration(directory,recipe):
    p=read_json(directory/'protocol.json');r=read_json(directory/'result.json')
    history=read_json(directory/'history.json');check=read_json(directory/'criterion_self_check.json')
    if (fingerprint(p)!=recipe['source_calibration_protocol_sha256'] or
            fingerprint(history)!=recipe['source_calibration_history_sha256'] or
            r.get('status')!='COMPLETED' or r.get('optimizer_updates')!=96 or r.get('optimizer_attempts')!=96 or
            r.get('official_val_used') is not False or check.get('status')!='PASSED' or check.get('device')!='cuda'):
        raise ValueError('Dogrulanmis 09 v2 kalibrasyonu gerekli')
    for row in history:
        events=row['amp_attempt_events']
        if len(events)!=32 or not all(e['optimizer_update_applied'] and e['unscaled_gradient_norm_finite'] and
                                     e['scale_before']==e['scale_after']==16. for e in events):
            raise ValueError('09 v2 AMP denetimi farkli')
    for name,digest in recipe['source_calibration_code_hashes'].items():
        if Path(name).name!=name or sha256(Path(__file__).with_name(name))!=digest:
            raise ValueError('Kalibrasyondaki kaynak kod degismis: '+name)
    return p


def configuration(recipe,catalog,weights):
    versions={name:importlib.metadata.version(name) for name in recipe['runtime_versions']}
    if (catalog!=recipe['catalog'] or sha256(weights)!=recipe['initial_weights_sha256'] or
            calibrated.upstream_hashes()!=recipe['upstream_hashes'] or versions!=recipe['runtime_versions']):
        raise ValueError('Veri, COCO agirligi veya sabit ortam kalibrasyonla farkli')
    return {**recipe,'protocol_id':recipe['id'],'code_hashes':
            {n:sha256(Path(__file__).with_name(n)) for n in
             ['crop_experiment.py','crop_loader.py','crop_criterion.py','crop_training_v2.py','stage1.py','common.py']},
            'recipe_sha256':fingerprint(recipe),'runtime_versions':versions,
            'checkpoint_schema':'hycado_crop_experiment_checkpoint_v1',
            'determinism':'seed/RNG saved; CUDA grid_sample backward may remain nondeterministic',
            'diagnostic':'32 fixed development crops, EMA eval loss only; not AP and not a selection criterion'}


def recover_journal(path,offset,digest=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        if offset:raise ValueError('Kayitli optimizer gunlugu kaybolmus')
        path.touch()
    with path.open('r+b') as stream:
        stream.seek(0,os.SEEK_END)
        if stream.tell()<offset:raise ValueError('Kayitli optimizer gunlugu kisalmis')
        stream.truncate(offset)
    if digest is not None and sha256(path)!=digest:raise ValueError('Tamamlanmis optimizer gunlugu degismis')


def append_event(stream,event):
    stream.write((json.dumps(event,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode('utf-8'))


def write_history(output,history):
    write_json(output/'history.json',history)
    fields=['round','learning_rate','training_samples','optimizer_attempts','optimizer_updates','amp_skipped_updates',
            'mean_training_loss','mean_ema_development_loss','train_seconds','development_seconds',
            'amp_scale_start','amp_scale_end','peak_cuda_reserved_gib']
    temporary=output/'round_results.csv.tmp'
    with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(history)
    os.replace(temporary,output/'round_results.csv')


def finite_states(model,ema,optimizer):
    import torch
    tensors=list(model.state_dict().values())+list(ema.ema.state_dict().values())
    tensors+=[x for state in optimizer.state.values() for x in state.values() if torch.is_tensor(x)]
    if not all(bool(torch.isfinite(x.detach()).all()) for x in tensors):
        raise FloatingPointError('Model/EMA/optimizer durumunda sonlu olmayan deger')


def train(data,output,weights,cfg,groups,schedule,dev_items,*,device='cuda'):
    import torch
    from ultralytics.utils.torch_utils import ModelEMA
    from .crop_criterion import IgnoreAwareRTDETRLoss,forward_loss
    digest=fingerprint(cfg);checkpoint=resume_path(output,digest)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False) if checkpoint else None
    history=saved['history'] if saved else []
    if saved and (saved['schema']!='hycado_crop_experiment_checkpoint_v1' or saved['protocol_sha256']!=digest or
            saved['schedule_sha256']!=fingerprint(schedule) or saved['development_plan_sha256']!=fingerprint(dev_items) or
            saved['scheduler']['next_round_index']!=len(history) or
            saved['round']!=len(history) or [h['round'] for h in history]!=list(range(1,len(history)+1))):
        raise ValueError('Checkpoint tur/protokol/veri plani kimligi farkli')
    journal=output/'optimizer_attempts.jsonl'
    recover_journal(journal,saved['journal_offset'] if saved else 0,saved['journal_sha256'] if saved else None)
    if len(history)>=cfg['rounds']:
        write_history(output,history)
        return history
    calibrated.seed_all(cfg['seed']);model=calibrated.build_model(weights,device)
    write_json(output/'model_identity.json',{'model_class':type(model).__name__,
        'parameters':sum(p.numel() for p in model.parameters()),'head_nc':getattr(model,'nc',None),
        'official_initial_weights_sha256':cfg.get('initial_weights_sha256'),
        'calibration_checkpoint_used':False,'continuing_own_round_checkpoint':checkpoint is not None})
    optimizer=calibrated.make_optimizer(model,cfg);scaler=calibrated.make_scaler(cfg,device)
    ema=ModelEMA(model,decay=cfg['ema_decay'],tau=cfg['ema_tau']);criterion=IgnoreAwareRTDETRLoss(nc=10)
    if saved:
        model.load_state_dict(saved['model'],strict=True);ema.ema.load_state_dict(saved['ema'],strict=True)
        ema.updates=saved['ema_updates'];optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        calibrated.restore_rng(saved['rng']);del saved
    gc.collect();global_attempts=sum(h['optimizer_attempts'] for h in history)
    with journal.open('ab') as events:
        for index in range(len(history),cfg['rounds']):
            items=source_round(groups['train'],index,cfg['seed'])
            expected=schedule[index]
            if len(items)!=expected['samples'] or order_fingerprint(items)!=expected['order_sha256']:
                raise ValueError('Tur crop sirasi sabit planla farkli')
            if shutil.disk_usage(output).free<2*2**30:raise OSError('Checkpoint icin en az 2 GiB bos disk alani gerekli')
            learning_rate=round_lr(cfg,index)
            for group in optimizer.param_groups:group['lr']=learning_rate
            model.train();calibrated.synchronize(device);started=time.perf_counter()
            if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
            attempts=updates=0;loss_sum=0.;audit=Counter();states=Counter();scale_start=scaler.get_scale()
            max_norm=0.;min_scale=scale_start
            for first in range(0,len(items),cfg['effective_batch']):
                window=items[first:first+cfg['effective_batch']];optimizer.zero_grad(set_to_none=True)
                for item in window:
                    batch=load_sample(data,item,cfg['imgsz'],device)
                    with torch.autocast(device_type=str(device).split(':')[0],dtype=torch.float16,
                                        enabled=cfg['amp'] and str(device).startswith('cuda')):
                        loss,parts=forward_loss(model,batch,criterion)
                    if not torch.isfinite(loss):raise FloatingPointError('Egitim loss sonlu degil')
                    loss_sum+=float(loss.detach());audit.update(criterion.audit);states[item['supervision_state']]+=1
                    scaler.scale(loss/len(window)).backward();del batch,loss,parts
                event=calibrated.scaled_optimizer_step(model,optimizer,scaler,cfg['gradient_clip_norm'])
                attempts+=1;global_attempts+=1
                if event['optimizer_update_applied']:updates+=1;ema.update(model)
                else:print(f'AMP atlama: tur={index+1}, adim={attempts}, olcek={event["scale_before"]}->{event["scale_after"]}',flush=True)
                if event['unscaled_gradient_norm'] is not None:max_norm=max(max_norm,event['unscaled_gradient_norm'])
                min_scale=min(min_scale,event['scale_after'])
                event.update(round=index+1,attempt_in_round=attempts,global_attempt=global_attempts,
                             sample_end=first+len(window),learning_rate=learning_rate)
                append_event(events,event)
                completed=first+len(window)
                if completed%64==0 or completed==len(items):
                    calibrated.synchronize(device);elapsed=time.perf_counter()-started
                    remaining_samples=(cfg['rounds']-index-1)*len(items)+len(items)-completed
                    eta_hours=remaining_samples*elapsed/completed/3600
                    print(f'Tur {index+1}/{cfg["rounds"]}: {completed}/{len(items)}; guncelleme={updates}/{attempts}; AMP={scaler.get_scale()}; kalan egitim ~{eta_hours:.1f} saat',flush=True)
                    events.flush()
                    write_json(output/'progress.json',{'status':'TRAINING','round':index+1,'target_rounds':cfg['rounds'],
                        'last_completed_round':len(history),'samples_in_round':completed,'samples_per_round':len(items),
                        'optimizer_updates_in_round':updates,'optimizer_attempts_in_round':attempts,
                        'amp_scale':scaler.get_scale(),'learning_rate':learning_rate,
                        'training_only_remaining_hours_estimate':eta_hours,'memory':memory_snapshot()})
            optimizer.zero_grad(set_to_none=True);calibrated.synchronize(device);train_seconds=time.perf_counter()-started
            if not attempts or updates/attempts<cfg['minimum_update_fraction_per_round']:
                raise FloatingPointError('Turde basarili AMP guncellemesi yuzde 90 altinda; tur tamamlandi sayilmadi')
            ema.ema.eval();dev_start=time.perf_counter();dev_loss=0.
            with torch.no_grad():
                for item in dev_items:
                    batch=load_sample(data,item,cfg['imgsz'],device)
                    with torch.autocast(device_type=str(device).split(':')[0],dtype=torch.float16,
                                        enabled=cfg['amp'] and str(device).startswith('cuda')):
                        loss,parts=forward_loss(ema.ema,batch,criterion)
                    if not torch.isfinite(loss):raise FloatingPointError('EMA tanisal gelistirme loss sonlu degil')
                    dev_loss+=float(loss);del batch,loss,parts
            calibrated.synchronize(device);dev_seconds=time.perf_counter()-dev_start
            finite_states(model,ema,optimizer)
            events.flush();os.fsync(events.fileno());journal_offset=events.tell();journal_hash=sha256(journal)
            record={'round':index+1,'learning_rate':learning_rate,'training_samples':len(items),
                    'source_round_order_sha256':expected['order_sha256'],'optimizer_attempts':attempts,
                    'optimizer_updates':updates,'amp_skipped_updates':attempts-updates,
                    'mean_training_loss':loss_sum/len(items),'mean_ema_development_loss':dev_loss/len(dev_items),
                    'diagnostic_development_samples':len(dev_items),'train_seconds':train_seconds,'development_seconds':dev_seconds,
                    'amp_scale_start':scale_start,'amp_scale_end':scaler.get_scale(),'minimum_amp_scale':min_scale,
                    'maximum_finite_unscaled_gradient_norm':max_norm,'ignore_training_audit':dict(audit),
                    'sample_supervision_states':dict(states),'memory':memory_snapshot(),
                    'peak_cuda_reserved_gib':torch.cuda.max_memory_reserved()/2**30 if str(device).startswith('cuda') else None,
                    'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if str(device).startswith('cuda') else None}
            next_history=history+[record]
            payload={'schema':'hycado_crop_experiment_checkpoint_v1','protocol_sha256':digest,
                     'schedule_sha256':fingerprint(schedule),'development_plan_sha256':fingerprint(dev_items),
                     'round':index+1,'history':next_history,'model':calibrated.cpu_tree(model.state_dict()),
                     'ema':calibrated.cpu_tree(ema.ema.state_dict()),'ema_updates':ema.updates,
                     'optimizer':calibrated.cpu_tree(optimizer.state_dict()),'scaler':scaler.state_dict(),
                     'rng':calibrated.rng_state(),'journal_offset':journal_offset,'journal_sha256':journal_hash,
                     'scheduler':{'next_round_index':index+1,'kind':'fixed cosine per source round'}}
            record['checkpoint_payload_memory']=memory_snapshot()
            write_json(output/'progress.json',{'status':'SAVING_CHECKPOINT','round':index+1,
                                               'last_completed_round':len(history),'target_rounds':cfg['rounds']})
            item=commit_checkpoint(output,index+1,lambda path:torch.save(payload,path),digest)
            history=next_history;del payload;gc.collect();write_history(output,history)
            write_json(output/'progress.json',{'status':'ROUND_COMPLETED','last_completed_round':len(history),
                                               'target_rounds':cfg['rounds'],'checkpoint':item})
            print(f'CHECKPOINT: tur {index+1}/{cfg["rounds"]}; {output/item["path"]}',flush=True)
    return history


def make_bundle(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/'10_rtdetr_crop_egitim_rapor.zip'
    fd,name=tempfile.mkstemp(prefix='10_report_',suffix='.zip.tmp',dir=output);os.close(fd);temporary=Path(name)
    names=['protocol.json','environment.json','model_identity.json','catalog_summary.json','round_schedule.json','development_plan_summary.json',
           'progress.json','result.json','history.json','round_results.csv','checkpoint_index.json','criterion_self_check.json']
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
            for name in names:
                path=output/name
                if path.exists():z.writestr(name,path.read_bytes())
            for name,limit,alias in [('console.log',150000,'console_tail.log'),
                                     ('optimizer_attempts.jsonl',60000,'optimizer_attempts_tail.jsonl')]:
                path=output/name
                if path.exists():
                    with path.open('rb') as stream:
                        start=max(0,path.stat().st_size-limit);stream.seek(start)
                        if start:stream.readline()
                        tail=stream.read()
                        if alias.endswith('.jsonl') and tail and not tail.endswith(b'\n'):
                            tail=tail.rsplit(b'\n',1)[0]+b'\n' if b'\n' in tail else b''
                        z.writestr(alias,tail)
        os.replace(temporary,target)
    finally:temporary.unlink(missing_ok=True)
    return target


def run():
    from .__main__ import _Tee
    output=ROOT/'experiments/rtdetr_crop_v1_seed42';data=ROOT/'experiments/crop_dataset_v1_seed42'
    output.mkdir(parents=True,exist_ok=True)
    with training_lock(output/'.training.lock'):
        with (output/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    import cv2
                    import torch
                    print('10 RT-DETR CROP UZUN EGITIM:',utc_now(),flush=True)
                    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU gerekli; CPU egitimi baslatilmadi')
                    cv2.setNumThreads(2);torch.set_num_threads(2)
                    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
                    torch.use_deterministic_algorithms(True,warn_only=True)
                    recipe=read_json(ROOT/'configs/10_RTDETR_Crop_Egitim_v1.json')
                    validate_calibration(ROOT/'experiments/rtdetr_crop_calibration_v2_seed42',recipe)
                    weights=ROOT/'work/fresh_official_pretrained/rtdetr-l.pt'
                    if sha256(weights)!=read_json(weights.with_suffix('.provenance.json'))['sha256']:
                        raise ValueError('Resmi COCO kaynak kaydi farkli')
                    with training_lock(data/'.crop_data.lock'):
                        groups,catalog=load_catalog(data);cfg=configuration(recipe,catalog,weights);digest=fingerprint(cfg)
                        schedule=make_schedule(groups,cfg);dev=source_round(groups['val'],0,cfg['seed'])[:cfg['development_diagnostic_samples']]
                        if len(dev)!=cfg['development_diagnostic_samples']:raise ValueError('Tanisal gelistirme verisi eksik')
                        if (output/'protocol.json').exists() and fingerprint(read_json(output/'protocol.json'))!=digest:
                            raise ValueError('10 kodu/protokolu/verisi degismis; mevcut checkpoint ile karistirilmadi')
                        for name,value in [('round_schedule.json',schedule),('development_plan.json',dev)]:
                            if (output/name).exists() and fingerprint(read_json(output/name))!=fingerprint(value):
                                raise ValueError('Sabit ornek plani degismis: '+name)
                            write_json(output/name,value)
                        write_json(output/'protocol.json',cfg);write_json(output/'catalog_summary.json',catalog)
                        write_json(output/'environment.json',system_info(include_torch=True))
                        from .crop_criterion import criterion_self_check
                        write_json(output/'criterion_self_check.json',criterion_self_check('cuda'))
                        write_json(output/'development_plan_summary.json',{'sha256':fingerprint(dev),'samples':len(dev),
                                                                         'diagnostic_loss_only':True,'AP_measured':False})
                        write_json(output/'result.json',{'status':'RUNNING','target_rounds':cfg['rounds'],'started_utc':utc_now()})
                        history=train(data,output,weights,cfg,groups,schedule,dev)
                        result={'status':'COMPLETED','completed_rounds':len(history),'target_rounds':cfg['rounds'],
                                'training_samples':sum(h['training_samples'] for h in history),
                                'optimizer_attempts':sum(h['optimizer_attempts'] for h in history),
                                'optimizer_updates':sum(h['optimizer_updates'] for h in history),
                                'selected_checkpoint':'fixed final round EMA','detection_AP_measured':False,
                                'official_val_used':False,'pilot_checkpoint_used':False}
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();result={'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',
                        'error_type':type(exc).__name__,'error':str(exc),'resume':'Rerun 10; partial round is replayed from latest committed round'}
                write_json(output/'result.json',result);write_json(output/'progress.json',result)
                print('SONUC:',result['status'],flush=True)
        print('GONDERILECEK RAPOR:',make_bundle(output),flush=True)
        return result['status']=='COMPLETED'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report-only',action='store_true');args=p.parse_args()
    if args.report_only:
        print(make_bundle(ROOT/'experiments/rtdetr_crop_v1_seed42'),flush=True);return 0
    return 0 if run() else 1


if __name__=='__main__':raise SystemExit(main())
