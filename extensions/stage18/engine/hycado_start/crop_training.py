"""Three short real-crop RT-DETR training blocks with ignore loss and recovery."""
from __future__ import annotations

import contextlib
import copy
import importlib.metadata
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path

from .common import ROOT,sha256,system_info,utc_now,write_json
from .stage1 import fingerprint,training_lock,memory_snapshot,commit_checkpoint,resume_path
from .crop_loader import SOURCE08_PROTOCOL,load_catalog,pilot_plan,load_sample


def configuration(catalog_info,weights):
    from .crop_criterion import IGNORE_RULE
    return {'protocol_id':'hycado_rtdetr_real_crop_calibration_v1','seed':42,
            'source08_protocol_sha256':SOURCE08_PROTOCOL,'catalog':catalog_info,
            'model':'rtdetr-l.pt','initial_weights_sha256':sha256(weights),
            'imgsz':960,'physical_batch':1,'effective_batch':8,
            'phases':3,'representative_samples_per_phase':240,'stress_samples_per_phase':16,
            'development_samples_per_phase':32,'warmup_timing_samples_per_phase':16,
            'optimizer':'AdamW','lr':1e-4,'weight_decay':1e-4,'betas':[.9,.999],
            'gradient_clip_norm':10.,'amp':True,'ema':True,'ema_decay':.9999,'ema_tau':2000,
            'gradient_accumulation':'sample mean; flush each group; checkpoint only after a complete phase',
            'determinism':'seed 42; deterministic flags warn_only; CUDA grid_sample backward is not guaranteed bitwise reproducible',
            'input':'RGB 0..1; native PNG -> cv2.INTER_LINEAR square scale-fill; no augmentation',
            'ignore_rule':IGNORE_RULE,'fully_ignored_no_positive':'excluded; other background crops retained',
            'long_run_sampling_candidate':'one eligible crop per source per round; deterministic cycle through each source crops',
            'long_training_budget_frozen':False,'pilot_weights_used_for_final_experiment':False,
            'evaluation':'32 fixed development crops per phase; eval-mode EMA loss only; not detection AP',
            'official_val_used':False,'old_experiment_artifacts_used':False,
            'runtime_versions':{n:importlib.metadata.version(n) for n in ['torch','torchvision','ultralytics','numpy','Pillow','opencv-python']},
            'upstream_hashes':upstream_hashes(),
            'code_hashes':{n:sha256(Path(__file__).with_name(n)) for n in ['crop_loader.py','crop_criterion.py','crop_training.py']}}


def upstream_hashes():
    import ultralytics
    base=Path(ultralytics.__file__).parent
    return {name:sha256(base/name) for name in ['models/utils/loss.py','models/utils/ops.py',
                                              'nn/tasks.py','nn/modules/head.py','utils/loss.py']}


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def rng_state():
    import numpy as np
    import torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def cpu_tree(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def build_model(weights,device):
    from ultralytics import RTDETR
    from ultralytics.nn.tasks import RTDETRDetectionModel
    pretrained=RTDETR(str(weights)).model
    model=RTDETRDetectionModel(copy.deepcopy(pretrained.yaml),nc=10,ch=3,verbose=False)
    model.load(pretrained,verbose=False)
    for parameter in model.parameters():parameter.requires_grad_(True)
    return model.to(device)


def make_optimizer(model,cfg):
    import torch
    decay=[];plain=[]
    for name,p in model.named_parameters():
        if p.requires_grad:(plain if p.ndim<2 or name.endswith('.bias') else decay).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':cfg['weight_decay']},
                              {'params':plain,'weight_decay':0.}],lr=cfg['lr'],betas=tuple(cfg['betas']),foreach=False)


def synchronize(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize()


def train_core(data,output,weights,cfg,plan,*,device='cuda'):
    import torch
    from ultralytics.utils.torch_utils import ModelEMA
    from .crop_criterion import IgnoreAwareRTDETRLoss,forward_loss
    digest=fingerprint(cfg);checkpoint=resume_path(output,digest)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False) if checkpoint else None
    if saved and (saved['protocol_sha256']!=digest or saved['plan_sha256']!=fingerprint(plan)):
        raise ValueError('Checkpoint ic protokolu veya veri plani farkli')
    history=saved['history'] if saved else []
    if saved and (saved['phase']!=len(history) or [h['phase'] for h in history]!=list(range(1,len(history)+1))):
        raise ValueError('Checkpoint blok sirasi farkli')
    if len(history)>=len(plan):
        write_json(output/'history.json',history)
        return history
    seed_all(cfg['seed']);model=build_model(weights,device);criterion=IgnoreAwareRTDETRLoss(nc=10)
    optimizer=make_optimizer(model,cfg)
    scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp'] and str(device).startswith('cuda'))
    ema=ModelEMA(model,decay=cfg['ema_decay'],tau=cfg['ema_tau'])
    if saved:
        model.load_state_dict(saved['model'],strict=True);ema.ema.load_state_dict(saved['ema'],strict=True)
        ema.updates=saved['ema_updates'];optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rng']);del saved
    for phase in plan[len(history):]:
        started=time.perf_counter();model.train();criterion.audit=Counter()
        if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        samples=phase['representative']+phase['stress'];n_rep=len(phase['representative'])
        times=[];loss_sum=0.;attempts=updates=0;audit=Counter();seen_states=Counter()
        for first in range(0,len(samples),cfg['effective_batch']):
            group=samples[first:first+cfg['effective_batch']];optimizer.zero_grad(set_to_none=True)
            for offset,item in enumerate(group):
                number=first+offset;synchronize(device);tick=time.perf_counter()
                batch=load_sample(data,item,cfg['imgsz'],device)
                with torch.autocast(device_type=str(device).split(':')[0],dtype=torch.float16,
                                    enabled=cfg['amp'] and str(device).startswith('cuda')):
                    loss,components=forward_loss(model,batch,criterion)
                if not torch.isfinite(loss):raise FloatingPointError('RT-DETR loss sonlu degil')
                loss_sum+=float(loss.detach());audit.update(criterion.audit);seen_states[item['supervision_state']]+=1
                scaler.scale(loss/len(group)).backward()
                if offset==len(group)-1:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
                    previous_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();attempts+=1
                    if scaler.get_scale()>=previous_scale:updates+=1;ema.update(model)
                synchronize(device)
                if cfg['warmup_timing_samples_per_phase']<=number<n_rep:times.append(time.perf_counter()-tick)
                del batch,loss,components
            if (first+len(group))%32==0 or first+len(group)==len(samples):
                print(f'RT-DETR blok {phase["phase"]}/{len(plan)}: {first+len(group)}/{len(samples)}; guncelleme={updates}/{attempts}',flush=True)
                write_json(output/'progress.json',{'status':'TRAINING','phase':phase['phase'],
                                                  'samples':first+len(group),'total':len(samples),
                                                  'optimizer_updates':updates,'optimizer_attempts':attempts,
                                                  'memory':memory_snapshot()})
        optimizer.zero_grad(set_to_none=True)
        # EMA diagnostic loss, eval-mode forward: no optimizer or BatchNorm updates.
        ema.ema.eval();dev_loss=0.
        with torch.no_grad():
            for item in phase['development']:
                batch=load_sample(data,item,cfg['imgsz'],device)
                with torch.autocast(device_type=str(device).split(':')[0],dtype=torch.float16,
                                    enabled=cfg['amp'] and str(device).startswith('cuda')):
                    loss,components=forward_loss(ema.ema,batch,criterion)
                if not torch.isfinite(loss):raise FloatingPointError('EMA gelistirme loss sonlu degil')
                dev_loss+=float(loss);del batch,loss,components
        synchronize(device)
        record={'phase':phase['phase'],'training_samples':len(samples),'representative_samples':n_rep,
                'stress_samples':len(phase['stress']),'development_samples':len(phase['development']),
                'optimizer_attempts':attempts,'optimizer_updates':updates,'amp_skipped_updates':attempts-updates,
                'mean_training_loss':loss_sum/len(samples),'mean_ema_development_loss':dev_loss/len(phase['development']),
                'representative_timed_samples':len(times),'mean_seconds_per_training_sample':statistics.mean(times),
                'median_seconds_per_training_sample':statistics.median(times),
                'p95_seconds_per_training_sample':sorted(times)[math.ceil(.95*len(times))-1],
                'ignore_training_audit':dict(audit),'sample_supervision_states':dict(seen_states),
                'phase_seconds_before_checkpoint':time.perf_counter()-started,'memory':memory_snapshot(),
                'peak_cuda_reserved_gib':torch.cuda.max_memory_reserved()/2**30 if str(device).startswith('cuda') else None,
                'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if str(device).startswith('cuda') else None}
        history.append(record)
        payload={'schema':'hycado_crop_calibration_checkpoint_v1','protocol_sha256':digest,'plan_sha256':fingerprint(plan),
                 'history':history,'model':cpu_tree(model.state_dict()),'ema':cpu_tree(ema.ema.state_dict()),
                 'ema_updates':ema.updates,'optimizer':cpu_tree(optimizer.state_dict()),'scaler':scaler.state_dict(),
                 'rng':rng_state(),'phase':phase['phase']}
        record['checkpoint_payload_memory']=memory_snapshot()
        commit_checkpoint(output,phase['phase'],lambda path:torch.save(payload,path),digest);del payload
        write_json(output/'history.json',history)
    return history


def budget_projection(history,catalog):
    seconds=sum(h['mean_seconds_per_training_sample']*h['representative_timed_samples'] for h in history)/sum(h['representative_timed_samples'] for h in history)
    parents=catalog['eligible_source_images']['train'];crops=catalog['total_crops']['train']-catalog['fully_ignored_crops_excluded'].get('train',0)
    return {'mean_seconds_per_representative_sample':seconds,
            'source_uniform_round_samples':parents,'all_eligible_crops_per_pass':crops,
            'source_uniform_train_only_hours':{str(n):parents*n*seconds/3600 for n in [20,40,80]},
            'all_crops_80_passes_train_only_hours':crops*80*seconds/3600,
            'limits':'Projection only: excludes full evaluation, checkpoint I/O, later augmentation and temperature/runtime variation; not a promised duration',
            'long_training_budget_frozen':False,'pilot_weights_must_not_initialize_final_experiment':True}


def make_bundle(output):
    target=output/'09_rtdetr_crop_rapor.zip'
    allowed=['protocol.json','environment.json','catalog_summary.json','training_plan_summary.json',
             'progress.json','result.json','checkpoint_index.json','criterion_self_check.json']
    # Only committed history may accompany an error, never stale success projections.
    if (output/'history.json').exists():allowed.append('history.json')
    if (output/'result.json').exists() and read_json(output/'result.json')['status']=='COMPLETED':allowed.append('budget_projection.json')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for name in allowed:
            if (output/name).exists():z.write(output/name,name)
        if (output/'console.log').exists():
            with (output/'console.log').open('rb') as stream:
                stream.seek(max(0,(output/'console.log').stat().st_size-100000));z.writestr('console_tail.log',stream.read())
    return target


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def run():
    from .__main__ import _Tee
    data=ROOT/'experiments/crop_dataset_v1_seed42';output=ROOT/'experiments/rtdetr_crop_calibration_v1_seed42'
    output.mkdir(parents=True,exist_ok=True)
    with training_lock(output/'.training.lock'):
        with (output/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    import cv2
                    import torch
                    import ultralytics
                    print('09 GERCEK CROP RT-DETR KISA EGITIM:',utc_now(),flush=True)
                    if ultralytics.__version__!='8.3.203' or not torch.__version__.startswith('2.6.0'):
                        raise RuntimeError('Sabit ortam gerekli: ultralytics 8.3.203, torch 2.6.0')
                    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU gorunmuyor; CPU egitimine otomatik gecilmedi')
                    cv2.setNumThreads(2);torch.set_num_threads(2)
                    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
                    torch.use_deterministic_algorithms(True,warn_only=True)
                    from .crop_criterion import criterion_self_check
                    write_json(output/'criterion_self_check.json',criterion_self_check('cuda'))
                    weights=ROOT/'work/fresh_official_pretrained/rtdetr-l.pt';provenance=weights.with_suffix('.provenance.json')
                    if not weights.exists() or not provenance.exists() or sha256(weights)!=read_json(provenance)['sha256']:
                        raise ValueError('Onceki kurulumda indirilen resmi COCO agirligi/kaynak kaydi gerekli')
                    with training_lock(data/'.crop_data.lock'):
                        groups,catalog=load_catalog(data);plan=pilot_plan(groups);cfg=configuration(catalog,weights);digest=fingerprint(cfg)
                        if (output/'protocol.json').exists() and fingerprint(read_json(output/'protocol.json'))!=digest:
                            raise ValueError('09 protokol/kod/veri kimligi degismis; kayitli egitimle karistirilmadi')
                        if (output/'training_plan.json').exists() and fingerprint(read_json(output/'training_plan.json'))!=fingerprint(plan):
                            raise ValueError('09 ornek plani degismis')
                        write_json(output/'protocol.json',cfg);write_json(output/'training_plan.json',plan)
                        write_json(output/'catalog_summary.json',catalog);write_json(output/'environment.json',system_info(include_torch=True))
                        write_json(output/'training_plan_summary.json',{'sha256':fingerprint(plan),'phases':len(plan),
                            'training_samples':sum(len(p['representative'])+len(p['stress']) for p in plan),
                            'representative_selection':'720 distinct source images, one deterministic crop each',
                            'stress_selection':'up to 8 highest target count, 4 largest native area, 4 most ignore boxes; fixed before training',
                            'development_selection':'same 32 source-disjoint development crops in each phase; loss diagnostic only'})
                        started=time.perf_counter();history=train_core(data,output,weights,cfg,plan)
                        updates=sum(h['optimizer_updates'] for h in history);attempts=sum(h['optimizer_attempts'] for h in history)
                        if not attempts or updates/attempts<.90:raise RuntimeError('AMP guncellemelerinin en az yuzde 90 i basarili olmali; uzun egitime gecilmedi')
                        write_json(output/'budget_projection.json',budget_projection(history,catalog))
                        result={'status':'COMPLETED','phases':len(history),'training_samples':sum(h['training_samples'] for h in history),
                                'optimizer_updates':updates,'optimizer_attempts':attempts,'ema_updates':updates,
                                'checkpoint_reload_next_invocation_supported':True,'official_val_used':False,
                                'detection_AP_measured':False,'long_training_started':False,
                                'elapsed_this_invocation_seconds':time.perf_counter()-started}
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();result={'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',
                        'error_type':type(exc).__name__,'error':str(exc),'resume':'Rerun 09; resume from latest committed complete phase'}
                write_json(output/'result.json',result);write_json(output/'progress.json',result)
                print('SONUC:',result['status'],flush=True)
        print('GONDERILECEK RAPOR:',make_bundle(output),flush=True)
        return result['status']=='COMPLETED'


def main():return 0 if run() else 1
if __name__=='__main__':raise SystemExit(main())
