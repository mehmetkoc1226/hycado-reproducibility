"""Frozen full-image RT-DETR EMA80 VAL inference and three-way accuracy.

All predictions are committed before annotation parsing. No training, NMS on
the standalone set predictor, parameter search, or runtime/FPS claim.
"""
from __future__ import annotations
import contextlib
from pathlib import Path
import sys
import traceback
import numpy as np
from .common import ROOT,sha256,write_json,utc_now,system_info
from .stage1 import fingerprint,training_lock
from .crop_loader import safe_path,read_json
from . import runtime_benchmark as r,full_image_training as training
from . import score_ablation as scoring
from .dev_metrics import raw_annotation,ignore_integral,prepare_groundtruth,outside_ignore,xyxy_to_det

OUT='rtdetr_full_image_eval_v1_seed42'
P18='0137544e585fa48eee80b1bc1713c77f7ca10b75560876420642f9814cf4643f'
W18='0d7f046673037e83accccf8c430f6978799f59557a93f3be4a7ac2ea182eb57c'
FILES18={'protocol.json':'7cb8d96239318d3f3ad0f12a8fd7a6fc8cc0aea37402e42811dfee8efab55aec',
 'result.json':'a5f1b8bbd0654a7193e5f283f27e9e994a4b11a066e21bbb1fdac215d6128542',
 'history.json':'4f8d02d3f1b66b1de4da9ae08130b6be130c0a7bae856e115010a8715efd72ec',
 'round_schedule.json':'7f196e8f9b49b66e0c013603015a7197cf48a04c8f86c4dcb0930273418c6493',
 'checkpoint_index.json':'01b5fc0698486e82e09fbc5bfc4045bcf665ce5096f2645596559f760d236698',
 'development_plan.json':'62d79491d496e2380b6cb444878df7985a791b2a821a29810aaafa3125a658e5',
 'data/sample_index.json':'e6cbb0dc6e41293b7c03e58e06a0c568ec5213bbfc02558018dbb7f85342899c'}
VARIANTS=['baseline','rtdetr_full960','hybrid_a025']


def source(root):
    cfg12,base,_,_,official=r.source(root)
    folder=root/'experiments'/training.OUT
    for name,digest in FILES18.items():
        if sha256(folder/name)!=digest:raise ValueError('Tamamlanan 18 kaydi farkli: '+name)
    cfg=read_json(folder/'protocol.json');history=read_json(folder/'history.json');schedule=read_json(folder/'round_schedule.json')
    if fingerprint(cfg)!=P18:raise ValueError('18 egitim protokolu farkli')
    runtime_protocol=root/'experiments'/r.OUT/'protocol.json'
    if (sha256(runtime_protocol)!=cfg['upstream_metadata_sha256'][runtime_protocol.relative_to(root).as_posix()] or
        sha256(r.__file__)!=read_json(runtime_protocol)['code_sha256']):raise ValueError('Dogrulanmis RGB cikarim yardimcisi degismis')
    for name,digest in cfg['code_hashes'].items():
        if Path(name).name!=name or sha256(root/'hycado_start'/name)!=digest:raise ValueError('18 kaynak kodu degismis: '+name)
    training.validate_history(history,cfg,schedule)
    result=read_json(folder/'result.json');index=read_json(folder/'checkpoint_index.json')
    if (result.get('status')!='COMPLETED' or result.get('optimizer_updates')!=58220 or
        index['protocol_sha256']!=P18 or index['latest']['epoch']!=80 or index['latest']['sha256']!=W18 or result['checkpoint']!=index['latest']):
        raise ValueError('Dogrulanmis son EMA80 gerekli')
    weights=safe_path(folder,index['latest']['path'])
    if sha256(weights)!=W18:raise ValueError('18 son agirligi degismis')
    data_index=read_json(folder/'data/sample_index.json');items=data_index['records']
    train=[x for x in items if x['split']=='train'];dev=[x for x in items if x['split']=='val']
    if (data_index['protocol_sha256']!=cfg['data_protocol_sha256'] or len(train)!=5824 or len(dev)!=32 or
            len({x['source_image'] for x in items})!=5856 or any(x['crop_index']!=0 or not x['source_image'].startswith('images/train/') for x in items)):
        raise ValueError('18 egitim veri kapsami farkli')
    dev_plan=read_json(folder/'development_plan.json')
    if fingerprint(dev_plan)!=cfg['development_plan_sha256'] or {x['source_image'] for x in dev_plan}!={x['source_image'] for x in dev}:raise ValueError('18 gelistirme kimlikleri farkli')
    if {x['group'] for x in train}&{x['group'] for x in dev}:raise ValueError('18 egitim/gelistirme grup kesisimi')
    # Verify every stored parent order without platform-dependent cosine recomputation.
    from .crop_loader import source_round
    from .crop_experiment import order_fingerprint
    groups=[{'source_image':x['source_image'],'crops':[x]} for x in train]
    for i,row in enumerate(schedule):
        if order_fingerprint(source_round(groups,i,42))!=row['order_sha256']:raise ValueError('18 ornek sirasi farkli')
    immutable={str(folder/n):v for n,v in FILES18.items()}
    immutable.update({str(base/n):v for n,v in r.FILES12.items()});immutable[str(weights)]=W18
    return cfg12,base,weights,official,immutable


def decode(query,width,height):
    if query.shape!=(300,14) or query.dtype!=np.float32 or not np.isfinite(query).all():raise ValueError('RT-DETR ham 300x14 float32 gerekli')
    if np.any(query[:,4:]<0) or np.any(query[:,4:]>1):raise ValueError('Sigmoid sinif skorlari [0,1] olmali')
    rows=r.old.map_queries(query,{'source_crop_xyxy':[0,0,width,height]})
    rows,dropped=r.old.canonical_detections(rows,width,height)
    if dropped:raise ValueError('Decoder sonrasi gecersiz kutu')
    return rows  # no NMS, no score rescaling; at most 300 native query predictions


def load_model(weights,official,device='cuda'):
    import torch,cv2,gc
    from . import crop_training_v2 as engine
    engine.seed_all(42);torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    saved=torch.load(weights,map_location='cpu',weights_only=False)
    if (saved.get('schema')!='hycado_crop_experiment_checkpoint_v1' or saved.get('protocol_sha256')!=P18 or
            saved.get('round')!=80 or len(saved.get('history',[]))!=80 or sum(x['optimizer_updates'] for x in saved['history'])!=58220):
        raise ValueError('Tam goruntu checkpoint icerigi farkli')
    model=engine.build_model(official,'cpu');model.load_state_dict(saved['ema'],strict=True)
    del saved;gc.collect();model=model.float().eval().to(device)
    for parameter in model.parameters():parameter.requires_grad_(False)
    if not all(bool(torch.isfinite(x).all()) for x in model.state_dict().values() if x.is_floating_point()):raise ValueError('EMA agirliklari sonlu degil')
    return model


def predict_path(model,path,record):
    from PIL import Image
    with Image.open(path) as image:
        if image.size!=(record['width'],record['height']):raise ValueError('Goruntu boyutu farkli')
        pixels=np.array(image.convert('RGB'))
    return r.direct_queries(model,pixels)


def save_prediction(folder,query,record,signature):
    import tempfile,shutil
    rows=decode(query,record['width'],record['height']);folder.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(dir=folder.parent,prefix=folder.name+'_building_'))
    try:
        np.savez_compressed(temporary/'queries.npz',raw_queries=query)
        r.val.signed_json(temporary/'prediction.json',{'protocol_sha256':signature,'source_image':record['image'],
            'image_sha256':record['image_sha256'],'width':record['width'],'height':record['height'],
            'rtdetr_full960':rows,'raw_archive_sha256':sha256(temporary/'queries.npz')})
        if folder.exists():raise ValueError('Mevcut tamamlanmis tahmin uzerine yazilmadi')
        temporary.rename(folder)
    finally:
        if temporary.exists():shutil.rmtree(temporary)


def checked_prediction(folder,record,signature):
    p=r.val.checked_json(folder/'prediction.json',signature,record)
    if sha256(folder/'queries.npz')!=p['raw_archive_sha256']:raise ValueError('19 ham sorgu arsivi degismis')
    with np.load(folder/'queries.npz',allow_pickle=False) as data:query=data['raw_queries']
    if decode(query,record['width'],record['height'])!=p['rtdetr_full960']:raise ValueError('19 decoder/tahmin farkli')
    return p


def inference(records,prepared,output,signature,factory):
    model=None;index=[];immutable={}
    for i,rec in enumerate(records,1):
        path=safe_path(prepared,rec['image'])
        if sha256(path)!=rec['image_sha256']:raise ValueError('VAL goruntusu degismis')
        folder=output/'predictions'/Path(rec['image']).stem
        if not folder.exists():
            if model is None:model=factory()
            save_prediction(folder,predict_path(model,path,rec),rec,signature)
        checked_prediction(folder,rec,signature)
        index.append({'source_image':rec['image'],'path':(folder/'prediction.json').relative_to(output).as_posix(),
            'sha256':sha256(folder/'prediction.json'),'raw_archive_sha256':sha256(folder/'queries.npz')})
        immutable[str(path)]=rec['image_sha256']
        if i==1 or i%25==0 or i==len(records):
            print(f'19 RT-DETR VAL cikarimi: {i}/{len(records)}',flush=True)
            write_json(output/'progress.json',{'status':'INFERENCE','images_done':i,'images_total':len(records)})
    for path,digest in immutable.items():
        if sha256(path)!=digest:raise ValueError('Goruntu cikarim sirasinda degismis')
    write_json(output/'prediction_index.json',{'protocol_sha256':signature,'records':index})
    del model
    import gc;gc.collect()
    return index


def measure(records,prepared,base,output,signature,index):
    if len(index)!=len(records) or {x['source_image'] for x in index}!={x['image'] for x in records}:raise ValueError('Butun cikarimlar tamamlanmadan GT okunamaz')
    new={x['source_image']:x for x in index};old={x['source_image']:x for x in read_json(base/'prediction_index.json')['records']}
    parts=[];immutable={}
    for i,rec in enumerate(records,1):
        item=new[rec['image']];pred=safe_path(output,item['path']);reference=safe_path(base,old[rec['image']]['path'])
        anno=prepared/'visdrone_annotations/val'/(Path(rec['image']).stem+'.txt')
        for path,digest in [(pred,item['sha256']),(reference,old[rec['image']]['sha256']),(anno,rec['annotation_sha256'])]:
            if sha256(path)!=digest:raise ValueError('19 olcum kaynagi degismis')
            immutable[str(path)]=digest
        p=checked_prediction(pred.parent,rec,signature);prior=r.val.checked_json(reference,r.P12,rec)
        raw=raw_annotation(anno.read_text(encoding='utf-8-sig'));integral=ignore_integral(raw,rec['width'],rec['height']);gt=prepare_groundtruth(raw,integral)
        data={'gt_denominators':np.array([np.count_nonzero(gt[:,5]==c) for c in range(1,11)],dtype=np.int64)}
        data['presence']=(data['gt_denominators']>0).astype(np.int64)
        for name,rows in [('baseline',prior['baseline']),('rtdetr_full960',p['rtdetr_full960']),('hybrid_a025',prior['hybrid_a025'])]:
            det=outside_ignore(xyxy_to_det(rows),integral)[:500]
            data[name+'_scores']=det[:,4].copy();data[name+'_classes']=det[:,5].astype(np.int8)
            data[name+'_flags']=np.column_stack([scoring.fast_flags(det,gt,float(t)) for t in scoring.THRESHOLDS])
        parts.append(data)
        if i==1 or i%25==0 or i==len(records):
            print(f'19 ayni VAL metrikleri: {i}/{len(records)}',flush=True)
            write_json(output/'progress.json',{'status':'METRICS','images_done':i,'images_total':len(records)})
    metrics={name:scoring.aggregate(parts,name) for name in VARIANTS}
    frozen=read_json(base/'metrics.json')['variants']
    for name in ['baseline','hybrid_a025']:
        scoring.compare_metrics(metrics[name],frozen[name])
        if metrics[name]['counts_at_fixed_max500']!=frozen[name]['counts_at_fixed_max500']:raise ValueError('12 TP/FP kontrol sayilari farkli')
    for path,digest in immutable.items():
        if sha256(path)!=digest:raise ValueError('19 kaynagi olcum sirasinda degismis')
    keys=['AP','AP50','AP75','AR1','AR10','AR100','AR500']
    report={'images':len(records),'variants':metrics,'frozen12_baseline_and_hybrid_metrics_equal':True,
        'delta_hybrid_minus_full_rtdetr_pp':{k:metrics['hybrid_a025'][k]-metrics['rtdetr_full960'][k] for k in keys},
        'delta_full_rtdetr_minus_baseline_pp':{k:metrics['rtdetr_full960'][k]-metrics['baseline'][k] for k in keys},
        'official_binary_parity_verified':False,'metric_scope':'unchanged provisional reference-aligned Python metric; units percent',
        'independent_new_test_claim':False,'multiple_training_seed_uncertainty_measured':False,
        'standalone_query_cap':300,'standalone_NMS':False,'training_started':False,'parameter_search':False,
        'timing_scope':'not a controlled runtime benchmark; no FPS claim'}
    write_json(output/'metrics.json',report)
    import csv
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['variant']+keys);w.writeheader()
        for name in VARIANTS:w.writerow({'variant':name,**{k:metrics[name][k] for k in keys}})
    with (output/'per_class.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['variant']+list(metrics['baseline']['per_class'][0]));w.writeheader()
        for name in VARIANTS:
            for row in metrics[name]['per_class']:w.writerow({'variant':name,**row})
    return report


def bundle(output):
    import zipfile,os,tempfile
    output=Path(output);output.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=output,suffix='.zip.tmp');os.close(fd)
    target=output/'19_val_karsilastirma_rapor.zip'
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(output.glob('*.json'))+sorted(output.glob('*.csv'))+sorted((output/'predictions').glob('*/prediction.json')):z.write(path,path.relative_to(output).as_posix())
            if (output/'console.log').exists():
                with (output/'console.log').open('rb') as f:
                    f.seek(max(0,(output/'console.log').stat().st_size-80000));z.writestr('console_tail.log',f.read())
        os.replace(tmp,target)
    finally:Path(tmp).unlink(missing_ok=True)
    return target


def backup(root,output):
    import os,shutil,tempfile
    target=root.parent/'HyCADO_Yedekleri/19_VAL'
    paths=[root/'hycado_start/full_image_evaluation.py']+[root/n for n in ['19_RTDETR_VAL_KARSILASTIR.cmd','19_RAPOR_AL.cmd','19_ONCE_OKUYUN.md','SHA256SUMS_19.txt'] if (root/n).exists()]
    paths += [x for x in output.iterdir() if x.is_file() and x.suffix in ['.json','.csv','.zip']]
    for path in paths:
        digest=sha256(path);dest=target/digest[:12]/path.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists() or sha256(dest)!=digest:
            fd,tmp=tempfile.mkstemp(dir=dest.parent,suffix='.tmp');os.close(fd)
            try:
                shutil.copy2(path,tmp)
                if sha256(tmp)!=digest:raise ValueError('19 kucuk yedek farkli')
                os.replace(tmp,dest)
            finally:Path(tmp).unlink(missing_ok=True)


def run():
    from .__main__ import _Tee
    output=ROOT/'experiments'/OUT;output.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        for path in [ROOT/'.backup.lock']+[ROOT/'experiments'/folder/name for folder,name in
           [(OUT,'.evaluation.lock'),(training.OUT,'.training.lock'),('stage1_v1_seed42','.training.lock'),
            ('rtdetr_crop_v1_seed42','.training.lock'),(r.val.OUT,'.evaluation.lock')]]:stack.enter_context(training_lock(path))
        with (output/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    write_json(output/'result.json',{'status':'RUNNING'});print('19 UC YONTEM AYNI VAL:',utc_now(),flush=True)
                    cfg12,base,weights,official,immutable=source(ROOT)
                    # Verify/repair the final 18 checkpoint backup before any new GPU work.
                    training.backup(ROOT,ROOT/'experiments'/training.OUT)
                    env=system_info(include_torch=True);write_json(output/'environment.json',env)
                    if not env.get('cuda_available'):raise RuntimeError('19 VAL cikarimi icin CUDA gerekli')
                    cfg={'id':'hycado_full_rtdetr_val_v1','date':'2026-11-02','source18_protocol_sha256':P18,
                        'source18_weights_sha256':W18,'source18_metadata_sha256':FILES18,'source12_protocol_sha256':r.P12,
                        'source12_metadata_sha256':r.FILES12,'records':cfg12['records'],'runtime_versions':cfg12['runtime_versions'],
                        'inference':'full image RGB, cv2 square scale-fill 960, float32 input /255, FP16 autocast; batch1; EMA80',
                        'decoder':'unchanged crop-query decoder applied to full image; best class per query; strict raw confidence>.001; canonical score order',
                        'NMS':False,'query_cap':300,'score_rescaling':False,'metric':cfg12['metric'],
                        'gpu':env['gpu'],'driver':env['nvidia_smi'].split(',')[-1].strip(),
                        'frozen_comparators':['12 YOLO baseline','12 HyCADO alpha.25'],
                        'training_started':False,'parameter_search':False,'independent_new_test_claim':False,
                        'code_sha256':sha256(__file__)}
                    if (output/'protocol.json').exists() and read_json(output/'protocol.json')!=cfg:raise ValueError('19 protokolu degismis')
                    write_json(output/'protocol.json',cfg)
                    for name in ['metrics.json','metrics.csv','per_class.csv']:(output/name).unlink(missing_ok=True)
                    backup(ROOT,output)
                    def factory():
                        import torch
                        if not torch.cuda.is_available():raise RuntimeError('Yeni cikarim icin CUDA gerekli')
                        return load_model(weights,official)
                    index=inference(cfg12['records'],ROOT/'data/prepared',output,fingerprint(cfg),factory)
                    report=measure(cfg12['records'],ROOT/'data/prepared',base,output,fingerprint(cfg),index)
                    for path,digest in immutable.items():
                        if sha256(path)!=digest:raise ValueError('Onceki deney/agirlilar degismis')
                    result={'status':'COMPLETED','images':548,'training_started':False,'parameter_search':False,
                        'frozen12_baseline_and_hybrid_metrics_equal':True,'original_weights_and_results_unchanged':True,
                        'source18_weights_sha256':W18,'final18_checkpoint_backup_verified':True,'official_binary_parity_verified':False}
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();result={'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR','error':str(exc),
                        'resume':'rerun 19; verified complete per-image predictions reused; metrics recomputed'}
                write_json(output/'result.json',result);write_json(output/'progress.json',result);print('SONUC:',result['status'],flush=True)
        path=bundle(output)
        try:backup(ROOT,output)
        except Exception as exc:print('19 RAPOR YEDEGI ALINAMADI:',str(exc),flush=True)
        print('GONDERILECEK RAPOR:',path,flush=True)
        return result['status']=='COMPLETED'


def main():
    if '--report-only' in sys.argv:print(bundle(ROOT/'experiments'/OUT),flush=True);return 0
    return 0 if run() else 1


if __name__=='__main__':raise SystemExit(main())
