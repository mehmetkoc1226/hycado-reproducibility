"""Recapture frozen crop inference with all queries and exact NMS provenance.

Original 11 predictions are read only. Trace causality applies to this replay;
an exact comparison records whether each replay equals the original output.
"""
from __future__ import annotations
from collections import Counter
import contextlib
import gc
import io
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
import numpy as np

from . import dev_evaluation as old
from .common import ROOT,sha256,write_json,utc_now,system_info
from .stage1 import fingerprint,training_lock
from .crop_loader import safe_path,read_json
from .dev_diagnostics import validate_source,SOURCE_PROTOCOL,detailed_matches,counts
from .dev_metrics import raw_annotation,ignore_integral,prepare_groundtruth,outside_ignore,xyxy_to_det,overlaps


def decode_with_ids(raw,crop):
    rows=old.map_queries(raw,crop)
    a=np.asarray(raw,dtype=np.float64)
    lo=np.clip(a[:,:2]-a[:,2:4]/2,0,1);hi=np.clip(a[:,:2]+a[:,2:4]/2,0,1)
    good=(a[:,4:].max(axis=1)>.001)&np.all(hi>lo,axis=1)
    ids=np.flatnonzero(good).astype(np.int32)
    if len(ids)!=len(rows):raise ValueError('Query kimligi / 11 decoder farkli')
    return rows,ids


def trace_nms(rows,origins,width,height):
    """Same fixed global classwise NMS as 11, retaining original candidate IDs.

    statuses: 1 selected, 2 suppressed, 3 discarded by max_det=1000.
    Every suppression records the selected candidate responsible for it.
    """
    a=np.asarray(rows,dtype=np.float64).reshape(-1,6)
    origins=np.asarray(origins,dtype=np.int32).reshape(-1,3)
    if len(a)!=len(origins):raise ValueError('Aday/origin sayisi farkli')
    clean,dropped=old.canonical_detections(rows,width,height)
    order=np.array(sorted(range(len(a)),key=lambda i:(-a[i,4],a[i,5],*a[i,:4],i)),dtype=np.int64)
    if dropped or a[order].tolist()!=clean:raise ValueError('Aday kutulari zaten canonical olmali')
    status=np.zeros(len(a),dtype=np.int8);by=np.full(len(a),-1,dtype=np.int64)
    area=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]);remaining=order;selected=[]
    while len(remaining) and len(selected)<1000:
        i=int(remaining[0]);selected.append(i);status[i]=1;other=remaining[1:]
        inter=np.maximum(0,np.minimum(a[i,2:4],a[other,2:4])-np.maximum(a[i,:2],a[other,:2])).prod(axis=1)
        union=area[i]+area[other]-inter
        iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
        suppress=(a[other,5]==a[i,5])&(iou>.7)
        status[other[suppress]]=2;by[other[suppress]]=i
        remaining=other[~suppress]
    status[remaining]=3
    selected=np.asarray(selected,dtype=np.int64)
    # Independent unchanged implementation checks both arithmetic and tie order.
    if a[selected].tolist()!=old.classwise_nms(rows,width,height):
        raise ValueError('Iz kayitli NMS, sabit 11 sonucuyla ayni degil')
    summary=Counter()
    for kind,label in [(0,'baseline'),(1,'crop')]:
        for code,state in [(1,'selected'),(2,'suppressed'),(3,'max1000_discarded')]:
            summary[label+'_'+state]=int(np.count_nonzero((origins[:,0]==kind)&(status==code)))
    for d in np.flatnonzero(status==2):
        left='baseline' if origins[d,0]==0 else 'crop'
        right='baseline' if origins[by[d],0]==0 else 'crop'
        summary[left+'_suppressed_by_'+right]+=1
    return a,origins,status,by,selected,dict(summary)


def load_model(weights,official,device):
    import torch
    import cv2
    from .crop_training_v2 import build_model,seed_all
    seed_all(42);torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    saved=torch.load(weights,map_location='cpu',weights_only=False)
    if (saved.get('schema')!='hycado_crop_experiment_checkpoint_v1' or saved.get('protocol_sha256')!=old.P10 or
            saved.get('round')!=80 or len(saved.get('history',[]))!=80):
        raise ValueError('80. tur checkpoint icerigi farkli')
    model=build_model(official,'cpu');model.load_state_dict(saved['ema'],strict=True)
    del saved;gc.collect();model=model.float().eval().to(device)
    for p in model.parameters():p.requires_grad_(False)
    if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values() if v.is_floating_point()):
        raise ValueError('EMA degerleri sonlu degil')
    return model


def predict_queries(model,path,crop,device='cuda'):
    import torch
    import cv2
    from PIL import Image
    pixels=Path(path).read_bytes()
    import hashlib
    if hashlib.sha256(pixels).hexdigest()!=crop['image_sha256']:raise ValueError('Crop PNG degismis')
    with Image.open(io.BytesIO(pixels)) as im:
        if im.size!=(crop['width'],crop['height']):raise ValueError('Crop boyutu farkli')
        rgb=np.array(im.convert('RGB'))
    resized=cv2.resize(rgb,(960,960),interpolation=cv2.INTER_LINEAR)
    x=torch.from_numpy(resized.transpose(2,0,1).copy()).unsqueeze(0).to(device,dtype=torch.float32)/255.
    with torch.inference_mode(),torch.autocast(str(device).split(':')[0],dtype=torch.float16,enabled=str(device).startswith('cuda')):
        output=model.predict(x)
    prediction=output[0] if isinstance(output,(tuple,list)) else output
    if tuple(prediction.shape)!=(1,300,14):raise ValueError('RT-DETR 300x14 cikti gerekli')
    raw=prediction[0].float().cpu().numpy().copy()
    if not np.isfinite(raw).all():raise ValueError('RT-DETR cikti sonlu degil')
    return raw


def capture_parent(doc,folder,baseline,predict):
    raw=[];rows=list(baseline)
    origins=[[0,-1,i] for i in range(len(baseline))]
    for crop in doc['crops']:
        # Only PNG and fixed geometry affect inference; annotation-derived status never filters a crop.
        query=predict(folder/crop['image'],crop)
        if query.shape!=(300,14) or query.dtype!=np.float32:raise ValueError('Ham query bicimi/dtype farkli')
        decoded,ids=decode_with_ids(query,crop);raw.append(query)
        rows.extend(decoded);origins.extend([1,crop['crop_index'],int(q)] for q in ids)
        k=len(raw)
        if k%8==0 or k==len(doc['crops']):print(f'  Ham crop {k}/{len(doc["crops"])} kaydediliyor',flush=True)
    a,o,status,by,selected,summary=trace_nms(rows,origins,doc['width'],doc['height'])
    arrays={'raw_queries':np.stack(raw) if raw else np.empty((0,300,14),dtype=np.float32),
            'crop_rects':np.asarray([c['source_crop_xyxy'] for c in doc['crops']],dtype=np.int32).reshape(-1,4),
            'candidate_rows':a,'candidate_origins':o,'nms_status':status,'suppressed_by_candidate':by,'selected_candidate_ids':selected}
    return arrays,summary


def save_parent(destination,arrays,metadata):
    """Commit one source folder atomically; partial sources are never reused."""
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix=destination.name+'_',dir=destination.parent))
    try:
        np.savez_compressed(temporary/'queries_and_trace.npz',**arrays)
        doc={**metadata,'archive_sha256':sha256(temporary/'queries_and_trace.npz')}
        doc['content_sha256']=fingerprint(doc);write_json(temporary/'capture.json',doc)
        if destination.exists():raise FileExistsError('Mevcut ham kaydin uzerine yazilmadi')
        temporary.rename(destination)
        return doc
    finally:
        if temporary.exists():shutil.rmtree(temporary)


def read_parent(destination,signature,source,doc_hash,old_hash):
    destination=Path(destination);meta=read_json(destination/'capture.json')
    if (meta.get('content_sha256')!=fingerprint({k:v for k,v in meta.items() if k!='content_sha256'}) or
            meta['protocol_sha256']!=signature or meta['source_image']!=source or
            meta['source_crop_document_sha256']!=doc_hash or meta['old11_prediction_sha256']!=old_hash or
            meta['archive_sha256']!=sha256(destination/'queries_and_trace.npz')):
        raise ValueError('11C ham kayit kimligi/icerigi degismis')
    with np.load(destination/'queries_and_trace.npz',allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files}
    if (arrays['raw_queries'].shape!=(meta['crop_count'],300,14) or arrays['raw_queries'].dtype!=np.float32 or
            len(arrays['nms_status'])!=len(arrays['candidate_rows'])):raise ValueError('Ham arsiv boyutlari farkli')
    return arrays,meta


def per_source_diagnostics(rawgt,width,height,baseline,old_hybrid,arrays):
    a=arrays['candidate_rows'];origin=arrays['candidate_origins'];selected=arrays['selected_candidate_ids']
    replay=a[selected].tolist()
    crop_only=old.classwise_nms(a[origin[:,0]==1].tolist(),width,height)
    integral=ignore_integral(rawgt,width,height);gt=prepare_groundtruth(rawgt,integral)
    variants={'baseline':baseline,'crop_only_fixed_NMS':crop_only,'hybrid_replay':replay,'hybrid_original_11':old_hybrid}
    output={};tp_trace={}
    baseline_rows=xyxy_to_det(baseline)
    baseline_keys={tuple(row):i for i,row in enumerate(baseline_rows)}
    baseline_candidate={int(origin[i,2]):int(i) for i in np.flatnonzero(origin[:,0]==0)}
    replay_prepared=outside_ignore(xyxy_to_det(replay),integral)
    replay_rank={tuple(row):i for i,row in enumerate(replay_prepared)}
    for threshold in [.5,.75]:
        output[str(threshold)]={}
        for variant,rows in variants.items():
            det=outside_ignore(xyxy_to_det(rows),integral)[:500]
            flags,_=detailed_matches(det,gt,threshold)
            output[str(threshold)][variant]=counts(flags)
        baseline_prepared=outside_ignore(baseline_rows,integral)[:500]
        flags,gt_ids=detailed_matches(baseline_prepared,gt,threshold)
        observations=Counter()
        for i in np.flatnonzero(flags==1):
            candidate=baseline_candidate[baseline_keys[tuple(baseline_prepared[i])]]
            state=arrays['nms_status'][candidate]
            if state==1:
                rank=replay_rank.get(tuple(baseline_prepared[i]))
                observations['selected_and_within_evaluation500' if rank is not None and rank<500 else 'selected_but_below_evaluation500']+=1
            elif state==3:observations['discarded_by_output_max1000']+=1
            else:
                suppressor=int(arrays['suppressed_by_candidate'][candidate])
                kind='crop' if origin[suppressor,0]==1 else 'baseline'
                suppressor_det=xyxy_to_det(a[suppressor:suppressor+1])
                same_gt=bool(overlaps(suppressor_det[:,:5],gt[gt_ids[i]:gt_ids[i]+1,:5])[0,0]>=threshold)
                observations['suppressed_by_'+kind+('_suppressor_can_match_same_GT' if same_gt else '_suppressor_cannot_match_same_GT')]+=1
        tp_trace[str(threshold)]=dict(observations)
    return {'original11_exactly_reproduced':replay==old_hybrid,
            'replay_detection_count':len(replay),'original11_detection_count':len(old_hybrid),
            'counts_at_fixed_max500':output,'baseline_TP_trace':tp_trace}


def capture_all(root,output,parents,index,signature,weights,official,*,factory=load_model):
    source=root/'experiments/development_eval_v1_seed42'
    old_records={x['source_image']:x for x in index['records']}
    model=None;aggregate=Counter();match_images=0;new=reused=0;crop_count=0
    combined={str(t):{v:Counter() for v in ['baseline','crop_only_fixed_NMS','hybrid_replay','hybrid_original_11']} for t in [.5,.75]}
    manifest=[];source_hashes=[];started=time.perf_counter();tp_traces={str(t):Counter() for t in [.5,.75]}
    for i,(record,doc,folder) in enumerate(parents,1):
        item=old_records[record['source_image']];path=safe_path(source,item['path'])
        annotation=source/'reference_annotations'/(path.stem+'.txt')
        if sha256(path)!=item['sha256'] or sha256(annotation)!=item['annotation_sha256']:
            raise ValueError('11 kaynak tahmin/annotation degismis')
        p=old.cached_payload(path,SOURCE_PROTOCOL,record['source_image'])
        if (p['width'],p['height'])!=(doc['width'],doc['height']) or p['all_crops_processed']!=len(doc['crops']):
            raise ValueError('11 ile 08 boyut/crop sayisi farkli')
        doc_hash=fingerprint(doc)
        if p['source_crop_document_sha256']!=doc_hash:raise ValueError('11 ile 08 crop kaynagi farkli')
        for crop in doc['crops']:
            crop_path=folder/crop['image']
            if crop_path.parent.resolve()!=folder.resolve() or sha256(crop_path)!=crop['image_sha256']:
                raise ValueError('Crop PNG yolu/kimligi farkli')
        destination=output/'sources'/path.stem
        if destination.exists():
            arrays,meta=read_parent(destination,signature,record['source_image'],doc_hash,item['sha256']);reused+=1
        else:
            if model is None:
                import torch
                if not torch.cuda.is_available():raise RuntimeError('CUDA GPU gerekli; egitim baslatilmaz')
                if shutil.disk_usage(output).free<3*2**30:raise OSError('Ham kayitlar icin 3 GiB bos disk payi gerekli')
                print('Ayni 80. tur EMA modeli yukleniyor...',flush=True)
                model=factory(weights,official,'cuda')
            arrays,summary=capture_parent(doc,folder,p['baseline'],lambda file,crop:predict_queries(model,file,crop))
            rawgt=raw_annotation(annotation.read_text(encoding='utf-8-sig'))
            diagnostic=per_source_diagnostics(rawgt,doc['width'],doc['height'],p['baseline'],p['hycado'],arrays)
            meta=save_parent(destination,arrays,{'protocol_sha256':signature,'source_image':record['source_image'],
                'source_crop_document_sha256':doc_hash,'old11_prediction_sha256':item['sha256'],
                'crop_count':len(doc['crops']),'trace_summary':summary,'diagnostics':diagnostic})
            new+=1
        aggregate.update(meta['trace_summary']);match_images+=meta['diagnostics']['original11_exactly_reproduced']
        for threshold,variants in meta['diagnostics']['counts_at_fixed_max500'].items():
            for variant,counts0 in variants.items():combined[threshold][variant].update(counts0)
        for threshold,observations in meta['diagnostics']['baseline_TP_trace'].items():tp_traces[threshold].update(observations)
        crop_count+=meta['crop_count']
        manifest.append({'source_image':record['source_image'],'path':(destination/'capture.json').relative_to(output).as_posix(),
                         'sha256':sha256(destination/'capture.json'),'archive_sha256':meta['archive_sha256'],
                         'original11_exactly_reproduced':meta['diagnostics']['original11_exactly_reproduced']})
        source_hashes.extend([(path,item['sha256']),(annotation,item['annotation_sha256'])])
        elapsed=time.perf_counter()-started
        write_json(output/'progress.json',{'status':'CAPTURING','images_done':i,'images_total':len(parents),
                   'new':new,'reused':reused,'elapsed_seconds':elapsed})
        print(f'HAM KAYIT: {i}/{len(parents)} goruntu; yeni={new}; onbellek={reused}; gecen={elapsed/60:.1f} dk',flush=True)
    model=None;gc.collect()
    for path,digest in source_hashes:
        if sha256(path)!=digest:raise ValueError('11 kaynak dosyasi kosu sirasinda degismis')
    write_json(output/'capture_index.json',{'protocol_sha256':signature,'records':manifest})
    summary={'images':len(parents),'crops':crop_count,'exact_original11_images':match_images,
             'different_original11_images':len(parents)-match_images,'trace_summary':dict(aggregate),
             'baseline_TP_trace':{t:dict(v) for t,v in tp_traces.items()},
             'counts_at_fixed_max500':{t:{v:dict(c) for v,c in variants.items()} for t,variants in combined.items()},
             'causal_trace_scope':'This replay only. Original 11 causal interpretation requires exact per-image reproduction.',
             'official_binary_parity_verified':False,'training_started':False,'parameter_search':False}
    write_json(output/'capture_summary.json',summary)
    return summary


def bundle(output):
    import os
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/'11C_ham_crop_rapor.zip';fd,name=tempfile.mkstemp(dir=output,prefix='11C_',suffix='.zip.tmp');os.close(fd)
    try:
        with zipfile.ZipFile(name,'w',zipfile.ZIP_DEFLATED) as z:
            for n in ['protocol.json','result.json','progress.json','environment.json','capture_summary.json','capture_index.json']:
                if (output/n).exists():z.write(output/n,n)
            # Small per-image summaries contain causal counts, not raw arrays.
            for path in sorted((output/'sources').glob('*/capture.json')):z.write(path,path.relative_to(output).as_posix())
            if (output/'console.log').exists():
                with (output/'console.log').open('rb') as f:
                    f.seek(max(0,(output/'console.log').stat().st_size-60000));z.writestr('console_tail.log',f.read())
        os.replace(name,target)
    finally:Path(name).unlink(missing_ok=True)
    return target


def run():
    from .__main__ import _Tee
    output=ROOT/'experiments/raw_crop_capture_v1_seed42';output.mkdir(parents=True,exist_ok=True)
    source=ROOT/'experiments/development_eval_v1_seed42';stage1=ROOT/'experiments/stage1_v1_seed42'
    source06=ROOT/'experiments/stage1_predictions_v1_seed42';crops=ROOT/'experiments/crop_dataset_v1_seed42'
    train=ROOT/'experiments/rtdetr_crop_v1_seed42'
    with training_lock(output/'.capture.lock'):
        with (output/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    print('11C HAM CROP CIKTISI VE NMS IZI:',utc_now(),flush=True)
                    print('Egitim yok. Ayni 647 goruntu, 10682 crop, ayni EMA ve esikler.',flush=True)
                    with training_lock(source/'.evaluation.lock'),training_lock(stage1/'.training.lock'),training_lock(source06/'.prediction.lock'),training_lock(crops/'.crop_data.lock'),training_lock(train/'.training.lock'):
                        cfg11,index=validate_source(source)
                        names=['protocol.json','prediction_index.json','result.json','metrics.json','metrics.csv']
                        before={n:sha256(source/n) for n in names}
                        sources=old.validate_sources(stage1,source06)
                        actual,weights,official=old.configuration(ROOT,sources)
                        if fingerprint(actual)!=SOURCE_PROTOCOL:raise ValueError('11 cikarim tarifi/kodu farkli')
                        cfg={'id':'hycado_raw_crop_capture_v1','date':'2026-10-29','source11_protocol_sha256':SOURCE_PROTOCOL,
                             'source11_metadata_hashes':before,'inference_recipe':cfg11['crop_inference'],
                             'fusion_recipe':cfg11['composition'],'capture':'all 300 normalized queries and 10 scores per crop; float32 NPZ',
                             'NMS_statuses':{'1':'selected','2':'suppressed','3':'max1000_discarded'},
                             'origin_columns':['kind_0_baseline_1_crop','crop_index_or_minus1','query_or_baseline_index'],
                             'training_started':False,'official_val_used':False,'parameter_search':False,
                             'code_sha256':sha256(__file__),'diagnostic_code_sha256':sha256(Path(__file__).with_name('dev_diagnostics.py'))}
                        signature=fingerprint(cfg)
                        if (output/'protocol.json').exists() and read_json(output/'protocol.json')!=cfg:raise ValueError('11C protokolu degismis; ham kayitlar karistirilmadi')
                        write_json(output/'protocol.json',cfg);write_json(output/'result.json',{'status':'RUNNING'})
                        (output/'capture_summary.json').unlink(missing_ok=True)
                        write_json(output/'environment.json',system_info(include_torch=True))
                        parents=old.load_parents(ROOT,sources,actual)
                        summary=capture_all(ROOT,output,parents,index,signature,weights,official)
                        if summary['images']!=647 or summary['crops']!=10682:raise ValueError('647 goruntu / 10682 crop kaydi gerekli')
                        if sha256(weights)!=old.W80 or before!={n:sha256(source/n) for n in names}:raise ValueError('Kaynak agirlik/11 kayitlari degismis')
                        result={'status':'COMPLETED','images':647,'crops':10682,'source_11_unchanged':True,
                            'source_checkpoint_unchanged':True,'training_started':False,'parameter_search':False,
                            'official_val_used':False,'exact_original11_images':summary['exact_original11_images'],
                            'trace_scope':'current replay; use per-image exact comparison for original-11 attribution'}
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();result={'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',
                         'error_type':type(exc).__name__,'error':str(exc),'resume':'Rerun 11C; committed source folders are reused'}
                write_json(output/'result.json',result);write_json(output/'progress.json',result);print('SONUC:',result['status'],flush=True)
        print('GONDERILECEK RAPOR:',bundle(output),flush=True)
        return result['status']=='COMPLETED'


def main():
    if '--report-only' in sys.argv:
        print(bundle(ROOT/'experiments/raw_crop_capture_v1_seed42'),flush=True);return 0
    return 0 if run() else 1


if __name__=='__main__':raise SystemExit(main())
