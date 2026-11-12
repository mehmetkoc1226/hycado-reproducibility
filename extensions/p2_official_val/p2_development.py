"""Frozen P2 strong-baseline DEVELOPMENT inference, with resumable predictions.

Only the 647 TRAIN-internal development images are used. Official VAL is not
opened, and annotations are read only after every prediction is complete.
Existing project code and training artifacts are used read-only.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
import traceback
import zipfile

WEIGHT_SHA = '3b61a12a89a448b849bc965e44f00afa92042a5fc0928b4c4f801e0182f84149'
PREFLIGHT_SHA = 'd64ddc3a35fbd9313105e1632f31ac7c5e4c29ca1a53ea4b7dab54f4bb71b792'
EXPECTED_NAMES = ['pedestrian','people','bicycle','car','van','truck','tricycle',
                  'awning-tricycle','bus','motor']
METHODS = ('yolo800_repeat','yolo1280','yolo1536','sahi640','sahi800')
FOLDER = 'p2_strong_baselines_v1_seed42'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,path)


def safe_image(data_root, rec):
    rel=PurePosixPath(rec['image'])
    if len(rel.parts)!=3 or rel.parts[:2]!=('images','val') or '\\' in rec['image']:
        raise ValueError('Development image path is not in frozen split')
    path=data_root.joinpath(*rel.parts)
    if digest(path)!=rec['image_sha256']:
        raise ValueError('Development image changed: '+rec['image'])
    return path


def setup(root):
    root=root.resolve();out=root/'experiments'/FOLDER
    preflight=out/'preflight.json';pilot=out/'sahi_pilot.json'
    if digest(preflight)!=PREFLIGHT_SHA:raise ValueError('P2 preflight changed')
    pre=read(preflight);p=read(pilot)
    if (pre['split_identity']['development']['images']!=647 or
        not pre['split_identity']['development']['all_image_and_annotation_sha256_verified'] or
        p['status']!='PILOT_OK' or len(p['images'])!=3 or
        p['preflight_sha256']!=PREFLIGHT_SHA or p['checkpoint_sha256']!=WEIGHT_SHA or
        p['sahi_version']!='0.11.35' or p['class_names']!=EXPECTED_NAMES or
        p['official_val_used'] or p['training_started'] or p['metric_computed']):
        raise ValueError('P2 preflight / SAHI pilot is incomplete or incompatible')
    frozen=pre['frozen_settings']
    if (frozen['id']!='hycado_p2_strong_baselines_v1' or
        [x['id'] for x in frozen['methods']]!=list(METHODS) or
        frozen['sahi']['package_version']!='0.11.35' or
        frozen['training'] is not False):
        raise ValueError('Frozen P2 settings changed')
    if importlib.metadata.version('ultralytics')!='8.3.203' or importlib.metadata.version('torch')!='2.6.0+cu124':
        raise ValueError('Pinned detector environment changed')
    weight=root/'release_stage/checkpoints/yolo_epoch_0080.pt'
    if digest(weight)!=WEIGHT_SHA:raise ValueError('Stage-1 checkpoint hash changed')
    data=root/'experiments/stage1_v1_seed42/data'
    manifest=data/'manifest.json'
    if digest(manifest)!=pre['split_identity']['development']['source_manifest_sha256']:
        raise ValueError('Stage-1 development manifest changed')
    document=read(manifest)
    records=sorted((x for x in document['records'] if x['split']=='val'),
                   key=lambda x:x['source_image'])
    if (len(records)!=647 or len({x['source_image'] for x in records})!=647 or
        len({x['image'] for x in records})!=647 or
        any(x['source_split']!='train' or not x['source_image'].startswith('images/train/')
            for x in records)):
        raise ValueError('Development split identity differs')
    sys.path[:0]=[str(out/'sahi_vendor'),str(root)]
    import torch
    import sahi
    from hycado_start import proposals
    if not torch.cuda.is_available() or importlib.metadata.version('sahi')!='0.11.35':
        raise ValueError('CUDA or pinned SAHI unavailable')
    p1=read(root/'experiments/stage1_v1_seed42/protocol.json')
    p6=read(root/'experiments/stage1_predictions_v1_seed42/protocol.json')
    p1hash=proposals.fingerprint(p1);p6hash=proposals.fingerprint(p6)
    if (p6['source_stage1_protocol_sha256']!=p1hash or
        p6['dataset_manifest_sha256']!=digest(manifest)):
        raise ValueError('Existing YOLO prediction protocol does not match frozen Stage-1')
    for name,expected in p6['code_hashes'].items():
        if Path(name).name!=name or digest(root/'hycado_start'/name)!=expected:
            raise ValueError('Frozen YOLO inference source code changed: '+name)
    metric_sources={
        'score_ablation.py':read(root/'experiments/score_ablation_v1_seed42/protocol.json')['code_sha256'],
        'dev_metrics.py':read(root/'experiments/development_eval_v1_seed42/protocol.json')['code_hashes']['dev_metrics.py']}
    for name,expected in metric_sources.items():
        if digest(root/'hycado_start'/name)!=expected:
            raise ValueError('Frozen VisDrone metric source code changed: '+name)
    original=p6['inference'];frozen_yolo=frozen['yolo']
    for key in ('batch','half','rect','conf','iou','max_det','agnostic_nms','augment'):
        if original[key]!=frozen_yolo[key]:raise ValueError('Original YOLO setting differs: '+key)
    if original['imgsz']!=800 or original['device'] not in ('0','cuda:0'):
        raise ValueError('Original YOLO inference differs')
    protocol={'id':'hycado_p2_development_v1','preflight_sha256':PREFLIGHT_SHA,
              'pilot_sha256':digest(pilot),'checkpoint_sha256':WEIGHT_SHA,
              'manifest_sha256':digest(manifest),'original_yolo_protocol_sha256':p6hash,
              'metric_source_sha256':metric_sources,
              'frozen_settings':frozen,'script_sha256':digest(__file__),
              'methods':list(METHODS),'images':647,'training':False,
              'official_val_used':False,'gt_read_only_after_all_inference':True,
              'time_measurements':'diagnostic only; not a controlled runtime benchmark'}
    proto=out/'development_protocol.json'
    if proto.exists() and read(proto)!=protocol:
        raise ValueError('Existing development protocol differs; outputs not overwritten')
    write(proto,protocol)
    return root,out,data,records,weight,pre,pilot,p1hash,p6hash,p6,proposals,fingerprint(protocol)


def checked(path,method,rec,signature,width=None,height=None):
    from hycado_start.proposal_geometry import canonical_detections
    body=read(path);sig=body.pop('content_sha256',None)
    if (sig!=fingerprint(body) or body.get('protocol_sha256')!=signature or
        body.get('method')!=method or body.get('source_image')!=rec['source_image'] or
        body.get('image_sha256')!=rec['image_sha256']):
        raise ValueError('Cached prediction identity/hash mismatch: '+str(path))
    w,h=body['width'],body['height']
    if not all(isinstance(x,int) and x>0 for x in (w,h)) or (width,height)!=(None,None) and (w,h)!=(width,height):
        raise ValueError('Cached prediction dimensions differ')
    clean,dropped=canonical_detections(body['detections'],w,h)
    if clean!=body['detections'] or dropped:
        raise ValueError('Cached predictions are not canonical')
    if method.startswith('yolo') and len(clean)>1000:
        raise ValueError('YOLO maximum detections exceeded')
    return body


def save_prediction(path,method,rec,signature,width,height,rows,dropped):
    from hycado_start.proposal_geometry import canonical_detections
    clean,bad=canonical_detections(rows,width,height)
    if method.startswith('yolo') and bad:
        raise ValueError('YOLO output unexpectedly became invalid')
    body={'protocol_sha256':signature,'method':method,'source_image':rec['source_image'],
          'image_sha256':rec['image_sha256'],'width':width,'height':height,
          'detections':clean,'dropped_zero_area':int(dropped+bad)}
    write(path,{**body,'content_sha256':fingerprint(body)})
    return checked(path,method,rec,signature,width,height)


def yolo_factory(root,out,weight,p1hash,p6,proposals,method):
    import torch
    cfg={'inference':dict(p6['inference'])}
    cfg['inference']['imgsz']={'yolo800_repeat':800,'yolo1280':1280,'yolo1536':1536}[method]
    model=proposals.load_ema(weight,p1hash)
    return proposals.make_predictor(model,cfg,out)


def sahi_factory(weight):
    from sahi import AutoDetectionModel
    model=AutoDetectionModel.from_pretrained(model_type='ultralytics',
            model_path=str(weight),confidence_threshold=.001,device='cuda:0',image_size=800)
    names=model.model.names
    names=[names[i] for i in range(len(names))] if isinstance(names,dict) else list(names)
    if names!=EXPECTED_NAMES:raise ValueError('SAHI model class names changed')
    return model


def sahi_predict(model,image,slice_size):
    from PIL import Image
    import numpy as np
    import torch
    from sahi.predict import get_sliced_prediction
    with Image.open(image) as im:
        w,h=im.size;rgb=np.array(im.convert('RGB'))
    if rgb.shape!=(h,w,3):raise ValueError('Decoded image size mismatch')
    torch.cuda.synchronize()
    pred=get_sliced_prediction(rgb,model,slice_height=slice_size,slice_width=slice_size,
              overlap_height_ratio=.2,overlap_width_ratio=.2,perform_standard_pred=True,
              postprocess_type='NMS',postprocess_match_metric='IOU',
              postprocess_match_threshold=.5,postprocess_class_agnostic=False,verbose=0)
    torch.cuda.synchronize()
    rows=[]
    for item in pred.object_prediction_list:
        xy=item.bbox.to_xyxy();rows.append([*map(float,xy),float(item.score.value),int(item.category.id)])
    return w,h,rows,0


def infer(root,out,data,records,weight,p1hash,p6hash,p6,proposals,signature,*,preview=False):
    import torch
    from PIL import Image
    indices={};total=len(records)
    for method in METHODS:
        cache=out/'development_predictions'/method
        cache.mkdir(parents=True,exist_ok=True)
        predictor=None;entries=[];start=time.perf_counter()
        for i,rec in enumerate(records,1):
            path=cache/(PurePosixPath(rec['image']).stem+'.json')
            image=safe_image(data,rec)
            if path.exists():
                payload=checked(path,method,rec,signature)
            else:
                if predictor is None:
                    predictor=(yolo_factory(root,out,weight,p1hash,p6,proposals,method)
                               if method.startswith('yolo') else sahi_factory(weight))
                if method.startswith('yolo'):
                    w,h,rows,dropped=predictor(image)
                else:
                    w,h,rows,dropped=sahi_predict(predictor,image,int(method[4:]))
                with Image.open(image) as im:
                    if im.size!=(w,h):raise ValueError('Prediction/image size mismatch')
                payload=save_prediction(path,method,rec,signature,w,h,rows,dropped)
            if method=='yolo800_repeat':
                source=proposals.prediction_path(root/'experiments/stage1_predictions_v1_seed42',rec)
                old=proposals.validate_cache(read(source),rec,p6hash)
                if (payload['width'],payload['height'],payload['detections']) != (old['width'],old['height'],old['detections']):
                    raise ValueError('YOLO800 exact baseline parity failed: '+rec['source_image'])
            entries.append({'source_image':rec['source_image'],'path':path.relative_to(out).as_posix(),
                            'sha256':digest(path),'detections':len(payload['detections'])})
            if i in (1,3,total) or i%25==0:
                state={'status':'SMOKE' if preview else 'INFERENCE',
                       'method':method,'images_done':i,'images_total':total,
                       'diagnostic_elapsed_seconds_this_method':time.perf_counter()-start,
                       'training':False,'official_val_used':False}
                write(out/'development_progress.json',state)
                print(f'P2 {"SMOKE" if preview else "DEVELOPMENT"} {method}: {i}/{total}',flush=True)
        indices[method]=entries
        predictor=None;gc.collect();torch.cuda.empty_cache()
    write(out/('development_smoke_index.json' if preview else 'development_prediction_index.json'),
          {'protocol_sha256':signature,'methods':indices,'images_per_method':total,
           'yolo800_exact_parity':True})
    return indices


def measure(out,data,records,indices,signature):
    import numpy as np
    from hycado_start.dev_metrics import raw_annotation,ignore_integral,prepare_groundtruth,outside_ignore,xyxy_to_det
    from hycado_start.score_ablation import fast_flags,aggregate,THRESHOLDS
    if any(len(indices.get(m,[]))!=647 for m in METHODS):
        raise ValueError('All predictions must be committed before GT is read')
    parts=[]
    for i,rec in enumerate(records,1):
        anno=data/'annotations/val'/(PurePosixPath(rec['image']).stem+'.txt')
        if digest(anno)!=rec['annotation_sha256']:
            raise ValueError('Development annotation changed: '+str(anno))
        predictions=[]
        for method in METHODS:
            item=indices[method][i-1]
            if item['source_image']!=rec['source_image']:
                raise ValueError('Prediction index order differs')
            path=out/item['path']
            if digest(path)!=item['sha256']:
                raise ValueError('Prediction file changed')
            predictions.append((method,checked(path,method,rec,signature)))
        w,h=predictions[0][1]['width'],predictions[0][1]['height']
        if any((p['width'],p['height'])!=(w,h) for _,p in predictions):
            raise ValueError('Methods disagree on image size')
        raw=raw_annotation(anno.read_text(encoding='utf-8-sig'))
        integral=ignore_integral(raw,w,h);gt=prepare_groundtruth(raw,integral)
        part={'gt_denominators':np.array([np.count_nonzero(gt[:,5]==c) for c in range(1,11)],dtype=np.int64)}
        part['presence']=(part['gt_denominators']>0).astype(np.int64)
        for method,p in predictions:
            det=outside_ignore(xyxy_to_det(p['detections']),integral)[:500]
            part[method+'_scores']=det[:,4].copy()
            part[method+'_classes']=det[:,5].astype(np.int8)
            part[method+'_flags']=np.column_stack([fast_flags(det,gt,float(t)) for t in THRESHOLDS])
        parts.append(part)
        if i==1 or i%25==0 or i==647:
            write(out/'development_progress.json',{'status':'METRICS','images_done':i,'images_total':647})
            print(f'P2 DEVELOPMENT METRICS: {i}/647',flush=True)
    metrics={method:aggregate(parts,method) for method in METHODS}
    result={'status':'DEVELOPMENT_COMPLETE','images':647,'methods':metrics,
            'yolo800_exact_parity':True,'official_val_used':False,'training':False,
            'primary_metric':'provisional VisDrone toolkit-aligned Python AP; percent',
            'runtime_scope':'diagnostic progress only; no benchmark/FPS claim'}
    write(out/'development_metrics.json',result)
    with (out/'development_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        cols=['method','AP','AP50','AP75','AR1','AR10','AR100','AR500','class_macro_AP_separate_diagnostic']
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for name in METHODS:writer.writerow({'method':name,**{c:metrics[name][c] for c in cols[1:]}})
    return result


def bundle(out):
    path=out/'05_p2_gelistirme_rapor.zip';temp=path.with_suffix('.partial')
    names=['preflight.json','sahi_pilot.json','development_protocol.json',
           'development_smoke_index.json',
           'development_prediction_index.json','development_metrics.json',
           'development_metrics.csv','development_result.json','development_progress.json']
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            p=out/name
            if p.exists():z.write(p,arcname=name)
    os.replace(temp,path)
    return path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path(r'D:\HyCADO_Temiz_Baslangic'))
    args=parser.parse_args();out=args.project/'experiments'/FOLDER
    try:
        context=setup(args.project)
        _,out,data,records,weight,_,_,p1hash,p6hash,p6,proposals,signature=context
        infer(args.project,out,data,records[:3],weight,p1hash,p6hash,p6,proposals,signature,preview=True)
        indices=infer(args.project,out,data,records,weight,p1hash,p6hash,p6,proposals,signature)
        results=measure(out,data,records,indices,signature)
        write(out/'development_result.json',{'status':'DEVELOPMENT_COMPLETE','images':647,
              'methods':list(METHODS),'yolo800_exact_parity':True,'training':False,
              'official_val_used':False})
        print('DEVELOPMENT_COMPLETE:',bundle(out),flush=True)
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        traceback.print_exc()
        write(out/'development_result.json',{'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',
              'error':repr(exc),'traceback':traceback.format_exc(),
              'resume':'Rerun the same CMD; completed prediction files are checked and reused'})
        print('P2 GELISTIRME DURDU; ayni CMD kaldigi yerden devam eder. Rapor:',bundle(out),flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
