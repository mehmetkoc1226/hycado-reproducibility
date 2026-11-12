"""Frozen P2 strong baselines on 548 official VisDrone VAL images.

Run only after the 647-image TRAIN-development gate. Every prediction is
committed before any official-VAL annotation is opened. Never trains or tunes.
"""
from __future__ import annotations
import argparse
import csv
import gc
from pathlib import Path, PurePosixPath
import traceback
import zipfile
import os
import sys
import time

import p2_development as d  # byte-identical 647-image development runner

DEV_METRICS_SHA='a532c30548c688951ed4a0039402d50ca2f5649c92d43b20cb621fdd106c2da9'
DEV_INDEX_SHA='8c72b5c99566539869acac45b213bbf18fd35a50fc9a5c31a98a870f0b2c32c0'
DEV_PROTOCOL_SHA='7de91b5fa14633790834e6ce3040d416603ffbaab816edd4aa7fba301b29736b'
DEVELOPMENT_SCRIPT_SHA='53ec58a853e3c93ca4363df9ec25df3fa08d808f5c5da31d1579a8cc979619b1'
# Independently recorded in the completed data/model audit (25/source_bindings.json).
PREPARED_MANIFEST_SHA='e2b610a276aea28c4d5d3ec47e45c381d4b4fc2b09f3ef20d57495a0677905fa'
# The preflight's official_val.source_manifest_sha256 field accidentally held
# the old official-VAL protocol FILE hash, not the prepared manifest file hash.
OLD_BASELINE_PROTOCOL_FILE_SHA='bf406f50d8441fc66346be30615ebdb94b009625e5946e5fa59dd1e5427e686f'
METHODS=d.METHODS

def verify_manifest_identity(pre, manifest, old_protocol_path):
    recorded=pre['split_identity']['official_val']['source_manifest_sha256']
    if recorded!=OLD_BASELINE_PROTOCOL_FILE_SHA or d.digest(old_protocol_path)!=recorded:
        raise ValueError('Frozen official-VAL baseline protocol differs from audited preflight')
    old_proto=d.read(old_protocol_path)
    if old_proto['prepared_manifest_sha256']!=PREPARED_MANIFEST_SHA:
        raise ValueError('Previously frozen official-VAL manifest identity differs from data audit')
    if d.digest(manifest)!=PREPARED_MANIFEST_SHA:
        raise ValueError('Official original manifest differs from frozen protocol and data audit')
    return old_proto

def setup(root):
    if d.digest(d.__file__)!=DEVELOPMENT_SCRIPT_SHA:
        raise ValueError('Development runner source changed')
    context=d.setup(root)
    root,out,_,_,weight,pre,pilot,p1hash,p6hash,p6,proposals,_=context
    for name,expected in [('development_metrics.json',DEV_METRICS_SHA),
                          ('development_prediction_index.json',DEV_INDEX_SHA),
                          ('development_protocol.json',DEV_PROTOCOL_SHA)]:
        if d.digest(out/name)!=expected:
            raise ValueError('Completed development gate changed: '+name)
    result=d.read(out/'development_result.json')
    metrics=d.read(out/'development_metrics.json')
    if (result.get('status')!='DEVELOPMENT_COMPLETE' or metrics.get('status')!='DEVELOPMENT_COMPLETE' or
        metrics.get('images')!=647 or not metrics.get('yolo800_exact_parity') or
        not result.get('yolo800_exact_parity') or metrics.get('official_val_used') or
        result.get('official_val_used') or set(metrics.get('methods',{}))!=set(METHODS)):
        raise ValueError('Development gate not complete')
    prepared=root/'data/prepared';manifest=prepared/'manifest.json'
    old_dir=root/'experiments/official_val_a025_v1_seed42'
    old_protocol_path=old_dir/'protocol.json'
    old_proto=verify_manifest_identity(pre,manifest,old_protocol_path)
    source=d.read(manifest)
    train=[x for x in source['records'] if x['split']=='train']
    raw=sorted((x for x in source['records'] if x['split']=='val'),key=lambda x:x['image'])
    if (source.get('purpose')!='fresh_original_visdrone_conversion' or
        len(train)!=6471 or len(raw)!=548 or
        {x['image_sha256'] for x in train}&{x['image_sha256'] for x in raw} or
        len({x['image'] for x in raw})!=548 or
        not pre['split_identity']['official_val']['all_image_and_annotation_sha256_verified']):
        raise ValueError('Official VAL identity differs')
    records=[]
    for rec in raw:
        rel=PurePosixPath(rec['image'])
        if len(rel.parts)!=3 or rel.parts[:2]!=('images','val') or '\\' in rec['image']:
            raise ValueError('Official VAL path invalid')
        records.append({k:rec[k] for k in ('image','image_sha256','annotation_sha256','width','height')}
                       | {'source_image':rec['image']})
    if [{k:rec[k] for k in ('image','image_sha256','annotation_sha256','width','height')}
        for rec in records]!=old_proto['records']:
        raise ValueError('Official VAL record identity differs from previously frozen baseline')
    old_signature=d.fingerprint(old_proto)
    old_metric=d.read(old_dir/'metrics.json')['variants']['baseline']
    protocol={'id':'hycado_p2_official_val_v1','development_protocol_sha256':DEV_PROTOCOL_SHA,
              'development_metrics_sha256':DEV_METRICS_SHA,
              'development_prediction_index_sha256':DEV_INDEX_SHA,
              'preflight_sha256':d.PREFLIGHT_SHA,'checkpoint_sha256':d.WEIGHT_SHA,
              'official_manifest_sha256':PREPARED_MANIFEST_SHA,'previous_baseline_protocol_sha256':old_signature,
              'preflight_field_correction':{'field':'split_identity.official_val.source_manifest_sha256',
                  'recorded_as_previous_baseline_protocol_file_sha256':OLD_BASELINE_PROTOCOL_FILE_SHA,
                  'actual_prepared_manifest_sha256':PREPARED_MANIFEST_SHA,
                  'independent_audit':'25_veri_model_denetim_rapor.zip/source_bindings.json'},
              'script_sha256':d.digest(__file__),'development_script_sha256':DEVELOPMENT_SCRIPT_SHA,
              'frozen_settings':pre['frozen_settings'],'methods':list(METHODS),'images':548,
              'training':False,'parameter_search':False,'official_val_used':True,
              'annotation_policy':'not opened until every inference result is committed',
              'metric':'same pinned toolkit-aligned Python evaluator as completed development; provisional',
              'timing':'diagnostic only; no end-to-end FPS or Pareto claim'}
    path=out/'official_val_protocol.json'
    if path.exists() and d.read(path)!=protocol:
        raise ValueError('An existing official-VAL protocol differs; outputs not mixed')
    d.write(path,protocol)
    return root,out,prepared,records,weight,p1hash,p6hash,p6,proposals,old_dir,old_signature,old_metric,d.fingerprint(protocol)

def old_baseline(old_dir,rec,signature):
    path=old_dir/'baseline'/(PurePosixPath(rec['image']).stem+'.json')
    item=d.read(path);body={k:v for k,v in item.items() if k!='content_sha256'}
    if (item.get('content_sha256')!=d.fingerprint(body) or
        item.get('protocol_sha256')!=signature or
        item.get('source_image')!=rec['image'] or
        item.get('image_sha256')!=rec['image_sha256'] or
        (item.get('width'),item.get('height'))!=(rec['width'],rec['height'])):
        raise ValueError('Previously frozen official baseline differs: '+rec['image'])
    return item['detections']

def infer(root,out,prepared,records,weight,p1hash,p6hash,p6,proposals,old_dir,old_signature,signature,*,preview=False):
    import torch
    from PIL import Image
    entries={};total=len(records)
    for method in METHODS:
        folder=out/'official_val_predictions'/method
        folder.mkdir(parents=True,exist_ok=True)
        predictor=None;index=[];started=time.perf_counter()
        for i,rec in enumerate(records,1):
            image=d.safe_image(prepared,rec)
            path=folder/(PurePosixPath(rec['image']).stem+'.json')
            if path.exists():
                p=d.checked(path,method,rec,signature,rec['width'],rec['height'])
            else:
                if predictor is None:
                    predictor=(d.yolo_factory(root,out,weight,p1hash,p6,proposals,method)
                               if method.startswith('yolo') else d.sahi_factory(weight))
                if method.startswith('yolo'):
                    width,height,rows,dropped=predictor(image)
                else:
                    width,height,rows,dropped=d.sahi_predict(predictor,image,int(method[4:]))
                if (width,height)!=(rec['width'],rec['height']):
                    raise ValueError('Official image/prediction dimensions differ')
                p=d.save_prediction(path,method,rec,signature,width,height,rows,dropped)
            if method=='yolo800_repeat' and p['detections']!=old_baseline(old_dir,rec,old_signature):
                raise ValueError('Official YOLO800 exact frozen baseline parity failed: '+rec['image'])
            index.append({'source_image':rec['image'],'path':path.relative_to(out).as_posix(),
                          'sha256':d.digest(path),'detections':len(p['detections'])})
            if i in (1,3,total) or i%25==0:
                d.write(out/'official_val_progress.json',{'status':'SMOKE' if preview else 'INFERENCE',
                    'method':method,'images_done':i,'images_total':total,'training':False,
                    'diagnostic_elapsed_seconds_this_method':time.perf_counter()-started})
                print(f'P2 OFFICIAL VAL {"SMOKE" if preview else "INFERENCE"} {method}: {i}/{total}',flush=True)
        entries[method]=index
        predictor=None;gc.collect();torch.cuda.empty_cache()
    d.write(out/('official_val_smoke_index.json' if preview else 'official_val_prediction_index.json'),
            {'protocol_sha256':signature,'methods':entries,'images_per_method':total,
             'yolo800_exact_frozen_parity':True})
    return entries

def evaluate(out,prepared,records,indices,signature,old_metric):
    import numpy as np
    from hycado_start.dev_metrics import raw_annotation,ignore_integral,prepare_groundtruth,outside_ignore,xyxy_to_det
    from hycado_start.score_ablation import fast_flags,aggregate,THRESHOLDS
    if any(len(indices.get(m,[]))!=548 for m in METHODS):
        raise ValueError('All 5x548 predictions required before official GT access')
    parts=[]
    for i,rec in enumerate(records,1):
        predictions=[]
        for method in METHODS:
            item=indices[method][i-1]
            if item['source_image']!=rec['image'] or d.digest(out/item['path'])!=item['sha256']:
                raise ValueError('Official prediction index/hash mismatch')
            predictions.append((method,d.checked(out/item['path'],method,rec,signature,rec['width'],rec['height'])))
        annotation=prepared/'visdrone_annotations/val'/(PurePosixPath(rec['image']).stem+'.txt')
        if d.digest(annotation)!=rec['annotation_sha256']:
            raise ValueError('Official annotation changed')
        raw=raw_annotation(annotation.read_text(encoding='utf-8-sig'))
        integral=ignore_integral(raw,rec['width'],rec['height'])
        gt=prepare_groundtruth(raw,integral)
        part={'gt_denominators':np.array([np.count_nonzero(gt[:,5]==c) for c in range(1,11)],dtype=np.int64)}
        part['presence']=(part['gt_denominators']>0).astype(np.int64)
        for method,p in predictions:
            det=outside_ignore(xyxy_to_det(p['detections']),integral)[:500]
            part[method+'_scores']=det[:,4].copy();part[method+'_classes']=det[:,5].astype(np.int8)
            part[method+'_flags']=np.column_stack([fast_flags(det,gt,float(t)) for t in THRESHOLDS])
        parts.append(part)
        if i in (1,548) or i%25==0:
            d.write(out/'official_val_progress.json',{'status':'METRICS','images_done':i,'images_total':548})
            print(f'P2 OFFICIAL VAL METRICS: {i}/548',flush=True)
    metrics={name:aggregate(parts,name) for name in METHODS}
    for key in ('AP','AP50','AP75','AR1','AR10','AR100','AR500'):
        if abs(metrics['yolo800_repeat'][key]-old_metric[key])>1e-8:
            raise ValueError('Official frozen baseline metric parity failed: '+key)
    result={'status':'OFFICIAL_VAL_COMPLETE','images':548,'methods':metrics,
            'yolo800_exact_frozen_parity':True,'old_baseline_metric_parity':True,
            'training':False,'parameter_search':False,'official_val_used':True,
            'metric_scope':'pinned toolkit-aligned Python metric; not COCO mAP or a new leaderboard',
            'runtime_scope':'diagnostic only; controlled timing remains a separate step'}
    d.write(out/'official_val_metrics.json',result)
    cols=['method','AP','AP50','AP75','AR1','AR10','AR100','AR500','class_macro_AP_separate_diagnostic']
    with (out/'official_val_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for method in METHODS:
            writer.writerow({'method':method,**{c:metrics[method][c] for c in cols[1:]}})
    return result

def bundle(out):
    path=out/'06_p2_resmi_val_rapor.zip';temp=path.with_suffix('.partial')
    names=['preflight.json','sahi_pilot.json','development_protocol.json','development_metrics.json',
           'official_val_protocol.json','official_val_smoke_index.json',
           'official_val_prediction_index.json','official_val_metrics.json','official_val_metrics.csv',
           'official_val_result.json','official_val_progress.json']
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            p=out/name
            if p.exists():z.write(p,arcname=name)
    os.replace(temp,path)
    return path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,default=Path(r'D:\HyCADO_Temiz_Baslangic'))
    args=parser.parse_args();out=args.project/'experiments'/d.FOLDER
    try:
        context=setup(args.project)
        root,out,prepared,records,weight,p1hash,p6hash,p6,proposals,old_dir,old_signature,old_metric,signature=context
        infer(root,out,prepared,records[:3],weight,p1hash,p6hash,p6,proposals,old_dir,old_signature,signature,preview=True)
        indices=infer(root,out,prepared,records,weight,p1hash,p6hash,p6,proposals,old_dir,old_signature,signature)
        evaluate(out,prepared,records,indices,signature,old_metric)
        d.write(out/'official_val_result.json',{'status':'OFFICIAL_VAL_COMPLETE','images':548,
             'methods':list(METHODS),'training':False,'parameter_search':False,
             'yolo800_exact_frozen_parity':True,'old_baseline_metric_parity':True})
        print('OFFICIAL_VAL_COMPLETE:',bundle(out),flush=True)
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        traceback.print_exc()
        d.write(out/'official_val_result.json',{'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',
               'error':repr(exc),'traceback':traceback.format_exc(),
               'resume':'Rerun the same CMD; signed completed predictions are checked and reused'})
        print('OFFICIAL VAL DURDU. Rapor:',bundle(out),flush=True)
        return 1

if __name__=='__main__':raise SystemExit(main())
