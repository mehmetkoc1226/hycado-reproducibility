"""Frozen development inference: Stage-1 cache vs crop RT-DETR EMA composition.

No training, no hyperparameter search, no official VAL access. All 10682 crops
are used, including crops excluded from training by annotation-derived status.
"""
from __future__ import annotations
import contextlib
import csv
import gc
import importlib.metadata
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import traceback
import zipfile
import numpy as np

from .common import ROOT, sha256, write_json, system_info, utc_now
from .stage1 import fingerprint, training_lock
from .crop_loader import safe_path, read_json
from .crop_merge import validate_sources
from .proposals import prediction_path, validate_cache
from .proposal_geometry import canonical_detections
from .dev_metrics import (raw_annotation, xyxy_to_det, ignore_integral, outside_ignore,
                          prepare_groundtruth, evaluate, REFERENCE_COMMIT)

P10 = '6ae4d2ae2c503b18fe467d3c275605d527bd8934e8f6128276daeaa5327491ae'
W80 = 'fa6e3991b26650e2426a4bbac9253351d47087e2533d3b98d86a2f24b586cbf9'
P06 = 'f4e33b895d33e9314607ae68cf051d94990b7f181fd0c9e13d76cd0ce3d3c354'


def configuration(root, sources):
    train = root / 'experiments/rtdetr_crop_v1_seed42'
    data = root / 'experiments/crop_dataset_v1_seed42'
    p = read_json(train / 'protocol.json'); result = read_json(train / 'result.json')
    index = read_json(train / 'checkpoint_index.json')
    if (fingerprint(p) != P10 or result.get('status') != 'COMPLETED' or
            result.get('completed_rounds') != 80 or index['protocol_sha256'] != P10 or
            index['latest']['epoch'] != 80 or index['latest']['sha256'] != W80):
        raise ValueError('Dogrulanmis 80/80 crop egitimi gerekli')
    weights = safe_path(train, index['latest']['path'])
    print('80. tur model dosyasi denetleniyor...', flush=True)
    if sha256(weights) != W80: raise ValueError('80. tur agirligi degismis')
    for name, value in p['code_hashes'].items():
        if Path(name).name != name or sha256(root / 'hycado_start' / name) != value:
            raise ValueError('Dondurulmus egitim kodu degismis: ' + name)
    for name, value in p['runtime_versions'].items():
        if importlib.metadata.version(name) != value: raise ValueError('Paket surumu farkli: ' + name)
    if fingerprint(sources['p6']) != P06: raise ValueError('Sabit 06 tahmin protokolu farkli')
    p8 = read_json(data / 'protocol.json')
    if (fingerprint(p8) != p['catalog']['source08_protocol_sha256'] or
            sha256(data / 'source_index.json') != p['catalog']['source_index_sha256'] or
            sha256(data / 'dataset_summary.json') != p['catalog']['summary_sha256'] or
            read_json(data / 'result.json')['status'] != 'COMPLETED'):
        raise ValueError('Dondurulmus crop verisi kimligi farkli')
    for name, value in p8['code_hashes'].items():
        if Path(name).name != name or sha256(root / 'hycado_start' / name) != value:
            raise ValueError('08 kaynak kodu farkli: ' + name)
    official = root / 'work/fresh_official_pretrained/rtdetr-l.pt'
    if sha256(official) != p['initial_weights_sha256']: raise ValueError('Mimari kaynagi COCO agirligi farkli')
    config = {'id': 'hycado_development_evaluation_v1_seed42', 'decision_date': '2026-10-29',
              'source_training_protocol_sha256': P10, 'crop_ema_checkpoint': index['latest'],
              'stage1_protocol_sha256': sources['p6']['source_stage1_protocol_sha256'],
              'stage1_prediction_protocol_sha256': P06,
              'prediction_index_sha256': sources['prediction_index_sha256'],
              'crop_source_index_sha256': p['catalog']['source_index_sha256'],
              'source08_protocol_sha256': fingerprint(p8),
              'dataset_manifest_sha256': sources['p6']['dataset_manifest_sha256'],
              'split': 'official TRAIN internal development only; 647 images',
              'official_val_used': False, 'training_started': False, 'parameter_search': False,
              'all_development_crops': 10682, 'annotation_based_crop_filtering_at_inference': False,
              'crop_inference': {'imgsz': 960, 'batch': 1, 'weights': 'round80 EMA',
                'precision': 'float32 weights/input with CUDA float16 autocast', 'fuse': False,
                'preprocess': 'Pillow RGB, OpenCV INTER_LINEAR square scale-fill, float32 /255',
                'output': 'normalized cxcywh + sigmoid class scores; best class per query',
                'confidence_strictly_greater_than': .001, 'per_crop_nms': False, 'augment': False},
              'composition': {'mode': 'union of unchanged Stage1 cache and mapped crop detections',
                'classwise_nms_iou': .7, 'suppress_if_iou_strictly_greater': True,
                'maximum_output_detections_per_image': 1000, 'score_calibration': 'none',
                'tie_order': '-score, class_id, x1,y1,x2,y2, stable input position',
                'no_pre_nms_candidate_cap': True},
              'geometry_policy': 'unchanged 08 materialized crop coordinates; no new GT-guided proposals',
              'evaluation': {'reference_commit': REFERENCE_COMMIT, 'reference_max_detections': [1,10,100,500],
                'annotation_source': 'raw original annotation copies, not YOLO labels',
                'python_results_provisional_until_official_binary_parity': True,
                'aggregation': 'literal toolkit class-image multiplicity; separate conventional macro diagnostic'},
              'runtime_versions': p['runtime_versions'],
              'timing': 'execution progress only; cached Stage1 makes these times unsuitable for detector speed claims',
              'code_hashes': {n: sha256(Path(__file__).with_name(n)) for n in ['dev_evaluation.py','dev_metrics.py']}}
    return config, weights, official


def load_parents(root, sources, cfg):
    base = root / 'experiments/crop_dataset_v1_seed42'
    index = read_json(base / 'source_index.json')
    mapping = {x['source_image']: x for x in index['records'] if x['split'] == 'val'}
    records = sorted((r for r in sources['manifest']['records'] if r['split'] == 'val'), key=lambda r:r['source_image'])
    if len(mapping) != len(records) or len(records) != 647: raise ValueError('647 gelistirme goruntusu gerekli')
    parents = []; count = 0
    for r in records:
        item = mapping[r['source_image']]; path = safe_path(base, item['source_record'])
        if sha256(path) != item['sha256']: raise ValueError('Crop kaynak dokumani degismis')
        doc = read_json(path)
        body = {k:v for k,v in doc.items() if k != 'content_sha256'}
        if (doc['content_sha256'] != fingerprint(body) or doc['protocol_sha256'] != cfg['source08_protocol_sha256'] or
                doc['source_image'] != r['source_image'] or doc['source_image_path'] != r['image'] or
                doc['group'] != r['group'] or doc['split'] != 'val' or
                doc['crop_count'] != len(doc['crops']) or item['crop_count'] != len(doc['crops'])):
            raise ValueError('Crop kaynak kimligi farkli')
        # Every crop, including fully-ignored and background crops; no target-based exclusion.
        for i, crop in enumerate(doc['crops']):
            x1,y1,x2,y2 = crop['source_crop_xyxy']
            if (crop['crop_index'] != i or not all(type(v) is int for v in (x1,y1,x2,y2)) or
                    not 0 <= x1 < x2 <= doc['width'] or not 0 <= y1 < y2 <= doc['height'] or
                    x2-x1 != crop['width'] or y2-y1 != crop['height']):
                raise ValueError('Crop geometrisi gecersiz')
        count += len(doc['crops'])
        parents.append((r, doc, path.parent))
    if count != 10682: raise ValueError('10682 crop gerekli; veri secimi degistirilmedi')
    return parents


def map_queries(prediction, crop):
    """Pure numpy counterpart of RT-DETR best-class decoding; no score retuning."""
    a = np.asarray(prediction, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 14 or not np.isfinite(a).all():
        raise ValueError('RT-DETR cikti bicimi/sonlulugu gecersiz')
    if np.any(a[:,2:4] < 0) or np.any((a[:,4:] < 0) | (a[:,4:] > 1)):
        raise ValueError('Normalize kutu veya sinif skor araligi gecersiz')
    cls = np.argmax(a[:, 4:], axis=1); score = a[np.arange(len(a)), cls+4]
    keep = score > .001
    a = a[keep]; cls = cls[keep]; score = score[keep]
    cx,cy,w,h = a[:,:4].T
    x0,y0,xend,yend = crop['source_crop_xyxy']
    width, height = xend-x0, yend-y0
    boxes = np.column_stack([np.clip(cx-w/2,0,1)*width+x0, np.clip(cy-h/2,0,1)*height+y0,
                             np.clip(cx+w/2,0,1)*width+x0, np.clip(cy+h/2,0,1)*height+y0,score,cls])
    good = (boxes[:,2] > boxes[:,0]) & (boxes[:,3] > boxes[:,1])
    return boxes[good].tolist()


def classwise_nms(rows, width, height, threshold=.7, limit=1000):
    clean, _ = canonical_detections(rows, width, height)
    a = np.asarray(clean, dtype=np.float64).reshape(-1,6)
    remaining = np.arange(len(a)); selected = []
    area = (a[:,2]-a[:,0])*(a[:,3]-a[:,1])
    while len(remaining) and len(selected) < limit:
        i = remaining[0]; selected.append(int(i)); other = remaining[1:]
        intersection = np.maximum(0,np.minimum(a[i,2:4],a[other,2:4])-np.maximum(a[i,:2],a[other,:2])).prod(axis=1)
        union = area[i]+area[other]-intersection
        iou = np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
        remaining = other[~((a[other,5]==a[i,5]) & (iou > threshold))]
    return a[selected].tolist()


def build_predictor(weights, official):
    import torch
    import cv2
    from PIL import Image
    from .crop_training_v2 import build_model, seed_all
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU gerekli; CPU tahmini baslatilmadi')
    seed_all(42); torch.set_num_threads(2); cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True,warn_only=True)
    saved = torch.load(weights,map_location='cpu',weights_only=False)
    if (saved.get('schema') != 'hycado_crop_experiment_checkpoint_v1' or saved.get('protocol_sha256') != P10 or
            saved.get('round') != 80 or len(saved.get('history',[])) != 80):
        raise ValueError('80. tur checkpoint icerigi farkli')
    model = build_model(official,'cpu')
    model.load_state_dict(saved['ema'],strict=True); del saved; gc.collect()
    model = model.float().eval().to('cuda')
    for p in model.parameters(): p.requires_grad_(False)
    if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values() if v.is_floating_point()):
        raise ValueError('EMA degerleri sonlu degil')

    def predict(path,crop):
        # Image bytes were hashed by the caller immediately before this read.
        with Image.open(path) as im:
            if im.size != (crop['width'],crop['height']): raise ValueError('Crop goruntu boyutu farkli')
            rgb = np.array(im.convert('RGB'))
        resized = cv2.resize(rgb,(960,960),interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(resized.transpose(2,0,1).copy()).unsqueeze(0).to('cuda',dtype=torch.float32)/255.
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
            output = model.predict(x)
        prediction = output[0] if isinstance(output,(tuple,list)) else output
        if tuple(prediction.shape) != (1,300,14): raise ValueError('Beklenen RT-DETR 300x14 cikti yok')
        return map_queries(prediction[0].float().cpu().numpy(),crop)
    return predict


def cached_payload(path, signature, source):
    p = read_json(path)
    if p.get('content_sha256') != fingerprint({k:v for k,v in p.items() if k!='content_sha256'}):
        raise ValueError('11 tahmin onbellegi degismis')
    if p.get('protocol_sha256') != signature or p.get('source_image') != source:
        raise ValueError('11 tahmin onbellegi protokol/kaynak farkli')
    for key in ['baseline','hycado']:
        clean,bad = canonical_detections(p[key],p['width'],p['height'])
        if bad or clean != p[key] or len(clean)>1000: raise ValueError('Onbellek tahminleri gecersiz')
    return p


def export_txt(path, rows):
    # Decimal 17-digit round-trip keeps Python arrays and reference export aligned.
    text = ''.join(','.join(format(float(v),'.17g') for v in [*r[:5],int(r[5]),-1,-1])+'\n' for r in xyxy_to_det(rows))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='ascii')


def infer(root, out, sources, cfg, weights, official, parents, *, predictor_factory=build_predictor):
    signature = fingerprint(cfg); base = root/'experiments/crop_dataset_v1_seed42'
    source06 = root/'experiments/stage1_predictions_v1_seed42'; data = root/'experiments/stage1_v1_seed42/data'
    predictor = None; index=[]; new=reused=0; started=time.perf_counter()
    for number,(record,doc,folder) in enumerate(parents,1):
        stem = Path(record['image']).stem
        image = safe_path(data,record['image']); annotation = data/'annotations/val'/(stem+'.txt')
        if sha256(image)!=record['image_sha256'] or sha256(annotation)!=record['annotation_sha256']:
            raise ValueError('Kaynak gelistirme verisi degismis: '+stem)
        path06 = prediction_path(source06,record)
        if sha256(path06)!=sources['index'][record['source_image']]['sha256']:
            raise ValueError('06 tahmin dosyasi degismis')
        original = validate_cache(read_json(path06),record,P06)
        if (original['width'],original['height'])!=(doc['width'],doc['height']):
            raise ValueError('Kaynak boyutu 06/08 arasinda farkli')
        cache = out/'predictions'/(stem+'.json')
        # Integrity checked even on cache reuse. Do not substitute annotation status for inference.
        paths=[]
        for crop in doc['crops']:
            relative = (folder/crop['image']).relative_to(base).as_posix(); path = safe_path(base,relative)
            if sha256(path)!=crop['image_sha256']: raise ValueError('Crop PNG degismis: '+str(path))
            paths.append(path)
        if cache.exists():
            payload = cached_payload(cache,signature,record['source_image']); reused+=1
        else:
            if predictor is None:
                print('EMA model yukleniyor; egitim yapilmaz...',flush=True)
                predictor = predictor_factory(weights,official)
            crops=[]; crop_start=time.perf_counter()
            for k,(crop,path) in enumerate(zip(doc['crops'],paths),1):
                crops.extend(predictor(path,crop))
                if k%8==0 or k==len(paths):
                    print(f'Goruntu {number}/647: crop {k}/{len(paths)}',flush=True)
            hybrid = classwise_nms(original['detections']+crops,doc['width'],doc['height'])
            payload={'protocol_sha256':signature,'source_image':record['source_image'],
                     'width':doc['width'],'height':doc['height'],'baseline':original['detections'],
                     'hycado':hybrid,'all_crops_processed':len(paths),'crop_candidates':len(crops),
                     'source_crop_document_sha256':fingerprint(doc),
                     'crop_and_merge_seconds':time.perf_counter()-crop_start}
            payload['content_sha256']=fingerprint(payload);write_json(cache,payload);new+=1
        if (payload['all_crops_processed']!=len(paths) or payload['source_crop_document_sha256']!=fingerprint(doc) or
                payload['baseline']!=original['detections'] or payload['width']!=doc['width'] or payload['height']!=doc['height']):
            raise ValueError('11/06/08 tahmin ve kaynak baglantisi farkli')
        for variant in ['baseline','hycado']: export_txt(out/'detections'/variant/(stem+'.txt'),payload[variant])
        # Byte-exact raw GT export; never substitute clipped/deduplicated YOLO labels.
        target=out/'reference_annotations'/(stem+'.txt');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(annotation.read_bytes())
        index.append({'source_image':record['source_image'],'path':cache.relative_to(out).as_posix(),
                      'sha256':sha256(cache),'annotation_sha256':record['annotation_sha256'],
                      'width':doc['width'],'height':doc['height']})
        state={'status':'PREDICTING','images_done':number,'images_total':647,'new':new,'reused':reused,
               'elapsed_this_invocation_seconds':time.perf_counter()-started}
        write_json(out/'progress.json',state)
        print(f'GELISTIRME TAHMINI: {number}/647; yeni={new}; onbellek={reused}',flush=True)
    predictor = None; gc.collect()
    import torch
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    write_json(out/'prediction_index.json',{'protocol_sha256':signature,'records':index})
    return index


def measure(out, index, signature):
    images=[]; raw_cells=[]; base_cells=[]; hybrid_cells=[]; sizes=[]
    for i,item in enumerate(index,1):
        cache=safe_path(out,item['path'])
        if sha256(cache)!=item['sha256']: raise ValueError('Degerlendirme onbellegi degismis')
        p=cached_payload(cache,signature,item['source_image']); stem=Path(item['path']).stem
        annotation=out/'reference_annotations'/(stem+'.txt')
        if sha256(annotation)!=item['annotation_sha256']:raise ValueError('Ham degerlendirme annotation kopyasi degismis')
        raw=raw_annotation(annotation.read_text(encoding='utf-8-sig'))
        integral=ignore_integral(raw,p['width'],p['height'])
        baseline=xyxy_to_det(p['baseline']);hycado=xyxy_to_det(p['hycado'])
        images.append({'gt':prepare_groundtruth(raw,integral),
                       'baseline':outside_ignore(baseline,integral),'hycado':outside_ignore(hycado,integral)})
        raw_cells.append(raw);base_cells.append(baseline);hybrid_cells.append(hycado);sizes.append([p['height'],p['width']])
        del integral
        if i==1 or i%50==0 or i==len(index):print(f'AP veri hazirlama: {i}/{len(index)}',flush=True)
    # Optional MATLAB reference runner consumes these original (not Python-filtered) arrays.
    from scipy.io import savemat
    def cells(values):
        arr=np.empty((len(values),1),dtype=object)
        for i,v in enumerate(values):arr[i,0]=v
        return arr
    savemat(out/'official_reference_inputs.mat',{'rawgt':cells(raw_cells),'baseline':cells(base_cells),
            'hycado':cells(hybrid_cells),'sizes':np.asarray(sizes)},do_compression=True)
    metrics={variant:evaluate(images,variant) for variant in ['baseline','hycado']}
    metrics['delta_AP_percentage_points']=metrics['hycado']['AP']-metrics['baseline']['AP']
    metrics['official_val_used']=False;metrics['official_binary_parity_verified']=False
    write_json(out/'metrics.json',metrics)
    with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['variant','AP','AP50','AP75','AR1','AR10','AR100','AR500','class_macro_AP_separate_diagnostic'])
        writer.writeheader()
        for variant in ['baseline','hycado']:writer.writerow({'variant':variant,**{k:metrics[variant][k] for k in writer.fieldnames if k!='variant'}})
    return metrics


def bundle(out):
    target=out/'11_gelistirme_rapor.zip'
    fd,name=tempfile.mkstemp(dir=out,prefix='11_report_',suffix='.zip.tmp')
    import os
    os.close(fd)
    try:
        with zipfile.ZipFile(name,'w',zipfile.ZIP_DEFLATED) as z:
            for n in ['protocol.json','environment.json','result.json','progress.json','metrics.json','metrics.csv','prediction_index.json','11_matlab_reference_check.json']:
                if (out/n).exists():z.write(out/n,n)
            if (out/'console.log').exists():
                with (out/'console.log').open('rb') as f:
                    f.seek(max(0,(out/'console.log').stat().st_size-100000));z.writestr('console_tail.log',f.read())
        os.replace(name,target)
    finally:Path(name).unlink(missing_ok=True)
    return target


def run():
    from .__main__ import _Tee
    out=ROOT/'experiments/development_eval_v1_seed42';out.mkdir(parents=True,exist_ok=True)
    stage1=ROOT/'experiments/stage1_v1_seed42';source06=ROOT/'experiments/stage1_predictions_v1_seed42'
    cropdata=ROOT/'experiments/crop_dataset_v1_seed42';train=ROOT/'experiments/rtdetr_crop_v1_seed42'
    with training_lock(out/'.evaluation.lock'):
        with (out/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    print('11 GERCEK GELISTIRME TESPIT DEGERLENDIRMESI:',utc_now(),flush=True)
                    print('647 goruntu; 10682 crop; egitim yok; resmi VAL kullanilmaz.',flush=True)
                    with training_lock(stage1/'.training.lock'),training_lock(source06/'.prediction.lock'),training_lock(cropdata/'.crop_data.lock'),training_lock(train/'.training.lock'):
                        sources=validate_sources(stage1,source06)
                        cfg,weights,official=configuration(ROOT,sources);signature=fingerprint(cfg)
                        if (out/'protocol.json').exists() and fingerprint(read_json(out/'protocol.json'))!=signature:
                            raise ValueError('11 protokolu degismis; onbellek karistirilmadi')
                        write_json(out/'protocol.json',cfg)
                        write_json(out/'result.json',{'status':'RUNNING','started_utc':utc_now()})
                        # Clear stale aggregate results before recomputation.
                        for n in ['metrics.json','metrics.csv']:(out/n).unlink(missing_ok=True)
                        write_json(out/'environment.json',system_info(include_torch=True))
                        parents=load_parents(ROOT,sources,cfg)
                        index=infer(ROOT,out,sources,cfg,weights,official,parents)
                        if sha256(weights)!=W80:raise ValueError('Egitim agirligi kosu sirasinda degismis')
                        write_json(out/'progress.json',{'status':'EVALUATING','images':len(index)})
                        metrics=measure(out,index,signature)
                        result={'status':'COMPLETED','development_images':647,'crops':10682,'training_started':False,
                                'official_val_used':False,'source_checkpoint_unchanged':True,
                                'official_binary_parity_verified':False,'metrics_are_development_diagnostics':True,
                                'baseline_AP':metrics['baseline']['AP'],'hycado_AP':metrics['hycado']['AP'],
                                'delta_AP_percentage_points':metrics['delta_AP_percentage_points']}
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();result={'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',
                        'error_type':type(exc).__name__,'error':str(exc),'resume':'Rerun 11; completed source-image caches are reused'}
                write_json(out/'result.json',result);write_json(out/'progress.json',result)
                print('SONUC:',result['status'],flush=True)
        print('GONDERILECEK RAPOR:',bundle(out),flush=True)
        return result['status']=='COMPLETED'


def main():
    if '--report-only' in sys.argv:
        print(bundle(ROOT/'experiments/development_eval_v1_seed42'),flush=True);return 0
    return 0 if run() else 1


if __name__=='__main__':raise SystemExit(main())
