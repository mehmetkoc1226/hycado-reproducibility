"""CPU candidate NMS implementation study; no changes to saved experiments.

Uses float64 CPU boxes and unique artificial priorities to preserve the exact
canonical tie order. All actual boxes/scores, thresholds and caps stay fixed.
"""
from __future__ import annotations
import contextlib
import csv
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from .common import ROOT, sha256, write_json, utc_now
from .stage1 import fingerprint, training_lock
from .crop_loader import safe_path, read_json
from . import runtime_benchmark as runtime
from . import dev_evaluation as old
from . import frozen_val as val

OUT = 'nms_equivalence_v1_seed42'


def fast_nms(rows, width, height, threshold=.7, limit=1000):
    import torch
    from torchvision.ops import nms
    clean, _ = old.canonical_detections(rows, width, height)
    a = np.asarray(clean, dtype=np.float64).reshape(-1, 6)
    selected = []
    for cls in range(10):
        indices = np.flatnonzero(a[:, 5] == cls)
        if not len(indices): continue
        boxes = torch.from_numpy(a[indices, :4].copy())  # float64, CPU
        # Strictly descending, exactly representable ranks remove score ties.
        # Real confidence values in a are never changed or thresholded here.
        priorities = torch.from_numpy(-indices.astype(np.float64))
        kept = nms(boxes, priorities, threshold).numpy()
        selected.extend(indices[kept].tolist())
    selected = np.array(sorted(selected)[:limit], dtype=np.int64)
    return a[selected].tolist()


def candidates(source, item, cfg):
    path = safe_path(source, item['path'])
    if sha256(path) != item['sha256']: raise ValueError('12 kaynak tahmini degismis')
    mapping = {r['image']: r for r in cfg['records']}; record = mapping[item['source_image']]
    p = val.checked_json(path, runtime.P12, record)
    archive = path.parent/'raw_crop_queries.npz'
    if sha256(archive) != item['raw_archive_sha256'] or p['raw_archive_sha256'] != item['raw_archive_sha256']:
        raise ValueError('12 ham sorgu kaydi degismis')
    with np.load(archive, allow_pickle=False) as data:
        raw = data['raw_queries']; rects = data['crop_rects']
    if (raw.shape != (p['crop_count'], 300, 14) or raw.dtype != np.float32 or not np.isfinite(raw).all() or
            rects.tolist() != p['crop_rects'] or len(rects) != item['crops']):
        raise ValueError('12 ham sorgu/geometri bicimi farkli')
    rows = list(p['baseline'])
    for query, rect in zip(raw, p['crop_rects']):
        crop_rows = old.map_queries(query, {'source_crop_xyxy': rect})
        for row in crop_rows: row[4] *= .25
        rows.extend(crop_rows)
    return rows, p


def trial_order(index, trial):
    return sorted(index['records'], key=lambda r: fingerprint({'14_trial': trial, 'image': r['source_image']}))


def validate_trial(payload, config, trial):
    if (payload.get('content_sha256') != fingerprint({k:v for k,v in payload.items() if k!='content_sha256'}) or
            payload.get('protocol_sha256') != fingerprint(config) or payload.get('trial') != trial or payload.get('status') != 'COMPLETED'):
        raise ValueError('14 tur kaydi kimligi farkli')
    rows = payload['rows']
    if len(rows) != 548 or {r['source_image'] for r in rows} != set(config['source_images']): raise ValueError('548 goruntu kapsami farkli')
    if any(not r['old_exact12'] or not r['cpp_exact12'] or not np.isfinite([r['old_ms'], r['cpp_ms']]).all() or min(r['old_ms'], r['cpp_ms']) <= 0 for r in rows):
        raise ValueError('14 eslik/sure denetimi basarisiz')
    return payload


def measure_trial(source, index, cfg12, output, config, trial):
    rows = []; started = time.perf_counter()
    for i, item in enumerate(trial_order(index, trial), 1):
        a, p = candidates(source, item, cfg12)
        methods = [('old', old.classwise_nms), ('cpp', fast_nms)]
        if (i+trial)%2: methods.reverse()
        results = {}; elapsed = {}
        for name, function in methods:
            start = time.perf_counter(); results[name] = function(a, p['width'], p['height']); elapsed[name] = (time.perf_counter()-start)*1000
        if results['old'] != p['hybrid_a025'] or results['cpp'] != p['hybrid_a025']:
            raise ValueError('Kutular/skorlar/sira 12 ile birebir farkli: '+item['source_image'])
        rows.append({'source_image': item['source_image'], 'candidate_count': len(a), 'old_ms': elapsed['old'],
                     'cpp_ms': elapsed['cpp'], 'old_exact12': True, 'cpp_exact12': True})
        if i==1 or i%10==0 or i==len(index['records']):
            print(f'14 CPU NMS tur {trial+1}/3: {i}/{len(index["records"])}; gecen={(time.perf_counter()-started)/60:.1f} dk', flush=True)
            write_json(output/'progress.json', {'status':'MEASURING', 'trial':trial, 'images_done':i})
    body = {'status':'COMPLETED', 'protocol_sha256':fingerprint(config), 'trial':trial, 'created_utc':utc_now(), 'rows':rows}
    body['content_sha256'] = fingerprint(body); validate_trial(body, config, trial)
    write_json(output/f'trial_{trial+1}.json', body)
    return body


def summary(trials):
    per_trial = []
    for p in trials:
        old_ms = float(np.mean([r['old_ms'] for r in p['rows']]))
        cpp_ms = float(np.mean([r['cpp_ms'] for r in p['rows']]))
        per_trial.append({'trial':p['trial'], 'old_mean_ms':old_ms, 'cpp_mean_ms':cpp_ms, 'speed_ratio':old_ms/cpp_ms})
    a = [r for p in trials for r in p['rows']]; result = {}
    for name in ['old', 'cpp']:
        values = np.array([r[name+'_ms'] for r in a])
        result[name] = {'mean_ms':float(values.mean()), 'median_ms':float(np.median(values)), 'p95_ms':float(np.percentile(values,95))}
    result.update({'per_trial':per_trial, 'all_outputs_exact12':True,
                   'implementation_candidate_recommended':bool(np.median([p['speed_ratio'] for p in per_trial])>=1.03 and all(p['speed_ratio']>=1 for p in per_trial)),
                   'automatic_installation':False, 'scope':'CPU final NMS only, input candidates in memory; no end-to-end speed claim',
                   'training_started':False, 'inference_started':False, 'parameter_search':False, 'new_AP_computed':False})
    return result


def backup(root, output, report=False):
    target = root.parent/'HyCADO_Yedekleri/14_NMS'
    import shutil
    files = [root/'hycado_start/nms_equivalence.py']
    files += [root/n for n in ['14_CPU_NMS_KONTROL.cmd', '14_RAPOR_AL.cmd', '14_ONCE_OKUYUN.md', 'SHA256SUMS_14.txt'] if (root/n).exists()]
    if report: files += [p for p in output.glob('*') if p.is_file() and p.suffix in ['.json','.csv','.zip']]
    for p in files:
        digest=sha256(p); dest=target/digest[:12]/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(p,dest)
        if sha256(dest)!=digest:raise ValueError('14 kucuk yedek dogrulanamadi')


def bundle(output):
    import zipfile, tempfile, os
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=output,prefix='14_',suffix='.zip.tmp');os.close(fd);target=output/'14_nms_rapor.zip'
    try:
        with zipfile.ZipFile(name,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.glob('*')):
                if p.is_file() and p.suffix in ['.json','.csv']:z.write(p,p.name)
            if (output/'console.log').exists():
                with (output/'console.log').open('rb') as f:
                    f.seek(max(0,(output/'console.log').stat().st_size-60000));z.writestr('console_tail.log',f.read())
        os.replace(name,target)
    finally:Path(name).unlink(missing_ok=True)
    return target


def run():
    from .__main__ import _Tee
    output=ROOT/'experiments'/OUT;output.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        for p in [ROOT/'.backup.lock', output/'.analysis.lock', ROOT/'experiments'/val.OUT/'.evaluation.lock',
                  ROOT/'experiments/stage1_v1_seed42/.training.lock', ROOT/'experiments/rtdetr_crop_v1_seed42/.training.lock']:
            stack.enter_context(training_lock(p))
        with (output/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)), contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    print('14 CPU NMS ESLIK VE HIZ:',utc_now(),flush=True)
                    print('Model/GPU cikarimi ve egitim yok. Eski sonuclar degistirilmez.',flush=True)
                    cfg12, source, *_ = runtime.source(ROOT)
                    cfg13=read_json(ROOT/'experiments'/runtime.OUT/'protocol.json')
                    if sha256(runtime.__file__)!=cfg13['code_sha256'] or read_json(ROOT/'experiments'/runtime.OUT/'result.json').get('status')!='COMPLETED':
                        raise ValueError('Tamamlanan sabit 13 kaydi/kodu gerekli')
                    index=read_json(source/'prediction_index.json')
                    config={'id':'hycado_cpu_nms_equivalence_v1','date':'2026-10-30','source12_protocol_sha256':runtime.P12,
                            'source13_protocol_sha256':fingerprint(cfg13),'source12_metadata_sha256':runtime.FILES12,
                            'source_images':[r['source_image'] for r in index['records']], 'trials':3,
                            'candidate_source':'12 float32 raw queries, unchanged mapping and raw score cutoff, crop score x.25',
                            'original_method':'canonical global classwise NMS IoU>.7, max1000',
                            'candidate_implementation':'torchvision C++ CPU NMS, float64 boxes, per-class unique canonical rank priorities, global first1000',
                            'implementation_only':True,'automatic_installation':False,'training_started':False,'inference_started':False,
                            'parameter_search':False,'code_sha256':sha256(__file__)}
                    if (output/'protocol.json').exists() and read_json(output/'protocol.json')!=config:raise ValueError('14 protokolu degismis')
                    write_json(output/'protocol.json',config);write_json(output/'result.json',{'status':'RUNNING'})
                    for name in ['summary.json','summary.csv']:(output/name).unlink(missing_ok=True)
                    backup(ROOT,output)
                    import torch,torchvision
                    torch.set_num_threads(2)
                    write_json(output/'environment.json',{'created_utc':utc_now(),'python':sys.version,'torch':torch.__version__,
                               'torchvision':torchvision.__version__,'numpy':np.__version__,'device':'CPU','torch_threads':2})
                    fast_nms([[0,0,10,10,.9,0],[1,1,11,11,.8,0]],20,20)
                    # Prime both implementations on the same first five real inputs.
                    for item in trial_order(index,0)[:5]:
                        a,p=candidates(source,item,cfg12)
                        if old.classwise_nms(a,p['width'],p['height'])!=fast_nms(a,p['width'],p['height']):raise ValueError('Isinmada NMS farki')
                    trials=[]
                    for trial in range(3):
                        path=output/f'trial_{trial+1}.json'
                        trials.append(validate_trial(read_json(path),config,trial) if path.exists() else measure_trial(source,index,cfg12,output,config,trial))
                    for name,digest in runtime.FILES12.items():
                        if sha256(source/name)!=digest:raise ValueError('12 kaynagi kosu sirasinda degismis')
                    s=summary(trials);write_json(output/'summary.json',s)
                    with (output/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
                        writer=csv.DictWriter(f,fieldnames=['implementation','mean_ms','median_ms','p95_ms']);writer.writeheader()
                        for name in ['old','cpp']:writer.writerow({'implementation':name,**s[name]})
                    result={'status':'COMPLETED','images_per_trial':548,'trials':3,'all_outputs_exact12':True,
                            'implementation_candidate_recommended':s['implementation_candidate_recommended'],
                            'automatic_installation':False,'training_started':False,'inference_started':False,'parameter_search':False}
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();result={'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR','error':str(exc),
                           'resume':'rerun 14; completed trials reused; partial trial measured again'}
                write_json(output/'result.json',result);write_json(output/'progress.json',result);print('SONUC:',result['status'],flush=True)
        path=bundle(output)
        try:backup(ROOT,output,report=True)
        except Exception as exc:print('14 RAPOR KUCUK YEDEGI ALINAMADI:',str(exc),flush=True)
        print('GONDERILECEK RAPOR:',path,flush=True)
        return result['status']=='COMPLETED'


def main():
    if '--report-only' in sys.argv:print(bundle(ROOT/'experiments'/OUT),flush=True);return 0
    return 0 if run() else 1


if __name__=='__main__':raise SystemExit(main())
