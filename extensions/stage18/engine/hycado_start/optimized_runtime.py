"""Fresh batch-one runtime with the exactly equivalent 14 CPU NMS.

No saved experiment code is edited. Separate, explicitly synchronized crop
diagnostics are excluded from the end-to-end latency statistics.
"""
from __future__ import annotations
import contextlib
import csv
import math
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from .common import ROOT, sha256, write_json, system_info, utc_now
from .stage1 import fingerprint, training_lock
from .crop_loader import safe_path, read_json
from . import runtime_benchmark as r, nms_equivalence as n

OUT = 'optimized_runtime_v1_seed42'
FILES14 = {
    'protocol.json': '31374523745ce3237ecc67136b806f98859ee7f19241da727695c273f7d9077c',
    'result.json': '302a87ad3277096b302bea33b27c38093b8b0c39d3c0f9aa6e12d57a61a47f52',
    'summary.json': 'd7f008384919574b2b21270e91912442afbf9a0b416d4b8ade9dabd45c976650'}
CODE14 = '0ce3781e35d1d2dfd1481b276aecc3c154aa2f40133ece6339be6481f0658d88'
PROFILE_COMPONENTS = ['decode_rgb_ms', 'extract_ms', 'resize_ms',
                      'transfer_normalize_ms', 'forward_ms', 'query_to_cpu_ms', 'map_scores_ms']


def sources(root):
    cfg12, base, w1, w2, official = r.source(root)
    d13 = root/'experiments'/r.OUT; d14 = root/'experiments'/n.OUT
    c13 = read_json(d13/'protocol.json'); c14 = read_json(d14/'protocol.json')
    if sha256(r.__file__) != c13['code_sha256'] or sha256(n.__file__) != CODE14:
        raise ValueError('Tamamlanan 13/14 kodu degismis')
    for name, digest in FILES14.items():
        if sha256(d14/name) != digest: raise ValueError('Dogrulanmis 14 raporu farkli: '+name)
    if c14['source13_protocol_sha256'] != fingerprint(c13): raise ValueError('13/14 kaynak baglantisi farkli')
    trials = [n.validate_trial(read_json(d14/f'trial_{i+1}.json'), c14, i) for i in range(3)]
    s14 = n.summary(trials)
    if (read_json(d14/'summary.json') != s14 or not s14['implementation_candidate_recommended'] or
            read_json(d13/'result.json').get('status') != 'COMPLETED'):
        raise ValueError('Basarili 13/14 sonuclari gerekli')
    p13 = [r.validate_phase(read_json(d13/f'pass_{p+1}_{v}.json'), c13, p, v) for p, v in r.PHASES]
    if r.summarize(p13) != read_json(d13/'summary.json'): raise ValueError('13 sure ozeti farkli')
    upstream = {}
    for folder, names in [(d13, ['protocol.json', 'result.json', 'summary.json'] +
                                [f'pass_{p+1}_{v}.json' for p, v in r.PHASES]),
                          (d14, list(FILES14) + [f'trial_{i+1}.json' for i in range(3)])]:
        for name in names: upstream[(folder/name).relative_to(root).as_posix()] = sha256(folder/name)
    return cfg12, base, w1, w2, official, upstream


def run_image(record, path, predict1, predict2, decision, hybrid, synchronize, *,
              before_crops=lambda: None, clock=time.perf_counter):
    """Same 13 measured path, replacing only final NMS with verified 14."""
    from PIL import Image
    synchronize(); start = clock()
    t = clock(); width, height, baseline, dropped = predict1(path); synchronize(); stage1_ms = (clock()-t)*1000
    if (width, height) != (record['width'], record['height']): raise ValueError('Goruntu boyutu farkli')
    geometry_ms = crop_ms = merge_ms = 0.; rects = []; result = baseline
    if hybrid:
        t = clock()
        _, clusters = r.crop_data.select_clusters({'detections': baseline, 'width': width, 'height': height}, decision)
        rects = [list(c['rect']) for c in clusters]; geometry_ms = (clock()-t)*1000
        t = clock(); candidates = list(baseline)
        if rects:
            before_crops()
            with Image.open(path) as image:
                if image.size != (width, height): raise ValueError('RGB kaynak boyutu farkli')
                rgb = image.convert('RGB')
            for rect in rects:
                query = predict2(np.array(rgb.crop(tuple(rect))))
                rows = r.old.map_queries(query, {'source_crop_xyxy': rect})
                for row in rows: row[4] *= .25
                candidates.extend(rows)
        synchronize(); crop_ms = (clock()-t)*1000
        t = clock(); result = n.fast_nms(candidates, width, height); merge_ms = (clock()-t)*1000
    synchronize(); total_ms = (clock()-start)*1000
    return {'stage1_ms': stage1_ms, 'geometry_ms': geometry_ms, 'crop_pipeline_ms': crop_ms,
            'merge_ms': merge_ms, 'total_ms': total_ms, 'crop_count': len(rects)}, baseline, result, rects


def profile_order(records):
    return sorted(records, key=lambda x: fingerprint({'15_profile': x['image']}))[:32]


def profile_queries(model, rgb, device='cuda', synchronize=None, clock=time.perf_counter):
    """Diagnostic wall times; extra sync points deliberately excluded from benchmark."""
    import torch
    if synchronize is None: synchronize = torch.cuda.synchronize if str(device).startswith('cuda') else lambda: None
    synchronize(); t = clock(); array = r.preprocess_array(rgb); resize = (clock()-t)*1000
    t = clock(); x = torch.from_numpy(array).unsqueeze(0).to(device, dtype=torch.float32)/255.
    synchronize(); transfer = (clock()-t)*1000
    t = clock()
    with torch.inference_mode(), torch.autocast(str(device).split(':')[0], dtype=torch.float16, enabled=str(device).startswith('cuda')):
        output = model.predict(x)
    prediction = output[0] if isinstance(output, (tuple, list)) else output
    if tuple(prediction.shape) != (1, 300, 14): raise ValueError('300x14 crop cikti gerekli')
    synchronize(); forward = (clock()-t)*1000
    t = clock(); query = prediction[0].float().cpu().numpy().copy(); synchronize(); pull = (clock()-t)*1000
    if not np.isfinite(query).all(): raise ValueError('Crop ciktisi sonlu degil')
    return query, {'resize_ms': resize, 'transfer_normalize_ms': transfer, 'forward_ms': forward, 'query_to_cpu_ms': pull}


def profile_image(path, expected, raw12, predict, clock=time.perf_counter):
    from PIL import Image
    start = clock(); t = clock()
    with Image.open(path) as image:
        if image.size != (expected['width'], expected['height']): raise ValueError('RGB kaynak boyutu farkli')
        rgb = image.convert('RGB')
    times = {k: 0. for k in PROFILE_COMPONENTS}; times['decode_rgb_ms'] = (clock()-t)*1000
    rects = expected['crop_rects']; candidates = list(expected['baseline'])
    if raw12.dtype != np.float32 or raw12.shape != (len(rects), 300, 14) or not np.isfinite(raw12).all():
        raise ValueError('12 ham sorgu bicimi farkli')
    for i, rect in enumerate(rects):
        t = clock(); crop = np.array(rgb.crop(tuple(rect))); times['extract_ms'] += (clock()-t)*1000
        query, detail = predict(crop)
        for k, value in detail.items(): times[k] += value
        # Equality check is outside each component timer, but inside diagnostic wall time.
        if not np.array_equal(query, raw12[i]): raise ValueError('Tani ham sorgusu 12 ile birebir farkli')
        t = clock(); rows = r.old.map_queries(query, {'source_crop_xyxy': rect})
        for row in rows: row[4] *= .25
        candidates.extend(rows); times['map_scores_ms'] += (clock()-t)*1000
    times['diagnostic_wall_ms'] = (clock()-start)*1000
    if n.fast_nms(candidates, expected['width'], expected['height']) != expected['hybrid_a025']:
        raise ValueError('Tani ciktisi 12 ile birebir farkli')
    return {**times, 'crop_count': len(rects), 'raw_queries_exact12': True, 'final_output_exact12': True}


def validate_profile(payload, config):
    if (payload.get('content_sha256') != fingerprint({k:v for k,v in payload.items() if k!='content_sha256'}) or
            payload.get('protocol_sha256') != fingerprint(config) or payload.get('status') != 'COMPLETED'):
        raise ValueError('15 tani kaydi imzasi farkli')
    rows = payload['rows']
    if [x['source_image'] for x in rows] != [x['image'] for x in profile_order(config['records'])]:
        raise ValueError('15 tani goruntu kapsami/sirasi farkli')
    for row in rows:
        if not row['raw_queries_exact12'] or not row['final_output_exact12']: raise ValueError('Tani esligi basarisiz')
        if any(not math.isfinite(row[k]) or row[k] < 0 for k in PROFILE_COMPONENTS+['diagnostic_wall_ms']):
            raise ValueError('Gecersiz tani suresi')
        if row['diagnostic_wall_ms'] <= 0 or sum(row[k] for k in PROFILE_COMPONENTS) > row['diagnostic_wall_ms']+.02:
            raise ValueError('Tani bilesen sureleri farkli')
        if not isinstance(row['crop_count'], int) or row['crop_count'] < 0: raise ValueError('Gecersiz crop sayisi')
    if sum(x['crop_count'] for x in rows) != config['profile_expected_crops']: raise ValueError('Tani crop kapsami farkli')
    return payload


def worker(job_path):
    job = read_json(job_path); output = Path(job['output']); config = read_json(output/'protocol.json')
    diagnostic = job['kind'] == 'profile'; pass_id = job.get('pass_id', 0); variant = job.get('variant', 'hybrid_a025')
    target = output/('crop_profile.json' if diagnostic else f'pass_{pass_id+1}_{variant}.json')
    try:
        cfg, base, w1, w2, official, upstream = sources(ROOT)
        if (config['code_sha256'] != sha256(__file__) or fingerprint(config) != job['protocol_sha256'] or
                upstream != config['upstream_metadata_sha256']): raise ValueError('15 kod/protokol/kaynak kimligi farkli')
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA gerekli')
        env = system_info(include_torch=True)
        if env['gpu'] != config['gpu'] or env['nvidia_smi'].split(',')[-1].strip() != config['driver']:
            raise ValueError('GPU/surucu degismis; fazlar karistirilmadi')
        loading = time.perf_counter(); hybrid = diagnostic or variant == 'hybrid_a025'
        predict1 = r.proposals.make_predictor(r.proposals.load_ema(w1, r.val.P1), {'inference':cfg['stage1_inference']}, output)
        state1 = r.backend_state(); model = None; state2 = None
        if hybrid: model = r.crop_capture.load_model(w2, official, 'cuda'); state2 = r.backend_state()
        load_seconds = time.perf_counter()-loading
        predict2 = lambda rgb: r.direct_queries(model, rgb)
        before = lambda: r.restore_backend(state2) if state2 else None
        order = profile_order(config['records']) if diagnostic else r.pass_order(config['records'], pass_id)
        prepared = ROOT/'data/prepared'
        for rec in order:
            if sha256(safe_path(prepared, rec['image'])) != rec['image_sha256']: raise ValueError('Kaynak goruntu degismis')
        index = {x['source_image']:x for x in read_json(base/'prediction_index.json')['records']}
        warm = time.perf_counter()
        for rec in order[:config['warmup_images']]:
            r.restore_backend(state1)
            run_image(rec, safe_path(prepared, rec['image']), predict1, predict2, cfg['crop_policy'], hybrid,
                      torch.cuda.synchronize, before_crops=before)
        warm_seconds = time.perf_counter()-warm; rows = []; started = time.perf_counter()
        for i, rec in enumerate(order, 1):
            expected = r.references(base, rec, cfg, index)
            path = safe_path(prepared, rec['image'])
            if diagnostic:
                archive = safe_path(base, index[rec['image']]['path']).parent/'raw_crop_queries.npz'
                if sha256(archive) != index[rec['image']]['raw_archive_sha256'] or sha256(archive) != expected['raw_archive_sha256']:
                    raise ValueError('12 ham sorgu arsivi degismis')
                with np.load(archive, allow_pickle=False) as data:
                    raw = data['raw_queries']
                    if data['crop_rects'].tolist() != expected['crop_rects']: raise ValueError('12 ham crop geometrisi farkli')
                before(); torch.cuda.synchronize()
                times = profile_image(path, expected, raw, lambda rgb: profile_queries(model, rgb))
            else:
                r.restore_backend(state1); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                times, baseline, result, rects = run_image(rec, path, predict1, predict2, cfg['crop_policy'], hybrid,
                                                         torch.cuda.synchronize, before_crops=before)
                times.update({'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                              'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2})
                if (baseline != expected['baseline'] or result != expected[variant] or
                        (hybrid and rects != expected['crop_rects'])):
                    raise ValueError('15 yeni GPU ciktisi 12 ile birebir farkli: '+rec['image'])
                times['exact_original12_output'] = True
            rows.append({'source_image':rec['image'], **times})
            if i==1 or i%10==0 or i==len(order):
                label = 'CROP TANI' if diagnostic else f'tur {pass_id+1}/3 {variant}'
                print(f'15 {label}: {i}/{len(order)}; gecen={(time.perf_counter()-started)/60:.1f} dk', flush=True)
                write_json(output/'progress.json', {'status':'MEASURING', 'kind':job['kind'], 'pass_id':pass_id,
                                                   'variant':variant, 'images_done':i, 'images_total':len(order)})
        for rec in order:
            if sha256(safe_path(prepared, rec['image'])) != rec['image_sha256']: raise ValueError('Goruntu olcum sirasinda degismis')
        if sha256(w1) != cfg['stage1_weights_sha256'] or sha256(w2) != cfg['crop_weights_sha256']:
            raise ValueError('Agirlik olcum sirasinda degismis')
        body = {'status':'COMPLETED', 'protocol_sha256':fingerprint(config), 'pass_id':pass_id, 'variant':variant,
                'created_utc':utc_now(), 'environment':env, 'model_load_seconds_excluded':load_seconds,
                'warmup_seconds_excluded':warm_seconds, 'stage1_backend':state1, 'crop_backend':state2, 'rows':rows}
        body['content_sha256'] = fingerprint(body)
        if diagnostic: validate_profile(body, config)
        else: r.validate_phase(body, config, pass_id, variant)
        write_json(target, body); return 0
    except (Exception, KeyboardInterrupt) as exc:
        traceback.print_exc(); write_json(output/'phase_error.json', {'status':'ERROR', 'kind':job['kind'],
                           'pass_id':pass_id, 'variant':variant, 'error':str(exc), 'created_utc':utc_now()}); return 1


def summary(phases, profile, historical13):
    result = r.summarize(phases)
    result['timing_implementation'] = '13 measured path, native RGB crops, verified 14 CPU float64 NMS; exact 12 output required'
    result['new_accuracy_metrics_computed'] = False
    result['nms_implementation_only_change'] = True
    previous = historical13['variants']['hybrid_a025']['mean_ms']
    current = result['variants']['hybrid_a025']['mean_ms']
    result['historical13_context'] = {'old_hybrid_mean_ms':previous, 'new_hybrid_mean_ms':current,
          'old_to_new_ratio':previous/current, 'note':'different runs; historical comparison, not paired old/new full-pipeline measurement'}
    rows = profile['rows']; components = {k:float(np.mean([x[k] for x in rows])) for k in PROFILE_COMPONENTS}
    total = sum(components.values()); crops = sum(x['crop_count'] for x in rows)
    result['crop_diagnostics'] = {'images':len(rows), 'crops':crops, 'component_mean_ms_per_image':components,
          'component_mean_ms_per_crop':{k:float(sum(x[k] for x in rows)/crops) for k in PROFILE_COMPONENTS},
          'share_of_summed_components_percent':{k:100*v/total for k,v in components.items()},
          'scope':'separate deterministic 32-image subset; cached frozen geometry; both models resident; extra CUDA synchronization',
          'excluded_from_end_to_end_statistics':True, 'raw_queries_and_final_output_exact12':True,
          'note':'decode is per source image; diagnostic wall time includes equality checks and gaps; synchronized components do not estimate unsynchronized overlap'}
    return result


def small_backup(root, output, report=False):
    import os, shutil, tempfile
    target = root.parent/'HyCADO_Yedekleri/15_Hizlandirilmis'
    files = [root/'hycado_start/optimized_runtime.py']
    files += [root/name for name in ['15_HIZLANDIRILMIS_OLCUM.cmd','15_RAPOR_AL.cmd','15_ONCE_OKUYUN.md','SHA256SUMS_15.txt'] if (root/name).exists()]
    if report: files += [p for p in output.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.zip']]
    for path in files:
        digest = sha256(path); dest = target/digest[:12]/path.name; dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or sha256(dest) != digest:
            fd, temporary = tempfile.mkstemp(dir=dest.parent, prefix=path.name+'_', suffix='.tmp'); os.close(fd)
            try:
                shutil.copy2(path,temporary)
                if sha256(temporary) != digest: raise ValueError('15 kucuk yedek kopyasi farkli')
                os.replace(temporary,dest)
            finally: Path(temporary).unlink(missing_ok=True)
        if sha256(dest) != digest: raise ValueError('15 kucuk yedek dogrulanamadi')
    return str(target)


def bundle(output):
    import os, tempfile, zipfile
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=output, prefix='15_', suffix='.zip.tmp'); os.close(fd)
    target = output/'15_hizlandirilmis_rapor.zip'
    try:
        with zipfile.ZipFile(name, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.iterdir()):
                if p.is_file() and p.suffix in ['.json','.csv']: z.write(p, p.name)
            if (output/'console.log').exists():
                with (output/'console.log').open('rb') as f:
                    f.seek(max(0,(output/'console.log').stat().st_size-80000)); z.writestr('console_tail.log', f.read())
        os.replace(name, target)
    finally: Path(name).unlink(missing_ok=True)
    return target


def run():
    from .__main__ import _Tee
    output = ROOT/'experiments'/OUT; output.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        locks = [ROOT/'.backup.lock'] + [ROOT/'experiments'/folder/name for folder,name in
            [(OUT,'.benchmark.lock'), (r.OUT,'.benchmark.lock'), (n.OUT,'.analysis.lock'), (r.val.OUT,'.evaluation.lock'),
             ('stage1_v1_seed42','.training.lock'), ('rtdetr_crop_v1_seed42','.training.lock')]]
        for lock in locks: stack.enter_context(training_lock(lock))
        with (output/'console.log').open('a', encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)), contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    print('15 HIZLANDIRILMIS SABIT CIKARIM:', utc_now(), flush=True)
                    cfg12, base, w1, w2, official, upstream = sources(ROOT)
                    env = system_info(include_torch=True)
                    if not env.get('cuda_available'): raise RuntimeError('CUDA gerekli')
                    index = read_json(base/'prediction_index.json')['records']
                    profile_set = {x['image'] for x in profile_order(cfg12['records'])}
                    config = {'id':'hycado_optimized_runtime_v1', 'date':'2026-10-30', 'source12_protocol_sha256':r.P12,
                        'source12_metadata_sha256':r.FILES12, 'upstream_metadata_sha256':upstream, 'records':cfg12['records'],
                        'expected_crop_count':10057, 'passes':3, 'warmup_images':5, 'phase_order':[[p,v] for p,v in r.PHASES],
                        'image_order':'unchanged 13 SHA256 {13_order:pass_id,image:source_name}, paired baseline/hybrid',
                        'nms':'14 exactly equivalent C++ CPU float64, unique canonical rank priorities, global first1000, IoU>.7',
                        'physical_inference_batch':1, 'crop_score_multiplier':.25, 'methods_unchanged':True,
                        'training_started':False, 'parameter_search':False, 'new_accuracy_metrics_computed':False,
                        'fresh_process_per_condition_pass':True, 'accuracy_gate':'all measured outputs/rectangles exactly 12',
                        'timing_scope':'13 warmed sequential path-to-CPU detections with CUDA synchronized boundaries; verified 14 final NMS',
                        'excluded':['model load','warmup','source validation','reference checks','report I/O','separate crop diagnostics'],
                        'memory_scope':'torch allocated/reserved including weights, warm allocator cache; fresh process per condition/pass',
                        'profile_images':32, 'profile_expected_crops':sum(x['crops'] for x in index if x['source_image'] in profile_set),
                        'profile_order':'first32 SHA256 {15_profile:source_name}, fixed without GT or scores',
                        'profile_scope':'separate extra synchronization; cached 12 rectangles; not end-to-end benchmark',
                        'gpu':env['gpu'], 'driver':env['nvidia_smi'].split(',')[-1].strip(),
                        'runtime_versions':cfg12['runtime_versions'], 'code_sha256':sha256(__file__)}
                    if (output/'protocol.json').exists() and read_json(output/'protocol.json') != config:
                        raise ValueError('15 protokolu/ortami/kodu degismis; fazlar karistirilmadi')
                    write_json(output/'protocol.json',config); write_json(output/'result.json',{'status':'RUNNING'})
                    write_json(output/'environment.json',env)
                    for name in ['summary.json','summary.csv']: (output/name).unlink(missing_ok=True)
                    backup_path = small_backup(ROOT,output)
                    immutable = {str(base/name):sha256(base/name) for name in r.FILES12}
                    immutable.update({str(p):sha256(p) for p in [w1,w2,official]})
                    immutable.update({str(ROOT/name):digest for name,digest in upstream.items()})
                    jobs = [{'kind':'benchmark','pass_id':p,'variant':v} for p,v in r.PHASES]+[{'kind':'profile'}]
                    phases = []; profile = None
                    for task in jobs:
                        diag = task['kind']=='profile'
                        path = output/('crop_profile.json' if diag else f'pass_{task["pass_id"]+1}_{task["variant"]}.json')
                        if not path.exists():
                            (output/'phase_error.json').unlink(missing_ok=True)
                            job = {'output':str(output),'protocol_sha256':fingerprint(config),**task}
                            write_json(output/'worker_job.json',job)
                            print('YENI GPU OLCUMU:',task,flush=True)
                            code = r.child_process([sys.executable,'-X','faulthandler','-u','-m','hycado_start.optimized_runtime',
                                                    '--worker',str(output/'worker_job.json')],ROOT)
                            if code != 0: raise RuntimeError('15 olcum alt islemi durdu: '+str(code))
                        else: print('TAMAMLANAN FAZ KULLANILIYOR:',path.name,flush=True)
                        if diag: profile = validate_profile(read_json(path),config)
                        else: phases.append(r.validate_phase(read_json(path),config,task['pass_id'],task['variant']))
                    if immutable != {p:sha256(p) for p in immutable}: raise ValueError('Onceki sonuclar/agirliklar degismis')
                    s = summary(phases,profile,read_json(ROOT/'experiments'/r.OUT/'summary.json')); write_json(output/'summary.json',s)
                    with (output/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
                        keys = ['mean_ms','median_ms','p95_ms','throughput_images_per_second','peak_allocated_mib','peak_reserved_mib']
                        writer = csv.DictWriter(f,fieldnames=['variant']+keys); writer.writeheader()
                        for variant,values in s['variants'].items(): writer.writerow({'variant':variant,**{k:values[k] for k in keys}})
                    result = {'status':'COMPLETED','phases':6,'images_per_phase':548,'measured_image_calls':3288,
                        'profile_images':32,'all_measured_outputs_exact_original12':True,'profile_raw_queries_and_outputs_exact12':True,
                        'original_weights_and_results_unchanged':True,'training_started':False,'parameter_search':False,
                        'small_backup_directory':backup_path}
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc(); result = {'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR','error':str(exc),
                        'resume':'rerun 15; completed whole phases reused; interrupted phase measured again'}
                write_json(output/'result.json',result); write_json(output/'progress.json',result)
                print('SONUC:',result['status'],flush=True)
        report = bundle(output)
        try: small_backup(ROOT,output,True)
        except Exception as exc: print('15 RAPOR KUCUK YEDEGI ALINAMADI:',str(exc),flush=True)
        print('GONDERILECEK RAPOR:',report,flush=True)
        return result['status']=='COMPLETED'


def main():
    if '--worker' in sys.argv: return worker(Path(sys.argv[sys.argv.index('--worker')+1]))
    if '--report-only' in sys.argv: print(bundle(ROOT/'experiments'/OUT),flush=True); return 0
    return 0 if run() else 1


if __name__ == '__main__': raise SystemExit(main())
