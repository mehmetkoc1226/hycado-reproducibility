"""Controlled batch-one runtime/memory study of the frozen 12 methods.

Fresh GPU inference, all 548 VAL images, 3 paired passes, fresh child process per
condition/pass. No GT, training, AP computation or parameter selection.
"""
from __future__ import annotations
import contextlib
import csv
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
import numpy as np
from .common import ROOT, sha256, write_json, system_info, utc_now
from .stage1 import fingerprint, training_lock
from .crop_loader import safe_path, read_json
from . import frozen_val as val, proposals, crop_data, crop_capture
from . import dev_evaluation as old

P12 = 'ae0ae6f46e530d4a2cdd3f699f490016a659a574e2fe3f7da0b3f6b947eafdd2'
FILES12 = {'protocol.json': 'bf406f50d8441fc66346be30615ebdb94b009625e5946e5fa59dd1e5427e686f',
           'result.json': 'a0f203b822ce5218419bd1215bbac4a9beb3f72f7cef0acbbf414fbd8d240222',
           'prediction_index.json': '7448cb758b2afbad87c3601221edd77a890be36e336460c66004b21c51abff17',
           'baseline_index.json': '64ab987d3551d9abe350377bf4a156cff39c1d390610c6f38fa4954a427a6aa0',
           'metrics.json': '331ca863f925dcb3145b1c32ec0d27ed243f58dabdad2c194cfe0e6b94d3876a'}
OUT = 'runtime_memory_v1_seed42'
PHASES = [(0, 'baseline'), (0, 'hybrid_a025'), (1, 'hybrid_a025'), (1, 'baseline'),
          (2, 'baseline'), (2, 'hybrid_a025')]
COMPONENTS = ['stage1_ms', 'geometry_ms', 'crop_pipeline_ms', 'merge_ms', 'total_ms']


def source(root):
    base = root / 'experiments' / val.OUT
    for name, digest in FILES12.items():
        if sha256(base / name) != digest: raise ValueError('Tamamlanan 12 kaydi degismis: ' + name)
    cfg = read_json(base / 'protocol.json'); result = read_json(base / 'result.json')
    if (fingerprint(cfg) != P12 or result.get('status') != 'COMPLETED' or result.get('images') != 548 or
            result.get('crops') != 10057 or result.get('backup_verified') is not True or cfg['crop_score_multiplier'] != .25):
        raise ValueError('Dogrulanmis tamamlanan 12 kosusu gerekli')
    if sha256(root/'data/prepared/manifest.json') != cfg['prepared_manifest_sha256']:
        raise ValueError('Orijinal veri manifesti degismis')
    for name, digest in cfg['code_sha256'].items():
        if Path(name).name != name or sha256(root / 'hycado_start' / name) != digest:
            raise ValueError('Sabit 12 kodu degismis: ' + name)
    for name, digest in cfg['crop_policy']['merge']['source_code_hashes'].items():
        if Path(name).name != name or sha256(root / 'hycado_start' / name) != digest:
            raise ValueError('Crop geometri kodu degismis')
    import importlib.metadata
    for name, version in cfg['runtime_versions'].items():
        if importlib.metadata.version(name) != version: raise ValueError('Paket surumu degismis: ' + name)
    index1 = read_json(root / 'experiments/stage1_v1_seed42/checkpoint_index.json')
    index2 = read_json(root / 'experiments/rtdetr_crop_v1_seed42/checkpoint_index.json')
    w1 = safe_path(root / 'experiments/stage1_v1_seed42', index1['latest']['path'])
    w2 = safe_path(root / 'experiments/rtdetr_crop_v1_seed42', index2['latest']['path'])
    official = root / 'work/fresh_official_pretrained/rtdetr-l.pt'
    if sha256(w1) != cfg['stage1_weights_sha256'] or sha256(w2) != cfg['crop_weights_sha256']:
        raise ValueError('80. tur EMA agirligi degismis')
    p10 = read_json(root / 'experiments/rtdetr_crop_v1_seed42/protocol.json')
    if sha256(official) != p10['initial_weights_sha256'] or fingerprint(p10) != old.P10:
        raise ValueError('Crop mimari kaynagi/protokolu farkli')
    for folder, signature in [('stage1_v1_seed42', val.P1), ('stage1_predictions_v1_seed42', old.P06),
                               ('rtdetr_crop_v1_seed42', old.P10)]:
        protocol = read_json(root / 'experiments' / folder / 'protocol.json')
        if fingerprint(protocol) != signature: raise ValueError('Kaynak egitim/cikarim protokolu degismis')
        for name, digest in protocol['code_hashes'].items():
            if Path(name).name != name or sha256(root/'hycado_start'/name) != digest:
                raise ValueError('Kaynak egitim/cikarim kodu degismis: '+name)
    return cfg, base, w1, w2, official


def pass_order(records, pass_id):
    return sorted(records, key=lambda r: fingerprint({'13_order': pass_id, 'image': r['image']}))


def preprocess_array(rgb):
    import cv2
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB uint8 crop gerekli')
    return cv2.resize(rgb, (960, 960), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1).copy()


def direct_queries(model, rgb, device='cuda'):
    import torch
    x = torch.from_numpy(preprocess_array(rgb)).unsqueeze(0).to(device, dtype=torch.float32) / 255.
    with torch.inference_mode(), torch.autocast(str(device).split(':')[0], dtype=torch.float16, enabled=str(device).startswith('cuda')):
        output = model.predict(x)
    prediction = output[0] if isinstance(output, (tuple, list)) else output
    if tuple(prediction.shape) != (1, 300, 14): raise ValueError('300x14 crop cikti gerekli')
    query = prediction[0].float().cpu().numpy().copy()
    if not np.isfinite(query).all(): raise ValueError('Crop ciktisi sonlu degil')
    return query


def run_image(record, path, predict1, predict2, decision, hybrid, synchronize, *, before_crops=lambda: None, clock=time.perf_counter):
    from PIL import Image
    synchronize(); start = clock()
    t = clock(); width, height, baseline, dropped = predict1(path); synchronize(); stage1_ms = (clock()-t)*1000
    if (width, height) != (record['width'], record['height']): raise ValueError('Goruntu boyutu farkli')
    geometry_ms = crop_ms = merge_ms = 0.; rects = []; result = baseline
    if hybrid:
        t = clock()
        _, clusters = crop_data.select_clusters({'detections': baseline, 'width': width, 'height': height}, decision)
        rects = [list(c['rect']) for c in clusters]; geometry_ms = (clock()-t)*1000
        t = clock(); candidates = list(baseline)
        if rects:
            before_crops()
            with Image.open(path) as image:
                if image.size != (width, height): raise ValueError('RGB kaynak boyutu farkli')
                rgb = image.convert('RGB')
            for rect in rects:
                crop = np.array(rgb.crop(tuple(rect)))
                query = predict2(crop)
                rows = old.map_queries(query, {'source_crop_xyxy': rect})
                for row in rows: row[4] *= .25
                candidates.extend(rows)
        synchronize(); crop_ms = (clock()-t)*1000
        t = clock(); result = old.classwise_nms(candidates, width, height); merge_ms = (clock()-t)*1000
    synchronize(); total_ms = (clock()-start)*1000
    return {'stage1_ms': stage1_ms, 'geometry_ms': geometry_ms, 'crop_pipeline_ms': crop_ms,
            'merge_ms': merge_ms, 'total_ms': total_ms, 'crop_count': len(rects)}, baseline, result, rects


def backend_state():
    import torch, cv2
    return {'benchmark': torch.backends.cudnn.benchmark, 'deterministic': torch.backends.cudnn.deterministic,
            'algorithms': torch.are_deterministic_algorithms_enabled(), 'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cv2_threads': cv2.getNumThreads()}


def restore_backend(state):
    import torch, cv2
    torch.backends.cudnn.benchmark = state['benchmark']; torch.backends.cudnn.deterministic = state['deterministic']
    torch.use_deterministic_algorithms(state['algorithms'], warn_only=state['warn_only']); cv2.setNumThreads(state['cv2_threads'])


def references(base, record, cfg, mapping):
    item = mapping[record['image']]; path = safe_path(base, item['path'])
    if sha256(path) != item['sha256']: raise ValueError('12 goruntu tahmini degismis')
    return val.checked_json(path, P12, record)


def validate_phase(payload, config, pass_id, variant):
    body = {k: v for k, v in payload.items() if k != 'content_sha256'}
    if (payload.get('content_sha256') != fingerprint(body) or payload.get('status') != 'COMPLETED' or
            payload.get('protocol_sha256') != fingerprint(config) or payload.get('pass_id') != pass_id or payload.get('variant') != variant):
        raise ValueError('Olcum fazi kimligi degismis')
    rows = payload['rows']; expected = [r['image'] for r in pass_order(config['records'], pass_id)]
    if [r['source_image'] for r in rows] != expected: raise ValueError('Olcum fazi goruntu kapsami/sirasi eksik')
    for row in rows:
        if row['exact_original12_output'] is not True: raise ValueError('12 sonucuyla uyusmayan olcum')
        if any(not math.isfinite(row[k]) or row[k] < 0 for k in COMPONENTS): raise ValueError('Gecersiz sure')
        if row['total_ms'] <= 0: raise ValueError('Pozitif toplam sure gerekli')
        if sum(row[k] for k in COMPONENTS if k != 'total_ms') > row['total_ms'] + .01:
            raise ValueError('Bilesen sureleri toplam sureyi asiyor')
        if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ['peak_allocated_mib', 'peak_reserved_mib']):
            raise ValueError('Gecersiz GPU bellek olcumu')
    if variant == 'hybrid_a025' and sum(r['crop_count'] for r in rows) != config['expected_crop_count']:
        raise ValueError('Olcum crop kapsami farkli')
    if variant == 'baseline' and any(r['crop_count'] != 0 for r in rows):
        raise ValueError('Baseline crop calistiramaz')
    return payload


def worker(job_path):
    job = read_json(job_path); output = Path(job['output']); config = read_json(output / 'protocol.json')
    signature = fingerprint(config); pass_id, variant = job['pass_id'], job['variant']
    target = output / f'pass_{pass_id+1}_{variant}.json'
    try:
        cfg, base, w1, w2, official = source(ROOT)
        if config['code_sha256'] != sha256(__file__) or signature != job['protocol_sha256']: raise ValueError('13 kod/protokol kimligi farkli')
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA gerekli')
        environment = system_info(include_torch=True)
        if environment['gpu'] != config['gpu'] or environment['nvidia_smi'].split(',')[-1].strip() != config['driver']:
            raise ValueError('GPU/surucu degismis; fazlar karistirilmadi')
        hybrid = variant == 'hybrid_a025'; loading = time.perf_counter()
        predict1 = proposals.make_predictor(proposals.load_ema(w1, val.P1), {'inference': cfg['stage1_inference']}, output)
        state1 = backend_state(); crop_model = None; state2 = None
        if hybrid:
            crop_model = crop_capture.load_model(w2, official, 'cuda'); state2 = backend_state()
        load_seconds = time.perf_counter()-loading
        predict2 = lambda rgb: direct_queries(crop_model, rgb)
        before_crops = lambda: restore_backend(state2) if state2 else None
        order = pass_order(config['records'], pass_id); prepared = ROOT / 'data/prepared'
        # Prime both conditions with the same verified image files outside timed regions.
        for r in order:
            if sha256(safe_path(prepared, r['image'])) != r['image_sha256']: raise ValueError('Kaynak goruntu degismis')
        mapping = {r['source_image']: r for r in read_json(base / 'prediction_index.json')['records']}
        warm_start = time.perf_counter()
        for r in order[:config['warmup_images']]:
            restore_backend(state1)
            run_image(r, safe_path(prepared, r['image']), predict1, predict2, cfg['crop_policy'], hybrid,
                      torch.cuda.synchronize, before_crops=before_crops)
        warmup_seconds = time.perf_counter()-warm_start; torch.cuda.synchronize(); rows = []
        started = time.perf_counter()
        for i, r in enumerate(order, 1):
            expected = references(base, r, cfg, mapping)  # read/hash excluded from timer
            restore_backend(state1); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            times, baseline, result, rects = run_image(r, safe_path(prepared, r['image']), predict1, predict2,
                   cfg['crop_policy'], hybrid, torch.cuda.synchronize, before_crops=before_crops)
            memory = {'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
                      'peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2}
            exact = baseline == expected['baseline'] and result == expected[variant]
            if hybrid: exact = exact and rects == expected['crop_rects']
            if not exact:
                raise ValueError('Yeni zamanlama ciktisi 12 ile birebir farkli: '+r['image']+'; dogruluk/sure karistirilmadi')
            rows.append({'source_image': r['image'], **times, **memory, 'exact_original12_output': True})
            if i == 1 or i % 10 == 0 or i == len(order):
                elapsed = time.perf_counter()-started
                print(f'13 tur {pass_id+1}/3 {variant}: {i}/{len(order)}; gecen={elapsed/60:.1f} dk', flush=True)
                write_json(output / 'phase_progress.json', {'status': 'MEASURING', 'pass_id': pass_id, 'variant': variant,
                           'images_done': i, 'images_total': len(order), 'elapsed_seconds': elapsed})
        for r in order:
            if sha256(safe_path(prepared, r['image'])) != r['image_sha256']: raise ValueError('Goruntu olcum sirasinda degismis')
        if sha256(w1) != cfg['stage1_weights_sha256'] or sha256(w2) != cfg['crop_weights_sha256']:
            raise ValueError('Agirlik olcum sirasinda degismis')
        body = {'status': 'COMPLETED', 'protocol_sha256': signature, 'pass_id': pass_id, 'variant': variant,
                'created_utc': utc_now(), 'environment': environment, 'model_load_seconds_excluded': load_seconds,
                'warmup_seconds_excluded': warmup_seconds, 'stage1_backend': state1, 'crop_backend': state2, 'rows': rows}
        body['content_sha256'] = fingerprint(body); validate_phase(body, config, pass_id, variant); write_json(target, body)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        traceback.print_exc(); write_json(output / 'phase_error.json', {'status': 'ERROR', 'pass_id': pass_id, 'variant': variant,
                        'error': str(exc), 'created_utc': utc_now()}); return 1


def summarize(phases):
    result = {'variants': {}, 'paired_passes': []}
    for name in ['baseline', 'hybrid_a025']:
        subset = [p for p in phases if p['variant'] == name]; rows = [r for p in subset for r in p['rows']]
        a = np.array([r['total_ms'] for r in rows]); total_s = float(a.sum()/1000)
        result['variants'][name] = {'images_per_pass': len(subset[0]['rows']), 'passes': len(subset),
            'measured_image_calls': len(rows), 'mean_ms': float(a.mean()), 'median_ms': float(np.median(a)),
            'p95_ms': float(np.percentile(a, 95)), 'measured_seconds': total_s,
            'throughput_images_per_second': len(rows)/total_s,
            'peak_allocated_mib': max(r['peak_allocated_mib'] for r in rows),
            'peak_reserved_mib': max(r['peak_reserved_mib'] for r in rows),
            'component_mean_ms': {k: float(np.mean([r[k] for r in rows])) for k in COMPONENTS},
            'crops_per_pass': sum(r['crop_count'] for r in rows)/len(subset)}
    for pass_id in sorted({p['pass_id'] for p in phases}):
        means = {p['variant']: float(np.mean([r['total_ms'] for r in p['rows']])) for p in phases if p['pass_id'] == pass_id}
        result['paired_passes'].append({'pass_id': pass_id, **means,
                                       'hybrid_to_baseline_latency_ratio': means['hybrid_a025']/means['baseline']})
    result['hybrid_to_baseline_latency_ratio'] = result['variants']['hybrid_a025']['mean_ms']/result['variants']['baseline']['mean_ms']
    result['scope'] = 'warmed batch-one sequential path-to-CPU-detections; OS file cache primed; model load and output artifact I/O excluded'
    result['reserved_memory_note'] = 'includes resident weights and allocator cache after warmup; baseline and hybrid measured in separate fresh processes'
    result['timing_implementation'] = 'in-memory native RGB crops; no temporary PNG files/hashes in measured regions; identical 12 output required'
    result['training_started'] = False; result['parameter_search'] = False; result['new_accuracy_metrics_computed'] = False
    return result


def small_backup(root, output, *, report=False):
    """Small code/report copies only; no scan/copy of the original project tree."""
    target = root.parent / 'HyCADO_Yedekleri' / '13_Hiz_Bellek'
    files = [root / 'hycado_start/runtime_benchmark.py']
    for name in ['13_HIZ_BELLEK_OLC.cmd', '13_RAPOR_AL.cmd', '13_ONCE_OKUYUN.md', 'SHA256SUMS_13.txt']:
        if (root/name).exists(): files.append(root/name)
    if report: files += [p for p in output.glob('*') if p.is_file() and p.suffix in ['.json', '.csv', '.zip']]
    for source_path in files:
        digest = sha256(source_path); destination = target / digest[:12] / source_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists(): shutil.copy2(source_path, destination)
        if sha256(destination) != digest: raise ValueError('13 kucuk yedek kopyasi farkli')
    return str(target)


def bundle(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output, prefix='13_', suffix='.zip.tmp'); os.close(fd)
    target = output/'13_hiz_bellek_rapor.zip'
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.glob('*.json')): z.write(p, p.name)
            for p in sorted(output.glob('*.csv')): z.write(p, p.name)
            if (output/'console.log').exists():
                with (output/'console.log').open('rb') as f:
                    f.seek(max(0, (output/'console.log').stat().st_size-80000)); z.writestr('console_tail.log', f.read())
        os.replace(temporary, target)
    finally: Path(temporary).unlink(missing_ok=True)
    return target


def child_process(command, cwd):
    child = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding='utf-8', errors='replace', bufsize=1)
    try:
        for line in child.stdout: print(line, end='', flush=True)
        return child.wait()
    except BaseException:
        child.terminate()
        try: child.wait(timeout=10)
        except subprocess.TimeoutExpired: child.kill(); child.wait()
        raise
    finally:
        child.stdout.close()


def run():
    from .__main__ import _Tee
    output = ROOT/'experiments'/OUT; output.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        stack.enter_context(training_lock(ROOT/'.backup.lock'))
        for folder, name in [(OUT, '.benchmark.lock'), (val.OUT, '.evaluation.lock'), ('stage1_v1_seed42', '.training.lock'),
                             ('rtdetr_crop_v1_seed42', '.training.lock')]:
            stack.enter_context(training_lock(ROOT/'experiments'/folder/name))
        with (output/'console.log').open('a', encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
                try:
                    cfg12, base, w1, w2, official = source(ROOT)
                    environment = system_info(include_torch=True)
                    if not environment.get('cuda_available'): raise RuntimeError('CUDA gerekli')
                    config = {'id': 'hycado_runtime_memory_v1', 'date': '2026-10-30', 'source12_protocol_sha256': P12,
                        'source12_metadata_sha256': FILES12, 'records': cfg12['records'], 'expected_crop_count': 10057,
                        'passes': 3, 'warmup_images': 5, 'phase_order': [[p, n] for p, n in PHASES],
                        'image_order': 'SHA256 canonical JSON {13_order:pass_id,image:source_name}; same within each paired pass',
                        'crop_score_multiplier': .25, 'methods_unchanged': True, 'training_started': False, 'parameter_search': False,
                        'new_accuracy_metrics_computed': False, 'fresh_process_per_condition_pass': True,
                        'accuracy_gate': 'every measured image output and crop rectangles exactly equal frozen 12',
                        'timing_scope': 'image path to CPU final detections, warm OS cache, CUDA synchronized wall time; native RGB crops in memory',
                        'excluded': ['model load', 'warmup', 'source validation', 'reference checks', 'temporary PNG serialization/hash', 'report writing'],
                        'memory_scope': 'torch CUDA allocated/reserved including resident model weights; reserved includes warm allocator cache',
                        'gpu': environment['gpu'], 'driver': environment['nvidia_smi'].split(',')[-1].strip(),
                        'runtime_versions': cfg12['runtime_versions'], 'code_sha256': sha256(__file__)}
                    if (output/'protocol.json').exists() and read_json(output/'protocol.json') != config:
                        raise ValueError('13 protokolu/ortami/kodu degismis; fazlar karistirilmadi')
                    write_json(output/'protocol.json', config); write_json(output/'result.json', {'status': 'RUNNING'})
                    write_json(output/'environment.json', environment)
                    for name in ['summary.json', 'summary.csv']: (output/name).unlink(missing_ok=True)
                    backup_path = small_backup(ROOT, output)
                    phases = []; immutable = {str(base/n): sha256(base/n) for n in FILES12}
                    immutable.update({str(w1): sha256(w1), str(w2): sha256(w2), str(official): sha256(official)})
                    for pass_id, variant in PHASES:
                        target = output/f'pass_{pass_id+1}_{variant}.json'
                        if target.exists():
                            phase = validate_phase(read_json(target), config, pass_id, variant)
                            print('TAMAMLANAN FAZ KULLANILIYOR:', target.name, flush=True)
                        else:
                            (output/'phase_error.json').unlink(missing_ok=True)
                            job = {'output': str(output), 'protocol_sha256': fingerprint(config), 'pass_id': pass_id, 'variant': variant}
                            write_json(output/'worker_job.json', job)
                            print(f'YENI GPU OLCUMU: tur {pass_id+1}/3 {variant}', flush=True)
                            code = child_process([sys.executable, '-X', 'faulthandler', '-u', '-m', 'hycado_start.runtime_benchmark', '--worker', str(output/'worker_job.json')], ROOT)
                            if code != 0: raise RuntimeError('GPU olcum alt islemi durdu: '+str(code))
                            phase = validate_phase(read_json(target), config, pass_id, variant)
                        phases.append(phase)
                    if immutable != {p: sha256(p) for p in immutable}: raise ValueError('12 kayitlari/agirliklari degismis')
                    summary = summarize(phases); write_json(output/'summary.json', summary)
                    with (output/'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
                        keys = ['mean_ms', 'median_ms', 'p95_ms', 'throughput_images_per_second', 'peak_allocated_mib', 'peak_reserved_mib']
                        writer = csv.DictWriter(f, fieldnames=['variant']+keys); writer.writeheader()
                        for name, values in summary['variants'].items(): writer.writerow({'variant': name, **{k: values[k] for k in keys}})
                    result = {'status': 'COMPLETED', 'phases': 6, 'images_per_phase': 548, 'measured_image_calls': 3288,
                        'all_measured_outputs_exact_original12': True, 'original_weights_and_results_unchanged': True,
                        'training_started': False, 'parameter_search': False, 'small_backup_directory': backup_path}
                except (Exception, KeyboardInterrupt) as exc:
                    traceback.print_exc(); result = {'status': 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'ERROR', 'error': str(exc),
                        'resume': 'rerun same 13 command; completed whole phases reused, interrupted phase measured again from start'}
                write_json(output/'result.json', result); print('SONUC:', result['status'], flush=True)
        report_path = bundle(output)
        try: small_backup(ROOT, output, report=True)
        except Exception as exc: print('13 RAPORUNUN KUCUK YEDEGI ALINAMADI:', str(exc), flush=True)
        print('GONDERILECEK RAPOR:', report_path, flush=True)
        return result['status'] == 'COMPLETED'


def main():
    if '--worker' in sys.argv: return worker(Path(sys.argv[sys.argv.index('--worker')+1]))
    if '--report-only' in sys.argv:
        print(bundle(ROOT/'experiments'/OUT), flush=True); return 0
    return 0 if run() else 1


if __name__ == '__main__': raise SystemExit(main())
