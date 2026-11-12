"""Frozen alpha=.25 comparison on official VAL; no training or parameter search.

Uses fresh seed42 round80 EMAs, original crop geometry and inference recipes.
The current Python metric remains provisional until reference binary parity.
"""
from __future__ import annotations
import contextlib
import gc
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
import numpy as np

from .common import ROOT, sha256, write_json, utc_now, system_info
from .stage1 import fingerprint, training_lock
from .crop_loader import safe_path, read_json
from . import dev_evaluation as old
from . import proposals, crop_data, crop_capture, score_ablation
from .dev_metrics import raw_annotation, ignore_integral, prepare_groundtruth, outside_ignore, xyxy_to_det
from .project_backup import verify_receipt

W1 = '3b61a12a89a448b849bc965e44f00afa92042a5fc0928b4c4f801e0182f84149'
P1 = '159ea5fec31f25542b18417823cbefb74ed4a88c442050fec64dece7365ab6be'
P11D = '1b86a58d0d0bd69c95f2eef7d961281a627024c52391fe83a4d6cdbe06298f75'
SOURCE_D = {'protocol.json': '9328c602654eded0d8477318450e1bb90e35571cd991ea49a33311aff6daab2d',
            'result.json': 'a1598fe3879306fc6b75c45875c737e1b56310bcae75a03e7058c75753da6cac',
            'metrics.json': '8109ec79f5266974eb9c9722fad467a6bfffeb7f48f448db2496b7b6bcf0e424'}
OUT = 'official_val_a025_v1_seed42'


def signed_json(path, body):
    payload = dict(body); payload['content_sha256'] = fingerprint(body); write_json(path, payload)
    return payload


def checked_json(path, signature, record):
    p = read_json(path)
    if (p.get('content_sha256') != fingerprint({k: v for k, v in p.items() if k != 'content_sha256'}) or
            p.get('protocol_sha256') != signature or p.get('source_image') != record['image'] or
            p.get('image_sha256') != record['image_sha256'] or
            (p.get('width'), p.get('height')) != (record['width'], record['height'])):
        raise ValueError('12 tahmin kaydi kimligi degismis: ' + str(path))
    return p


def val_records(prepared_manifest, training_manifest, *, expected=(6471, 548)):
    raw = prepared_manifest['records']
    if prepared_manifest.get('purpose') != 'fresh_original_visdrone_conversion' or prepared_manifest.get('is_small_fixture'):
        raise ValueError('Ilk veri hazirliginin orijinal manifesti gerekli')
    train = [r for r in raw if r['split'] == 'train']; val = [r for r in raw if r['split'] == 'val']
    if (len(train), len(val)) != expected: raise ValueError('Resmi TRAIN/VAL sayisi farkli')
    if len({r['image'] for r in raw}) != len(raw): raise ValueError('Yinelenen kaynak goruntu yolu')
    if {r['image_sha256'] for r in val} & {r['image_sha256'] for r in train}:
        raise ValueError('Resmi VAL / TRAIN goruntu hash kesisimi')
    if {r['image_sha256'] for r in train} != {r['image_sha256'] for r in training_manifest['records']}:
        raise ValueError('Hazir TRAIN ve dondurulmus egitim manifesti farkli')
    records = []
    for r in sorted(val, key=lambda x: x['image']):
        parts = Path(r['image']).parts
        if len(parts) != 3 or parts[:2] != ('images', 'val') or '\\' in r['image']:
            raise ValueError('Resmi VAL kaynak yolu farkli')
        # Deliberately discard all annotation-derived counts/boxes from the input record.
        records.append({k: r[k] for k in ['image', 'width', 'height', 'image_sha256', 'annotation_sha256']})
    if len({Path(r['image']).stem for r in records}) != len(records): raise ValueError('VAL kok adi tekrari')
    return records


def inputs(root):
    source = root / 'experiments/score_ablation_v1_seed42'
    score_ablation.validate_capture(root)
    for name, digest in SOURCE_D.items():
        if sha256(source / name) != digest: raise ValueError('Dondurulacak 11D raporu farkli: ' + name)
    p = read_json(source / 'protocol.json'); result = read_json(source / 'result.json')
    if (fingerprint(p) != P11D or result.get('status') != 'COMPLETED' or result.get('images') != 647 or
            result.get('official_val_used') is not False or
            sha256(Path(__file__).with_name('score_ablation.py')) != p['code_sha256']):
        raise ValueError('Tamamlanan 11D kaydi/kodu farkli')
    sources = old.validate_sources(root / 'experiments/stage1_v1_seed42', root / 'experiments/stage1_predictions_v1_seed42')
    config11, weights, official = old.configuration(root, sources)
    if fingerprint(config11) != score_ablation.SOURCE_PROTOCOL: raise ValueError('Sabit 11 tarifi degismis')
    if fingerprint(read_json(root / 'experiments/stage1_v1_seed42/protocol.json')) != P1:
        raise ValueError('Stage1 protokolu farkli')
    stage_index = read_json(root / 'experiments/stage1_v1_seed42/checkpoint_index.json')
    stage_weights = safe_path(root / 'experiments/stage1_v1_seed42', stage_index['latest']['path'])
    if stage_index['latest']['sha256'] != W1 or sha256(stage_weights) != W1: raise ValueError('Stage1 80 EMA kimligi farkli')
    prepared = root / 'data/prepared'
    if sha256(prepared / 'manifest.json') != sources['manifest']['source_manifest_sha256']:
        raise ValueError('Ilk hazirlanan data/prepared manifesti degismis')
    records = val_records(read_json(prepared / 'manifest.json'), sources['manifest'])
    p8 = read_json(root / 'experiments/crop_dataset_v1_seed42/protocol.json'); decision = p8['decision']
    if decision['selected_configuration_id'] != 'c3_near_z3_b10': raise ValueError('Sabit crop politikasi farkli')
    for name, digest in decision['merge']['source_code_hashes'].items():
        if Path(name).name != name or sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError('Crop birlestirme kodu farkli: ' + name)
    required = [str(path.relative_to(root)).replace('\\', '/') for path in [stage_weights, weights, official]]
    required += ['data/prepared/manifest.json']
    required += ['hycado_start/' + path.name for path in (root / 'hycado_start').glob('*.py')]
    for r in records:
        required += ['data/prepared/' + r['image'], 'data/prepared/visdrone_annotations/val/' + Path(r['image']).stem + '.txt']
    for folder in ['raw_crop_capture_v1_seed42', 'score_ablation_v1_seed42']:
        base = root / 'experiments' / folder
        required += [path.relative_to(root).as_posix() for path in base.glob('*.json')]
        file_name = 'capture_index.json' if folder == 'raw_crop_capture_v1_seed42' else 'cache_index.json'
        index = read_json(base / file_name)
        if len(index['records']) != 647 or len({r['source_image'] for r in index['records']}) != 647:
            raise ValueError('Yedeklenecek 11C/11D 647 kaynak indeksi eksik')
        for r in index['records']:
            if folder == 'raw_crop_capture_v1_seed42':
                meta = safe_path(base, r['path']); raw_path = meta.parent / 'queries_and_trace.npz'
                expected_raw = r['archive_sha256']; expected_meta = r['sha256']
            else:
                raw_path = base / 'sources' / Path(r['source_image']).stem / 'matches.npz'
                meta = raw_path.parent / 'summary.json'; expected_raw = r['npz_sha256']; expected_meta = r['summary_sha256']
            if sha256(raw_path) != expected_raw or sha256(meta) != expected_meta:
                raise ValueError('Yedeklenecek 11C/11D ham/ara kayit eksik veya degismis')
            required += [raw_path.relative_to(root).as_posix(), meta.relative_to(root).as_posix()]
    for folder in ['stage1_v1_seed42', 'rtdetr_crop_v1_seed42', 'development_eval_v1_seed42', 'crop_dataset_v1_seed42']:
        required += ['experiments/' + folder + '/protocol.json']
    print('Yedekteki kritik kod/veri/model/ham kayitlar dogrulaniyor...', flush=True)
    receipt = verify_receipt(root, required)
    configuration = {'id': 'hycado_frozen_official_val_a025_v1', 'decision_date': '2026-10-29',
                     'selection_source11D_protocol_sha256': P11D, 'selection_source11D_metadata': SOURCE_D,
                     'selection_criterion': 'maximum primary development AP among fixed 11D multipliers; alpha .25 selected before VAL inference',
                     'selection_uncertainty': 'alpha .25 exceeded .50 by only .044674172623064834 development AP points; no superiority significance claim',
                     'scope': '548 official VAL images; independent of parameter selection, one seed42 model pair',
                     'stage1_protocol_sha256': P1, 'stage1_weights_sha256': W1, 'crop_weights_sha256': old.W80,
                     'crop_training_protocol_sha256': old.P10, 'development_protocol_sha256': score_ablation.SOURCE_PROTOCOL,
                     'prepared_manifest_sha256': sha256(prepared / 'manifest.json'), 'records': records,
                     'stage1_inference': sources['p6']['inference'], 'crop_inference': config11['crop_inference'],
                     'crop_policy': decision, 'crop_score_multiplier': .25,
                     'fusion': {'nms_iou': .7, 'max_det': 1000, 'raw_conf_strictly_greater': .001,
                                'second_confidence_cutoff_after_scaling': False, 'baseline_scores_unchanged': True},
                     'variants': ['baseline', 'hybrid_a025'], 'metric': config11['evaluation'],
                     'official_val_used_for_evaluation': True, 'official_val_used_for_parameter_selection': False,
                     'official_binary_parity_verified': False, 'training_started': False, 'parameter_search': False,
                     'freeze_before_val_annotation_parsing': True, 'runtime_versions': config11['runtime_versions'],
                     'timing': 'progress wall time only; not a controlled end-to-end benchmark',
                     'code_sha256': {n: sha256(Path(__file__).with_name(n)) for n in ['frozen_val.py', 'project_backup.py', 'score_ablation.py', 'crop_capture.py', 'dev_evaluation.py', 'dev_metrics.py']}}
    return configuration, records, prepared, stage_weights, weights, official, sources['p6'], receipt


def baseline_pass(records, prepared, output, signature, factory):
    predictor = None; manifest = []; start = time.perf_counter()
    for i, record in enumerate(records, 1):
        image = safe_path(prepared, record['image'])
        if sha256(image) != record['image_sha256']: raise ValueError('VAL goruntusu degismis')
        path = output / 'baseline' / (image.stem + '.json')
        if path.exists(): p = checked_json(path, signature, record)
        else:
            if predictor is None: predictor = factory()
            width, height, rows, dropped = predictor(image)
            if (width, height) != (record['width'], record['height']): raise ValueError('VAL goruntu boyutu farkli')
            clean, bad = old.canonical_detections(rows, width, height)
            if bad or clean != rows or len(rows) > 1000: raise ValueError('Stage1 tahmin bicimi gecersiz')
            p = signed_json(path, {'protocol_sha256': signature, 'source_image': record['image'],
                    'image_sha256': record['image_sha256'], 'width': width, 'height': height,
                    'detections': rows, 'dropped_zero_area': dropped})
        rows, bad = old.canonical_detections(p['detections'], p['width'], p['height'])
        if bad or rows != p['detections'] or len(rows) > 1000: raise ValueError('Stage1 onbellegi gecersiz')
        manifest.append({'source_image': record['image'], 'path': path.relative_to(output).as_posix(), 'sha256': sha256(path)})
        if i == 1 or i % 10 == 0 or i == len(records):
            print(f'VAL STAGE1: {i}/{len(records)}; gecen={(time.perf_counter()-start)/60:.1f} dk', flush=True)
            write_json(output / 'progress.json', {'status': 'STAGE1_INFERENCE', 'images_done': i, 'images_total': len(records)})
    predictor = None; gc.collect()
    write_json(output / 'baseline_index.json', {'protocol_sha256': signature, 'records': manifest})
    return manifest


def crop_pass(records, prepared, output, signature, baseline_index, decision, factory):
    from PIL import Image
    mapping = {r['source_image']: r for r in baseline_index}; predictor = None; manifest = []; start = time.perf_counter(); crops = 0
    for i, record in enumerate(records, 1):
        image_path = safe_path(prepared, record['image']); item = mapping[record['image']]
        base_path = safe_path(output, item['path'])
        if sha256(base_path) != item['sha256'] or sha256(image_path) != record['image_sha256']:
            raise ValueError('Kaynak VAL/Stage1 kaydi degismis')
        base = checked_json(base_path, signature, record)
        # The geometry function receives NO GT annotations, counts or ignore regions.
        _, clusters = crop_data.select_clusters(base, decision)
        rects = [list(c['rect']) for c in clusters]
        destination = output / 'sources' / image_path.stem
        if destination.exists():
            p = checked_json(destination / 'prediction.json', signature, record)
            if (p['baseline_sha256'] != item['sha256'] or p['crop_rects'] != rects or
                    sha256(destination / 'raw_crop_queries.npz') != p['raw_archive_sha256']):
                raise ValueError('VAL crop kaydi degismis')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=image_path.stem + '_', dir=destination.parent))
            try:
                raw = []; candidates = list(base['detections'])
                with Image.open(image_path) as im:
                    if im.size != (record['width'], record['height']): raise ValueError('VAL goruntu boyutu farkli')
                    rgb = im.convert('RGB')
                for j, rect in enumerate(rects):
                    crop_path = temporary / f'crop_{j:04d}.png'
                    rgb.crop(tuple(rect)).save(crop_path, format='PNG', compress_level=1)
                    crop = {'width': rect[2]-rect[0], 'height': rect[3]-rect[1], 'source_crop_xyxy': rect,
                            'image_sha256': sha256(crop_path)}
                    if predictor is None: predictor = factory()
                    query = predictor(crop_path, crop)
                    if query.shape != (300, 14) or query.dtype != np.float32 or not np.isfinite(query).all():
                        raise ValueError('VAL ham query bicimi farkli')
                    raw.append(query)
                    rows = old.map_queries(query, crop)
                    for row in rows: row[4] *= .25
                    candidates.extend(rows); crop_path.unlink()
                hybrid = old.classwise_nms(candidates, record['width'], record['height'])
                archive = temporary / 'raw_crop_queries.npz'
                np.savez_compressed(archive, raw_queries=np.stack(raw) if raw else np.empty((0, 300, 14), dtype=np.float32),
                                    crop_rects=np.asarray(rects, dtype=np.int32).reshape(-1, 4))
                p = signed_json(temporary / 'prediction.json', {'protocol_sha256': signature,
                    'source_image': record['image'], 'image_sha256': record['image_sha256'],
                    'width': record['width'], 'height': record['height'], 'baseline_sha256': item['sha256'],
                    'crop_rects': rects, 'crop_count': len(rects), 'baseline': base['detections'], 'hybrid_a025': hybrid,
                    'raw_archive_sha256': sha256(archive)})
                temporary.rename(destination)
            finally:
                if temporary.exists(): shutil.rmtree(temporary)
        for key in ['baseline', 'hybrid_a025']:
            clean, bad = old.canonical_detections(p[key], record['width'], record['height'])
            if bad or clean != p[key] or len(clean) > 1000: raise ValueError('VAL sonuc kutulari gecersiz')
        if p['baseline'] != base['detections'] or p['crop_count'] != len(rects): raise ValueError('Baseline/crop kapsami farkli')
        crops += p['crop_count']
        path = destination / 'prediction.json'
        manifest.append({'source_image': record['image'], 'path': path.relative_to(output).as_posix(),
                         'sha256': sha256(path), 'raw_archive_sha256': p['raw_archive_sha256'],
                         'annotation_sha256': record['annotation_sha256'], 'crops': p['crop_count']})
        if i == 1 or i % 10 == 0 or i == len(records):
            print(f'VAL CROP: {i}/{len(records)}; crop={crops}; gecen={(time.perf_counter()-start)/60:.1f} dk', flush=True)
            write_json(output / 'progress.json', {'status': 'CROP_INFERENCE', 'images_done': i, 'images_total': len(records), 'crops_done': crops})
    predictor = None; gc.collect()
    write_json(output / 'prediction_index.json', {'protocol_sha256': signature, 'records': manifest})
    return manifest


def measure(records, prepared, output, signature, index):
    if len(index) != len(records) or {r['source_image'] for r in index} != {r['image'] for r in records}:
        raise ValueError('Tum VAL cikarimlari tamamlanmadan etiketler degerlendirmeye acilmaz')
    parts = []; mapping = {r['source_image']: r for r in index}; source_hashes = []
    for i, record in enumerate(records, 1):
        item = mapping[record['image']]; path = safe_path(output, item['path'])
        annotation = prepared / 'visdrone_annotations/val' / (Path(record['image']).stem + '.txt')
        # First parsing/use of VAL annotation rows occurs after ALL inference is committed.
        if sha256(annotation) != record['annotation_sha256'] or sha256(path) != item['sha256']:
            raise ValueError('VAL annotation/tahmin dosyasi degismis')
        p = checked_json(path, signature, record)
        raw = raw_annotation(annotation.read_text(encoding='utf-8-sig'))
        integral = ignore_integral(raw, record['width'], record['height']); gt = prepare_groundtruth(raw, integral)
        data = {'gt_denominators': np.array([np.count_nonzero(gt[:, 5] == c) for c in range(1, 11)], dtype=np.int64)}
        data['presence'] = (data['gt_denominators'] > 0).astype(np.int64)
        for name in ['baseline', 'hybrid_a025']:
            det = outside_ignore(xyxy_to_det(p[name]), integral)[:500]
            data[name + '_scores'] = det[:, 4].copy(); data[name + '_classes'] = det[:, 5].astype(np.int8)
            data[name + '_flags'] = np.column_stack([score_ablation.fast_flags(det, gt, float(t)) for t in score_ablation.THRESHOLDS])
            old.export_txt(output / 'detections_txt' / name / (path.parent.name + '.txt'), p[name])
        parts.append(data); source_hashes += [(annotation, record['annotation_sha256']), (path, item['sha256'])]
        if i == 1 or i % 25 == 0 or i == len(records): print(f'VAL OLCUM: {i}/{len(records)}', flush=True)
    metrics = {name: score_ablation.aggregate(parts, name) for name in ['baseline', 'hybrid_a025']}
    for path, digest in source_hashes:
        if sha256(path) != digest: raise ValueError('VAL kaynak dosyasi olcum sirasinda degismis')
    report = {'protocol_sha256': signature, 'images': len(records), 'variants': metrics,
              'delta_hybrid_minus_baseline_pp': {k: metrics['hybrid_a025'][k]-metrics['baseline'][k] for k in ['AP', 'AP50', 'AP75', 'AR1', 'AR10', 'AR100', 'AR500']},
              'official_binary_parity_verified': False, 'scope': 'official VAL, fixed seed42 model pair; provisional Python metric',
              'official_val_used_for_selection': False, 'parameter_search': False, 'training_started': False,
              'multiple_training_seed_uncertainty_measured': False, 'confidence_intervals_computed': False}
    write_json(output / 'metrics.json', report)
    import csv
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['variant', 'AP', 'AP50', 'AP75', 'AR1', 'AR10', 'AR100', 'AR500'])
        writer.writeheader()
        for name in metrics: writer.writerow({'variant': name, **{k: metrics[name][k] for k in writer.fieldnames[1:]}})
    return report


def bundle(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output, prefix='12_', suffix='.zip.tmp'); os.close(fd)
    target = output / '12_sabit_val_rapor.zip'
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in ['protocol.json', 'result.json', 'progress.json', 'environment.json', 'metrics.json', 'metrics.csv',
                         'baseline_index.json', 'prediction_index.json', 'backup_receipt.json', 'source_integrity.json']:
                if (output / name).exists(): z.write(output / name, name)
            if (output / 'console.log').exists():
                with (output / 'console.log').open('rb') as f:
                    f.seek(max(0, (output / 'console.log').stat().st_size - 80000)); z.writestr('console_tail.log', f.read())
        os.replace(temporary, target)
    finally: Path(temporary).unlink(missing_ok=True)
    return target


def run():
    from .__main__ import _Tee
    output = ROOT / 'experiments' / OUT; output.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        stack.enter_context(training_lock(ROOT / '.backup.lock'))
        for path in [output / '.evaluation.lock', ROOT / 'experiments/stage1_v1_seed42/.training.lock',
                     ROOT / 'experiments/rtdetr_crop_v1_seed42/.training.lock', ROOT / 'experiments/raw_crop_capture_v1_seed42/.capture.lock',
                     ROOT / 'experiments/score_ablation_v1_seed42/.analysis.lock', ROOT / 'experiments/development_eval_v1_seed42/.evaluation.lock']:
            stack.enter_context(training_lock(path))
        with (output / 'console.log').open('a', encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
                try:
                    print('12B SABIT RESMI VAL DEGERLENDIRMESI:', utc_now(), flush=True)
                    print('Egitim yok. Alpha .25 sabit. Resmi VAL sonucuna gore esik aranmaz.', flush=True)
                    config, records, prepared, stage_weights, weights, official, cfg06, receipt = inputs(ROOT)
                    if (output / 'protocol.json').exists() and read_json(output / 'protocol.json') != config:
                        raise ValueError('12 protokolu/kodu degismis; resmi VAL sonuclari karistirilmadi')
                    # Seal method, model/data identities and sole alpha before any VAL inference.
                    write_json(output / 'protocol.json', config); signature = fingerprint(config)
                    write_json(output / 'result.json', {'status': 'RUNNING'})
                    for name in ['metrics.json', 'metrics.csv']: (output / name).unlink(missing_ok=True)
                    write_json(output / 'backup_receipt.json', receipt)
                    write_json(output / 'environment.json', system_info(include_torch=True))
                    import torch
                    if not torch.cuda.is_available(): raise RuntimeError('CUDA gerekli; CPU cikarimi sessizce baslatilmadi')
                    if shutil.disk_usage(output).free < 2*1024**3: raise OSError('En az 2 GiB bos alan gerekli')
                    critical = [stage_weights, weights, official, ROOT / 'data/prepared/manifest.json']
                    critical += [ROOT / 'experiments' / folder / name for folder, names in [
                        ('score_ablation_v1_seed42', list(SOURCE_D)),
                        ('development_eval_v1_seed42', ['protocol.json', 'result.json', 'metrics.json', 'prediction_index.json']),
                        ('raw_crop_capture_v1_seed42', ['protocol.json', 'result.json', 'capture_index.json'])] for name in names]
                    before = {str(p): sha256(p) for p in critical}
                    base_index = baseline_pass(records, prepared, output, signature,
                                lambda: proposals.make_predictor(proposals.load_ema(stage_weights, P1), cfg06, output))
                    gc.collect(); torch.cuda.empty_cache()
                    model = None
                    def factory():
                        nonlocal model
                        model = crop_capture.load_model(weights, official, 'cuda')
                        return lambda path, crop: crop_capture.predict_queries(model, path, crop, 'cuda')
                    index = crop_pass(records, prepared, output, signature, base_index, config['crop_policy'], factory)
                    model = None; gc.collect(); torch.cuda.empty_cache()
                    report = measure(records, prepared, output, signature, index)
                    for record in records:
                        if sha256(safe_path(prepared, record['image'])) != record['image_sha256']:
                            raise ValueError('VAL goruntu dosyasi kosu sirasinda degismis')
                    if before != {str(p): sha256(p) for p in critical}: raise ValueError('Eski agirlik/protokol/sonuc degismis')
                    write_json(output / 'source_integrity.json', {'unchanged': True, 'sha256': before})
                    if report['images'] != 548 or len(index) != 548: raise ValueError('548 goruntu tam bitmeli')
                    result = {'status': 'COMPLETED', 'images': 548, 'crops': sum(r['crops'] for r in index),
                              'crop_score_multiplier': .25, 'variants': ['baseline', 'hybrid_a025'],
                              'backup_verified': True, 'original_weights_and_results_unchanged': True,
                              'training_started': False, 'parameter_search': False, 'official_val_used_for_evaluation': True,
                              'official_val_used_for_selection': False, 'official_binary_parity_verified': False}
                except (Exception, KeyboardInterrupt) as exc:
                    traceback.print_exc()
                    result = {'status': 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'ERROR',
                              'error_type': type(exc).__name__, 'error': str(exc),
                              'resume': 'Rerun same 12B command; completed image caches reused; no parameter changes'}
                write_json(output / 'result.json', result); write_json(output / 'progress.json', result)
                print('SONUC:', result['status'], flush=True)
        print('GONDERILECEK RAPOR:', bundle(output), flush=True)
        return result['status'] == 'COMPLETED'


def main():
    if '--report-only' in sys.argv:
        print(bundle(ROOT / 'experiments' / OUT), flush=True); return 0
    return 0 if run() else 1


if __name__ == '__main__': raise SystemExit(main())
