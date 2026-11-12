"""Frozen Stage-1 EMA inference cache and TRAIN-development crop coverage."""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import html
import importlib.metadata
import itertools
import json
import math
import statistics
import sys
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from .common import ROOT, sha256, system_info, utc_now, write_json
from .data import NAMES
from .experiment_data import verify_research_data
from .proposal_geometry import (canonical_detections, coverage_counts, load_targets,
                                select_regions, union_area)
from .stage1 import fingerprint, training_lock


EXPECTED_STAGE1_CODE = 'aa9bc3ee156b149b5f52a4ba4c233ef08163beb67d264d418765280ea7187877'


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_reference(run_dir):
    """Read-only validation: no training files, weights, or protocols rewritten."""
    run_dir = Path(run_dir).resolve()
    protocol = load_json(run_dir/'protocol.json')
    result = load_json(run_dir/'result.json')
    index = load_json(run_dir/'checkpoint_index.json')
    history = load_json(run_dir/'history.json')
    if (protocol['protocol_id'] != 'hycado_stage1_development_v1' or
            protocol['code_hashes']['stage1.py'] != EXPECTED_STAGE1_CODE or
            protocol['epochs'] != 80 or protocol['imgsz'] != 800 or protocol['seed'] != 42):
        raise ValueError('Beklenen sabit Stage-1 v1 protokolu bulunamadi')
    if result['status'] != 'COMPLETED' or result['completed_epochs'] != 80:
        raise ValueError('Stage-1 egitimi 80 epoch tamamlanmamis')
    if [r['epoch'] for r in history] != list(range(1, 81)):
        raise ValueError('Stage-1 history 1-80 sirasi eksik')
    if index['protocol_sha256'] != fingerprint(protocol) or index['latest']['epoch'] != 80:
        raise ValueError('Son checkpoint protokolu/epoch uyusmuyor')
    checkpoint = (run_dir/index['latest']['path']).resolve()
    if not checkpoint.is_relative_to(run_dir) or sha256(checkpoint) != index['latest']['sha256']:
        raise ValueError('Son checkpoint dosyasi degismis/eksik')
    for name, digest in protocol['code_hashes'].items():
        if Path(name).name != name or sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError(f'Kaynak kod Stage-1 kaydiyla uyusmuyor: {name}')
    if sha256(run_dir/'data/manifest.json') != protocol['dataset_manifest_sha256']:
        raise ValueError('Veri manifesti kimligi degismis')
    if sha256(run_dir/'data/dataset.yaml') != protocol['dataset_yaml_sha256']:
        raise ValueError('Dataset YAML kimligi degismis')
    manifest = verify_research_data(run_dir/'data')
    validate_manifest(manifest)
    for name, version in protocol['runtime_versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Kurulu paket surumu degismis: {name}')
    return protocol, index, checkpoint, manifest


def validate_manifest(manifest):
    records = manifest['records']
    counts = Counter(r['split'] for r in records)
    if (dict(counts) != {'train': 5824, 'val': 647} or
            len({r['source_image'] for r in records}) != 6471 or
            any(r['source_split'] != 'train' for r in records) or
            manifest['official_val_used_for_training_or_selection'] is not False):
        raise ValueError('Beklenen 5824/647 resmi TRAIN ayrimi gerekli')
    cache_keys = [(r['split'], PurePosixPath(r['image']).stem) for r in records]
    if len(set(cache_keys)) != len(cache_keys):
        raise ValueError('Tahmin dosya adlarinda ayni split/stem tekrari')
    for key in ['group', 'image_sha256']:
        train = {r[key] for r in records if r['split'] == 'train'}
        dev = {r[key] for r in records if r['split'] == 'val'}
        if train & dev:
            raise ValueError('Train/gelistirme kesisimi: '+key)


def config_for(protocol, index, environment):
    return {'protocol_id': 'hycado_stage1_predictions_coverage_v1',
            'source_stage1_protocol_sha256': fingerprint(protocol),
            'checkpoint': index['latest'], 'weights_used': 'ema',
            'dataset_manifest_sha256': protocol['dataset_manifest_sha256'],
            'runtime_versions': protocol['runtime_versions'],
            'gpu': environment['gpu'], 'torch_cuda_runtime': environment['torch_cuda_runtime'],
            'inference': {'imgsz': 800, 'batch': 1, 'half': True, 'rect': True,
                          'conf': .001, 'iou': .7, 'max_det': 1000, 'agnostic_nms': False,
                          'augment': False, 'device': '0', 'fuse': True,
                          'seed': 42, 'cpu_threads': 2, 'opencv_threads': 1,
                          'nms_max_nms': 30000, 'nms_max_time_img': .05,
                          'coordinates': 'original_image_xyxy_continuous',
                          'cache_type': 'post_classwise_NMS_not_raw_head_output'},
            'coverage': {'source_split': 'official TRAIN internal development only',
                         'thresholds': [.05, .15, .30], 'top_k': [10, 30, 60],
                         'context_scales': [1.5, 2., 3.],
                         'context_definition': 'crop width/height = anchor width/height * scale',
                         'geometry': 'floor left/top, ceil right/bottom, clip to image, exact dedup after Top-K',
                         'gt_source': 'audited derived clean development labels',
                         'metric': 'max single crop intersection / GT area; not detection recall/AP',
                         'small_definition': 'original-pixel GT area < 1024; descriptive, not official VisDrone AP-small',
                         'auto_select_configuration': False},
            'official_val_used': False, 'official_visdrone_metrics_computed': False,
            'code_hashes': {name: sha256(Path(__file__).with_name(name))
                            for name in ['proposals.py', 'proposal_geometry.py']}}


def load_ema(checkpoint, protocol_hash):
    import torch
    torch.set_num_threads(2)
    # Full Python model objects are our own newly trained, SHA-verified checkpoint.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (saved.get('protocol_sha256') != protocol_hash or saved.get('epoch') != 79 or
            len(saved.get('history', [])) != 80 or saved['history'][-1]['epoch'] != 80):
        raise ValueError('Checkpoint payload protokol/epoch uyusmuyor')
    model = saved.get('ema')
    if not isinstance(model, torch.nn.Module):
        raise ValueError('Checkpoint icinde EMA modeli bulunamadi')
    names = model.names
    names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    if names != NAMES or model.model[-1].nc != 10:
        raise ValueError('EMA sinif sirasi/sayisi beklenen VisDrone 10 sinifla uyusmuyor')
    if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values() if t.is_floating_point()):
        raise ValueError('EMA agirliklari veya buffer degerleri sonlu degil')
    del saved
    gc.collect()
    return model.float().eval()


def make_predictor(model, configuration, output, *, _cpu_test=False):
    import torch
    import torchvision  # Ensure pinned torchvision NMS path, also for one-image calls.
    import cv2
    import numpy as np
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    args = configuration['inference']
    torch.set_num_threads(args['cpu_threads'])
    cv2.setNumThreads(args['opencv_threads'])
    torch.manual_seed(args['seed'])
    if not _cpu_test:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA bulunamadi; CPU uzerinde sessizce baslatilmadi')
        torch.cuda.manual_seed_all(args['seed'])
        torch.cuda.reset_peak_memory_stats()
    overrides = {k: args[k] for k in ['imgsz', 'batch', 'half', 'rect', 'conf', 'iou',
                                     'max_det', 'agnostic_nms', 'augment', 'device']}
    if _cpu_test:
        overrides.update(device='cpu', half=False, imgsz=64)
    overrides.update(task='detect', save=False, save_txt=False, save_crop=False, show=False,
                     verbose=False, stream_buffer=False, compile=False,
                     project=str(output/'predictor_metadata'), name='inference', exist_ok=True)
    predictor = DetectionPredictor(overrides=overrides)
    predictor.setup_model(model, verbose=False)
    if bool(predictor.args.half) != overrides['half']:
        raise ValueError('Istenen inference precision etkinlestirilemedi')

    def predict(path):
        pixels = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if pixels is None:
            raise ValueError(f'Goruntu okunamadi: {path}')
        height, width = pixels.shape[:2]
        result = predictor(source=pixels, stream=False)
        if len(result) != 1 or tuple(result[0].orig_shape) != (height, width):
            raise ValueError('Tahmin ile kaynak goruntu boyutu eslesmiyor')
        detections, dropped = canonical_detections(result[0].boxes.data.float().cpu().tolist(), width, height)
        return width, height, detections, dropped
    return predict


def prediction_path(output, record):
    p = PurePosixPath(record['image'])
    if len(p.parts) != 3 or p.parts[:2] != ('images', record['split']) or record['split'] not in {'train', 'val'}:
        raise ValueError('Tahmin kaynak yolu gecersiz')
    return Path(output)/'predictions'/record['split']/(p.stem+'.json')


def validate_cache(payload, record, digest):
    signature = payload.get('content_sha256')
    body = {k: v for k, v in payload.items() if k != 'content_sha256'}
    if signature != fingerprint(body):
        raise ValueError('Tahmin onbellegi icerik hash uyusmazligi')
    if (payload['protocol_sha256'] != digest or payload['source_image'] != record['source_image'] or
            payload['image_sha256'] != record['image_sha256'] or payload['split'] != record['split']):
        raise ValueError('Tahmin onbellegi kimligi uyusmuyor')
    w, h = payload['width'], payload['height']
    if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
        raise ValueError('Tahmin boyutu gecersiz')
    clean, dropped = canonical_detections(payload['detections'], w, h)
    if dropped or clean != payload['detections'] or len(clean) > 1000:
        raise ValueError('Tahmin onbellegi kutulari gecersiz')
    return payload


def cache_predictions(records, data_root, output, digest, predict_factory):
    """One image at a time, atomic files; resume never reruns valid cached images."""
    predict = None
    new, reused, truncated, boxes, dropped, index = 0, 0, 0, 0, 0, []
    start = time.perf_counter()
    for number, record in enumerate(records, 1):
        path = prediction_path(output, record)
        if path.exists():
            payload = validate_cache(load_json(path), record, digest)
            reused += 1
        else:
            if predict is None:
                predict = predict_factory()
            width, height, detections, bad = predict(Path(data_root)/record['image'])
            detections, invalid = canonical_detections(detections, width, height)
            payload = {'protocol_sha256': digest, 'source_image': record['source_image'],
                       'image_sha256': record['image_sha256'], 'split': record['split'],
                       'width': int(width), 'height': int(height), 'detections': detections,
                       'dropped_zero_area': bad+invalid}
            payload['content_sha256'] = fingerprint(payload)
            validate_cache(payload, record, digest)
            write_json(path, payload)
            new += 1
        boxes += len(payload['detections'])
        dropped += payload['dropped_zero_area']
        truncated += len(payload['detections'])+payload['dropped_zero_area'] >= 1000
        index.append({'source_image': record['source_image'], 'split': record['split'],
                      'path': path.relative_to(output).as_posix(), 'sha256': sha256(path),
                      'detections': len(payload['detections'])})
        if number == 1 or number % 100 == 0 or number == len(records):
            state = {'status': 'PREDICTING', 'images_processed': number, 'images_total': len(records),
                     'new': new, 'reused': reused, 'elapsed_this_invocation_seconds': time.perf_counter()-start,
                     'source_image': record['source_image']}
            write_json(Path(output)/'progress.json', state)
            print(f'Tahmin: {number}/{len(records)}; yeni={new}; onbellekten={reused}', flush=True)
    write_json(Path(output)/'prediction_index.json', {'protocol_sha256': digest, 'records': index})
    return {'images': len(records), 'new_this_invocation': new, 'reused_this_invocation': reused,
            'detections': boxes, 'images_at_max_det_cap': truncated, 'dropped_zero_area': dropped,
            'elapsed_this_invocation_seconds': time.perf_counter()-start,
            'timing_is_not_a_fair_detector_runtime_benchmark': True}


def add_counts(destination, source):
    for k, v in source.items():
        destination[k] = destination.get(k, 0)+v


def measure_coverage(records, data_root, output, digest, configuration):
    """Only internal TRAIN development; no generated crop training labels here."""
    c = configuration['coverage']
    configs = list(itertools.product(c['thresholds'], c['top_k'], c['context_scales']))
    aggregates = [{'threshold': t, 'top_k': k, 'context_scale': s, 'images': 0,
                   'all': {}, 'small_native_area_lt_1024': {},
                   'classes': {str(j): {} for j in range(10)},
                   'crop_counts': [], 'union_fractions': [], 'sum_fractions': []}
                  for t, k, s in configs]
    selected = [r for r in records if r['split'] == 'val']
    if not selected:
        raise ValueError('Gelistirme goruntusu yok')
    per_image_path = Path(output)/'coverage_per_image.jsonl'
    with per_image_path.open('w', encoding='utf-8') as per_image:
        for number, record in enumerate(selected, 1):
            payload = validate_cache(load_json(prediction_path(output, record)), record, digest)
            width, height = payload['width'], payload['height']
            label = Path(data_root)/'labels'/record['split']/(PurePosixPath(record['image']).stem+'.txt')
            targets = load_targets(label.read_text(encoding='utf-8'), width, height)
            if len(targets) != record['gt_count']:
                raise ValueError('Hedef sayisi manifestle uyusmuyor: '+str(label))
            for a in aggregates:
                regions = select_regions(payload['detections'], width, height,
                                         a['threshold'], a['top_k'], a['context_scale'])
                counts = coverage_counts(targets, regions)
                a['images'] += 1
                for key in ['all', 'small_native_area_lt_1024']:
                    add_counts(a[key], counts[key])
                for cls in a['classes']:
                    add_counts(a['classes'][cls], counts['classes'][cls])
                union = union_area(regions)/(width*height)
                total = sum((r[2]-r[0])*(r[3]-r[1]) for r in regions)/(width*height)
                a['crop_counts'].append(len(regions)); a['union_fractions'].append(union); a['sum_fractions'].append(total)
                per_image.write(json.dumps({'source_image': record['source_image'],
                                'threshold': a['threshold'], 'top_k': a['top_k'], 'context_scale': a['context_scale'],
                                'crops': len(regions), 'union_fraction': union, 'sum_fraction': total,
                                **counts}, allow_nan=False, separators=(',', ':'))+'\n')
            if number == 1 or number % 100 == 0 or number == len(selected):
                print(f'Crop kapsami: {number}/{len(selected)}; {len(configs)} ayar', flush=True)
                write_json(Path(output)/'progress.json', {'status': 'COVERAGE', 'images_processed': number,
                                                         'images_total': len(selected)})
    summary, csv_rows = [], []
    for a in aggregates:
        n = a['all']['targets']
        small = a['small_native_area_lt_1024']['targets']
        row = {key: a[key] for key in ['threshold', 'top_k', 'context_scale', 'images']}
        row.update(targets=n, **{'coverage_'+k: a['all'][k]/n if n else None
                                for k in ['center', 'ioa50', 'ioa90', 'ioa100']})
        row.update(small_targets=small, small_coverage_ioa90=a['small_native_area_lt_1024']['ioa90']/small if small else None,
                   mean_crops=statistics.mean(a['crop_counts']),
                   p95_crops=sorted(a['crop_counts'])[math.ceil(.95*a['images'])-1],
                   mean_union_fraction=statistics.mean(a['union_fractions']),
                   mean_sum_fraction=statistics.mean(a['sum_fractions']),
                   zero_crop_images=a['crop_counts'].count(0))
        csv_rows.append(row)
        summary.append({**row, 'counts': {key: a[key] for key in ['all', 'small_native_area_lt_1024', 'classes']}})
    report = {'protocol_sha256': digest, 'purpose': 'development spatial crop coverage, not detector AP/recall',
              'official_val_used': False, 'official_visdrone_metrics_computed': False,
              'auto_selected_configuration': None, 'configurations': summary}
    write_json(Path(output)/'coverage_summary.json', report)
    with (Path(output)/'coverage_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    pct = lambda value: '—' if value is None else f'{100*value:.2f}'
    table = ''.join('<tr>'+''.join(f'<td>{v}</td>' for v in [r['threshold'],r['top_k'],r['context_scale'],
                    pct(r['coverage_ioa90']), pct(r['small_coverage_ioa90']),
                    f"{r['mean_crops']:.2f}", f"{100*r['mean_union_fraction']:.2f}", f"{100*r['mean_sum_fraction']:.2f}"])
                    +'</tr>' for r in csv_rows)
    page = '<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage-1 crop kapsam raporu</title><style>body{max-width:1100px;margin:35px auto;padding:0 20px;font:16px/1.6 system-ui;color:#253a46}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:8px;border-bottom:1px solid #ddd}th{background:#edf4f7}code{overflow-wrap:anywhere}</style><h1>Stage-1 crop kapsam raporu</h1>'
    page += '<p>Yalnızca resmi TRAIN içinden ayrılan geliştirme görüntüleri. Bu değerler tespit recall’u veya resmi VisDrone AP değildir. IoA90: GT kutusunun en az %90 alanının tek bir crop içinde kalması. Crop sınıfı ile GT sınıfının eşleşmesi aranmaz.</p><p>Context scale, crop genişlik ve yüksekliğinin anchor kutusuna oranıdır. Birleşim alanı örtüşmeyi bir kez sayar; toplam alan örtüşen crop’ları ayrı sayar ve %100’ü aşabilir. Küçük hedef: orijinal piksel alanı &lt;1024; resmi AP-small tanımı olarak kullanılmaz.</p>'
    page += '<div style="overflow:auto"><table><tr><th>Eşik</th><th>Top-K</th><th>Context</th><th>IoA90 %</th><th>Küçük IoA90 %</th><th>Ort. crop</th><th>Birleşim alanı %</th><th>Toplam alan %</th></tr>'+table+'</table></div>'
    page += '<p>Otomatik optimum seçilmedi. Bu tablo crop eğitimi başlamadan kapsam/iş yükü kararını destekler. Orijinal görüntülerden crop koordinatları kullanılır; bu adım crop görüntüsü veya crop eğitim etiketi üretmez.</p><p>Protokol: <code>'+html.escape(digest)+'</code></p></html>'
    (Path(output)/'coverage_summary.html').write_text(page, encoding='utf-8')
    return {'development_images': len(selected), 'configurations': len(configs),
            'targets': summary[0]['targets'], 'auto_selected_configuration': None}


def make_report(output):
    output = Path(output)
    bundle = output/'06_tahmin_kapsam_rapor.zip'
    allowed = ['protocol.json', 'environment.json', 'result.json', 'progress.json',
               'prediction_summary.json', 'coverage_summary.json', 'coverage_summary.csv',
               'coverage_summary.html', 'validation_record.json']
    # A failed rerun must not bundle a previous success's coverage as current output.
    if not (output/'result.json').exists() or load_json(output/'result.json').get('status') != 'COMPLETED':
        allowed = ['protocol.json', 'environment.json', 'result.json', 'progress.json', 'validation_record.json']
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in allowed:
            path = output/name
            if path.exists():
                z.write(path, name)
        if (output/'console.log').exists():
            with (output/'console.log').open('rb') as stream:
                stream.seek(max(0, (output/'console.log').stat().st_size-150000))
                z.writestr('console_tail.log', stream.read())
    return bundle


def run(stage1_run=None, output=None):
    from .__main__ import _Tee
    stage1_run = Path(stage1_run or ROOT/'experiments/stage1_v1_seed42').resolve()
    output = Path(output or ROOT/'experiments/stage1_predictions_v1_seed42').resolve()
    if output == stage1_run or output.is_relative_to(stage1_run) or stage1_run.is_relative_to(output):
        raise ValueError('Tahmin klasoru egitim klasorunden ayri olmali')
    output.mkdir(parents=True, exist_ok=True)
    with training_lock(output/'.prediction.lock'):
        with (output/'console.log').open('a', encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
                try:
                    with training_lock(stage1_run/'.training.lock'):
                        print(f'\nSTAGE-1 TAHMIN VE CROP KAPSAMI: {utc_now()}', flush=True)
                        print('Egitim baslatilmaz. Resmi VAL okunmaz. Referans: epoch 80 EMA.', flush=True)
                        protocol, index, checkpoint, manifest = validate_reference(stage1_run)
                        environment = system_info(include_torch=True)
                        if not environment.get('cuda_available'):
                            raise RuntimeError('CUDA bulunamadi; once mevcut Python/GPU ortamini kontrol edin')
                        configuration = config_for(protocol, index, environment)
                        digest = fingerprint(configuration)
                        old = output/'protocol.json'
                        if old.exists() and fingerprint(load_json(old)) != digest:
                            raise ValueError('Tahmin protokolu/kod/donanimi degismis; eski onbellekle karistirilmadi')
                        write_json(old, configuration)
                        write_json(output/'environment.json', environment)
                        write_json(output/'validation_record.json', {'created_utc': utc_now(),
                                   'stage1_checkpoint_sha256': sha256(checkpoint), 'source_images_labels_annotations_verified': True,
                                   'source_code_hashes_verified': True, 'official_val_used': False})
                        data_root = stage1_run/'data'

                        def factory():
                            print('Dogrulanmis checkpoint icinden EMA yukleniyor...', flush=True)
                            return make_predictor(load_ema(checkpoint, fingerprint(protocol)), configuration, output)

                        predictions = cache_predictions(manifest['records'], data_root, output, digest, factory)
                        write_json(output/'prediction_summary.json', predictions)
                        gc.collect()
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        coverage = measure_coverage(manifest['records'], data_root, output, digest, configuration)
                        result = {'status': 'COMPLETED', 'predictions': predictions, 'coverage': coverage,
                                  'official_visdrone_metrics_computed': False,
                                  'weights_updated': False, 'official_val_used': False,
                                  'max_reserved_gib': torch.cuda.max_memory_reserved()/2**30}
                except (Exception, KeyboardInterrupt) as exc:
                    traceback.print_exc()
                    result = {'status': 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'ERROR',
                              'error': str(exc), 'error_type': type(exc).__name__,
                              'resume': 'Rerun 06; valid cached image predictions are reused'}
                write_json(output/'result.json', result)
                write_json(output/'progress.json', result)
                print('SONUC:', result['status'], flush=True)
        bundle = make_report(output)
        print('GONDERILECEK RAPOR:', bundle, flush=True)
        return result['status'] == 'COMPLETED'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1-run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        success = run(args.stage1_run, args.output)
    except Exception:
        traceback.print_exc()
        return 1
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
