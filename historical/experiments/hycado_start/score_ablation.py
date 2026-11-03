"""Separate exploratory TRAIN-development score ablation from frozen 11C arrays.

No model loading or inference. Original training/evaluation files are read only.
Score multipliers are a new fusion experiment, not the original fixed protocol.
"""
from __future__ import annotations
import contextlib
import csv
import importlib.metadata
from pathlib import Path
import os
import sys
import tempfile
import time
import traceback
import zipfile
import numpy as np

from .common import ROOT, sha256, write_json, utc_now
from .stage1 import fingerprint, training_lock
from .crop_loader import safe_path, read_json
from .dev_diagnostics import validate_source, SOURCE_PROTOCOL
from .dev_evaluation import cached_payload, classwise_nms
from .dev_metrics import (raw_annotation, ignore_integral, prepare_groundtruth,
                          outside_ignore, xyxy_to_det, overlaps, precision_area, REFERENCE_COMMIT)

CAPTURE_SIGNATURE = 'c17174506389490b32766f892d15c08fcbd6704b685aa444086eb3cb33ac9e8e'
CAPTURE_INDEX = '9a706540bead203e07dd9e1283739f27b903d01a4e0657fe6d440d1b1187a49b'
CAPTURE_SUMMARY = '498759b074709c178eb699efb1e9e788252942081cfef80e0b39658c958ac164'
ALPHAS = [('hybrid_a100', 1.), ('hybrid_a075', .75), ('hybrid_a050', .5),
          ('hybrid_a025', .25), ('hybrid_a010', .1)]
VARIANTS = ['baseline', 'crop_only'] + [n for n, _ in ALPHAS]
THRESHOLDS = np.linspace(.5, .95, 10)
CAPS = (1, 10, 100, 500)


def fast_flags(det, gt, threshold):
    """Equivalent greedy matching: normal first, last equal IoU, ignore unlimited."""
    result = np.zeros(len(det), dtype=np.int8)
    for cls in range(1, 11):
        did = np.flatnonzero(det[:, 5] == cls)
        target = gt[gt[:, 5] == cls, :5]
        target = target[np.argsort(target[:, 4], kind='stable')]
        state = -target[:, 4].astype(np.int8)
        matrix = overlaps(det[did, :5], target)
        for j, d in enumerate(did):
            available = np.flatnonzero((state == 0) & (matrix[j] >= threshold))
            if len(available):
                best = available[matrix[j, available] == matrix[j, available].max()][-1]
                state[best] = 1
                result[d] = 1
            elif np.any((state == -1) & (matrix[j] >= threshold)):
                result[d] = -1
    return result


def fusion_variants(candidates, origins, width, height, baseline, original):
    a = np.asarray(candidates, dtype=np.float64)
    o = np.asarray(origins)
    if (a.ndim != 2 or a.shape[1] != 6 or o.shape != (len(a), 3) or
            not np.isfinite(a).all() or not np.isin(o[:, 0], [0, 1]).all() or
            np.any(a[:, 4] <= .001) or np.any(a[:, 4] > 1) or
            not np.equal(a[:, 5], np.floor(a[:, 5])).all() or
            np.any((a[:, 5] < 0) | (a[:, 5] > 9))):
        raise ValueError('11C aday bicimi/skoru/sinifi gecersiz')
    if (a[o[:, 0] == 0].tolist() != baseline or
            not np.array_equal(o[o[:, 0] == 0, 2], np.arange(len(baseline)))):
        raise ValueError('11C baseline adaylari sabit 11 ile farkli')
    crop = o[:, 0] == 1
    result = {'baseline': baseline,
              'crop_only': classwise_nms(a[crop].tolist(), width, height)}
    for name, alpha in ALPHAS:
        changed = a.copy()
        changed[crop, 4] *= alpha
        # Confidence .001 was applied to RAW scores in 11C. No second cutoff:
        # isolate the effect of cross-source score ordering and NMS competition.
        result[name] = classwise_nms(changed.tolist(), width, height)
    if result['hybrid_a100'] != original:
        raise ValueError('alpha=1 sabit 11 sonucunu birebir uretmedi')
    return result


def analyze_parent(rawgt, width, height, candidates, origins, baseline, original):
    variants = fusion_variants(candidates, origins, width, height, baseline, original)
    integral = ignore_integral(rawgt, width, height)
    gt = prepare_groundtruth(rawgt, integral)
    data = {'gt_denominators': np.array([np.count_nonzero(gt[:, 5] == c) for c in range(1, 11)], dtype=np.int64)}
    data['presence'] = (data['gt_denominators'] > 0).astype(np.int64)
    for name, rows in variants.items():
        det = outside_ignore(xyxy_to_det(rows), integral)[:500]
        data[name + '_scores'] = det[:, 4].copy()
        data[name + '_classes'] = det[:, 5].astype(np.int8)
        data[name + '_flags'] = np.column_stack([fast_flags(det, gt, float(t)) for t in THRESHOLDS])
    return data


def aggregate(parts, variant):
    denominator = np.sum([p['gt_denominators'] for p in parts], axis=0)
    presence = np.sum([p['presence'] for p in parts], axis=0)
    if presence.sum() == 0:
        raise ValueError('Hedef sinif yok')
    ap = np.zeros((10, 10)); ar = np.zeros((10, 10, 4)); per_class = []
    for c in range(1, 11):
        positions = [np.flatnonzero(p[variant + '_classes'] == c) for p in parts]
        for t in range(10):
            for j, cap in enumerate(CAPS):
                ids = [q[q < cap] for q in positions]
                scores = np.concatenate([p[variant + '_scores'][q] for p, q in zip(parts, ids)])
                flags = np.concatenate([p[variant + '_flags'][q, t] for p, q in zip(parts, ids)])
                area, recall = precision_area(scores, flags, int(denominator[c - 1]))
                ar[c - 1, t, j] = recall
                if j == 3: ap[c - 1, t] = area
        per_class.append({'class_id': c, 'gt_count_in_reference_denominator': int(denominator[c - 1]),
                          'image_presence_count': int(presence[c - 1]), 'AP': float(ap[c - 1].mean()),
                          'AP50': float(ap[c - 1, 0]), 'AP75': float(ap[c - 1, 5]),
                          'AR500': float(ar[c - 1, :, 3].mean())})
    weights = presence / presence.sum()
    output = {'AP': float(np.sum(ap.mean(axis=1) * weights)),
              'AP50': float(np.sum(ap[:, 0] * weights)), 'AP75': float(np.sum(ap[:, 5] * weights)),
              **{f'AR{cap}': float(np.sum(ar[:, :, j].mean(axis=1) * weights)) for j, cap in enumerate(CAPS)},
              'class_macro_AP_separate_diagnostic': float(ap[presence > 0].mean()), 'per_class': per_class}
    output['counts_at_fixed_max500'] = {}
    for t, key in [(0, '0.5'), (5, '0.75')]:
        flags = np.concatenate([p[variant + '_flags'][:, t] for p in parts])
        output['counts_at_fixed_max500'][key] = {
            'TP': int(np.count_nonzero(flags == 1)), 'FP': int(np.count_nonzero(flags == 0)),
            'ignored_detections': int(np.count_nonzero(flags == -1))}
    return output


def compare_metrics(actual, expected):
    keys = ['AP', 'AP50', 'AP75', 'AR1', 'AR10', 'AR100', 'AR500', 'class_macro_AP_separate_diagnostic']
    for key in keys:
        if abs(actual[key] - expected[key]) > 1e-10:
            raise ValueError('Sabit 11 olcum esligi basarisiz: ' + key)
    if len(actual['per_class']) != len(expected['per_class']): raise ValueError('Sinif sayisi farkli')
    for a, b in zip(actual['per_class'], expected['per_class']):
        for key, value in a.items():
            if abs(value - b[key]) > 1e-10: raise ValueError('Sabit sinif olcumu farkli: ' + key)


def load_cache(folder, bindings):
    meta = read_json(folder / 'summary.json')
    if (meta.get('content_sha256') != fingerprint({k: v for k, v in meta.items() if k != 'content_sha256'}) or
            meta['bindings'] != bindings or meta['npz_sha256'] != sha256(folder / 'matches.npz')):
        raise ValueError('11D ara kayit kimligi/hash farkli')
    with np.load(folder / 'matches.npz', allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if data['gt_denominators'].shape != (10,) or data['presence'].shape != (10,):
        raise ValueError('11D GT sayaci bicimi farkli')
    for name in VARIANTS:
        n = len(data[name + '_scores'])
        if (n > 500 or data[name + '_classes'].shape != (n,) or data[name + '_flags'].shape != (n, 10) or
                not np.isfinite(data[name + '_scores']).all() or
                not np.isin(data[name + '_flags'], [-1, 0, 1]).all()):
            raise ValueError('11D eslesme bicimi farkli')
    return data


def save_cache(folder, data, bindings):
    import shutil
    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=folder.name + '_', dir=folder.parent))
    try:
        np.savez_compressed(temporary / 'matches.npz', **data)
        meta = {'bindings': bindings, 'npz_sha256': sha256(temporary / 'matches.npz')}
        meta['content_sha256'] = fingerprint(meta)
        write_json(temporary / 'summary.json', meta)
        if folder.exists(): raise FileExistsError('Mevcut 11D kaydinin uzerine yazilmadi')
        temporary.rename(folder)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)


def validate_capture(root):
    source = root / 'experiments/development_eval_v1_seed42'
    capture = root / 'experiments/raw_crop_capture_v1_seed42'
    cfg, old_index = validate_source(source)
    protocol = read_json(capture / 'protocol.json'); result = read_json(capture / 'result.json')
    summary = read_json(capture / 'capture_summary.json'); index = read_json(capture / 'capture_index.json')
    if (fingerprint(protocol) != CAPTURE_SIGNATURE or sha256(capture / 'capture_index.json') != CAPTURE_INDEX or
            sha256(capture / 'capture_summary.json') != CAPTURE_SUMMARY or
            index['protocol_sha256'] != CAPTURE_SIGNATURE or result.get('status') != 'COMPLETED' or
            result.get('images') != 647 or result.get('crops') != 10682 or
            result.get('exact_original11_images') != 647 or summary['different_original11_images'] != 0 or
            result.get('official_val_used') is not False):
        raise ValueError('Dogrulanmis 647/10682 birebir 11C kaydi gerekli')
    for name, digest in protocol['source11_metadata_hashes'].items():
        if Path(name).name != name or sha256(source / name) != digest:
            raise ValueError('Sabit 11 metadata degismis: ' + name)
    if sha256(Path(__file__).with_name('crop_capture.py')) != protocol['code_sha256']:
        raise ValueError('11C kaynak kodu degismis')
    if sha256(Path(__file__).with_name('dev_diagnostics.py')) != protocol['diagnostic_code_sha256']:
        raise ValueError('11B kaynak kodu degismis')
    if len(index['records']) != 647 or len({r['source_image'] for r in index['records']}) != 647:
        raise ValueError('11C kayit sayisi farkli')
    if {r['source_image'] for r in index['records']} != {r['source_image'] for r in old_index['records']}:
        raise ValueError('11 ve 11C goruntu kumeleri farkli')
    return source, capture, old_index, index


def process(root, output, old_index, capture_index, signature, *, expected_counts=(647, 10682)):
    source = root / 'experiments/development_eval_v1_seed42'
    capture = root / 'experiments/raw_crop_capture_v1_seed42'
    mapping = {r['source_image']: r for r in capture_index['records']}
    parts = []; source_hashes = []; cache_index = []; crops = 0; reused = 0; started = time.perf_counter()
    for i, item in enumerate(old_index['records'], 1):
        r = mapping[item['source_image']]
        old_path = safe_path(source, item['path'])
        gt_path = source / 'reference_annotations' / (old_path.stem + '.txt')
        meta_path = safe_path(capture, r['path']); archive = meta_path.parent / 'queries_and_trace.npz'
        for path, digest in [(old_path, item['sha256']), (gt_path, item['annotation_sha256']),
                             (meta_path, r['sha256']), (archive, r['archive_sha256'])]:
            if sha256(path) != digest: raise ValueError('Kaynak dosya degismis veya eksik: ' + str(path))
            source_hashes.append((path, digest))
        meta = read_json(meta_path)
        if (meta.get('content_sha256') != fingerprint({k: v for k, v in meta.items() if k != 'content_sha256'}) or
                meta['protocol_sha256'] != CAPTURE_SIGNATURE or meta['source_image'] != item['source_image'] or
                meta['old11_prediction_sha256'] != item['sha256'] or
                meta['archive_sha256'] != r['archive_sha256'] or not meta['diagnostics']['original11_exactly_reproduced']):
            raise ValueError('11C kaynak kaydi farkli')
        old = cached_payload(old_path, SOURCE_PROTOCOL, item['source_image'])
        if (old['all_crops_processed'] != meta['crop_count'] or old['width'] != item['width'] or
                old['height'] != item['height'] or old['source_crop_document_sha256'] != meta['source_crop_document_sha256']):
            raise ValueError('11C / 11 crop kimligi farkli')
        bindings = {'protocol_sha256': signature, 'source_image': item['source_image'],
                    'old11_prediction_sha256': item['sha256'], 'annotation_sha256': item['annotation_sha256'],
                    'capture_metadata_sha256': r['sha256'], 'archive_sha256': r['archive_sha256']}
        folder = output / 'sources' / old_path.stem
        if folder.exists():
            data = load_cache(folder, bindings); reused += 1
        else:
            with np.load(archive, allow_pickle=False) as z:
                # No raw query decoding or GPU required: float64 mapped candidates are already frozen.
                candidates = z['candidate_rows']; origins = z['candidate_origins']
            data = analyze_parent(raw_annotation(gt_path.read_text(encoding='utf-8-sig')),
                                  old['width'], old['height'], candidates, origins, old['baseline'], old['hycado'])
            save_cache(folder, data, bindings)
        parts.append(data); crops += meta['crop_count']
        cache_index.append({'source_image': item['source_image'], 'summary_sha256': sha256(folder / 'summary.json'),
                            'npz_sha256': sha256(folder / 'matches.npz')})
        if i == 1 or i % 10 == 0 or i == len(old_index['records']):
            elapsed = time.perf_counter() - started
            print(f'CPU SKOR: {i}/{len(old_index["records"])}; tekrar kullanilan={reused}; gecen={elapsed/60:.1f} dk', flush=True)
            write_json(output / 'progress.json', {'status': 'ANALYZING', 'images_done': i, 'reused': reused,
                                                  'images_total': len(old_index['records']), 'elapsed_seconds': elapsed})
    if (len(parts), crops) != expected_counts: raise ValueError('Goruntu/crop kapsam esligi basarisiz')
    metrics = {}
    for name in VARIANTS:
        print('AP/AR hesaplaniyor: ' + name, flush=True)
        metrics[name] = aggregate(parts, name)
    original = read_json(source / 'metrics.json')
    compare_metrics(metrics['baseline'], original['baseline'])
    compare_metrics(metrics['hybrid_a100'], original['hycado'])
    old_counts = read_json(capture / 'capture_summary.json')['counts_at_fixed_max500']
    for variant, old_name in [('baseline', 'baseline'), ('crop_only', 'crop_only_fixed_NMS'), ('hybrid_a100', 'hybrid_original_11')]:
        for threshold in ['0.5', '0.75']:
            if metrics[variant]['counts_at_fixed_max500'][threshold] != old_counts[threshold][old_name]:
                raise ValueError('11C TP/FP sayaclari farkli')
    for path, digest in source_hashes:
        if sha256(path) != digest: raise ValueError('Kaynak dosya kosu sirasinda degismis')
    write_json(output / 'cache_index.json', {'protocol_sha256': signature, 'records': cache_index})
    report = {'protocol_sha256': signature, 'images': len(parts), 'crops': crops, 'variants': metrics,
              'baseline_and_alpha1_original11_metrics_equal': True, 'original11C_counts_equal': True,
              'official_binary_parity_verified': False, 'training_started': False, 'inference_started': False,
              'official_val_used': False, 'parameter_search': True,
              'scope': 'exploratory internal TRAIN development; new fusion protocol; no held-out claim',
              'best_observed_development_hybrid': max((n for n, _ in ALPHAS), key=lambda n: metrics[n]['AP']),
              'automatic_deployment_or_protocol_replacement': False}
    for name, values in metrics.items():
        values['delta_AP_vs_baseline_pp'] = values['AP'] - metrics['baseline']['AP']
        values['delta_AP_vs_original_hybrid_pp'] = values['AP'] - metrics['hybrid_a100']['AP']
    write_json(output / 'metrics.json', report)
    fields = ['variant', 'crop_score_multiplier', 'AP', 'AP50', 'AP75', 'AR1', 'AR10', 'AR100', 'AR500',
              'delta_AP_vs_baseline_pp', 'delta_AP_vs_original_hybrid_pp']
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for name in VARIANTS:
            writer.writerow({'variant': name, 'crop_score_multiplier': dict(ALPHAS).get(name, ''),
                             **{key: metrics[name][key] for key in fields[2:]}})
    with (output / 'per_class.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['variant'] + list(metrics['baseline']['per_class'][0]))
        writer.writeheader()
        for name in VARIANTS:
            for row in metrics[name]['per_class']: writer.writerow({'variant': name, **row})
    return report


def bundle(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output, prefix='11D_', suffix='.zip.tmp'); os.close(fd)
    target = output / '11D_skor_rapor.zip'
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in ['protocol.json', 'result.json', 'progress.json', 'environment.json',
                         'metrics.json', 'metrics.csv', 'per_class.csv', 'cache_index.json']:
                if (output / name).exists(): z.write(output / name, name)
            if (output / 'console.log').exists():
                with (output / 'console.log').open('rb') as f:
                    f.seek(max(0, (output / 'console.log').stat().st_size - 60000))
                    z.writestr('console_tail.log', f.read())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def run():
    from .__main__ import _Tee
    output = ROOT / 'experiments/score_ablation_v1_seed42'; output.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'experiments/development_eval_v1_seed42'
    capture = ROOT / 'experiments/raw_crop_capture_v1_seed42'
    with training_lock(output / '.analysis.lock'):
        with (output / 'console.log').open('a', encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
                try:
                    print('11D AYRI GELISTIRME DENEYI:', utc_now(), flush=True)
                    print('CPU; egitim/cikarim yok; yeni skor carpani karsilastirmasi. Sabit 11 korunur.', flush=True)
                    with training_lock(source / '.evaluation.lock'), training_lock(capture / '.capture.lock'):
                        _, _, old_index, capture_index = validate_capture(ROOT)
                        config = {'id': 'hycado_development_score_ablation_v1', 'date': '2026-10-29',
                                  'source11_protocol_sha256': SOURCE_PROTOCOL, 'capture_protocol_sha256': CAPTURE_SIGNATURE,
                                  'capture_index_sha256': CAPTURE_INDEX, 'capture_summary_sha256': CAPTURE_SUMMARY,
                                  'scope': 'new exploratory fusion experiment, internal TRAIN development only',
                                  'crop_score_multipliers': [[n, a] for n, a in ALPHAS], 'baseline_scores': 'unchanged',
                                  'controls': ['baseline', 'crop_only'], 'classwise_nms_iou': .7, 'max_output': 1000,
                                  'candidate_confidence': 'raw score strictly > .001; no cutoff after multiplication',
                                  'candidate_geometry_and_classes': 'unchanged frozen 11C float64 mapped candidates',
                                  'evaluation': 'unchanged 11 pixel ignore, caps 1/10/100/500, AP IoU .50:.05:.95',
                                  'reference_commit': REFERENCE_COMMIT,
                                  'official_binary_parity_verified': False, 'parameter_search': True,
                                  'training_started': False, 'inference_started': False, 'official_val_used': False,
                                  'automatic_deployment_or_protocol_replacement': False,
                                  'numpy': importlib.metadata.version('numpy'), 'code_sha256': sha256(__file__)}
                        signature = fingerprint(config)
                        if (output / 'protocol.json').exists() and read_json(output / 'protocol.json') != config:
                            raise ValueError('11D protokol/kod degismis; kayitlar karistirilmadi')
                        write_json(output / 'protocol.json', config); write_json(output / 'result.json', {'status': 'RUNNING'})
                        write_json(output / 'environment.json', {'created_utc': utc_now(), 'python': sys.version,
                                   'numpy': np.__version__, 'CPU_only': True})
                        for name in ['metrics.json', 'metrics.csv', 'per_class.csv']:
                            (output / name).unlink(missing_ok=True)
                        metadata_paths = [(base / name) for base, names in [(source, ['protocol.json', 'prediction_index.json', 'result.json', 'metrics.json', 'metrics.csv']),
                                           (capture, ['protocol.json', 'result.json', 'capture_summary.json', 'capture_index.json'])] for name in names]
                        before = {str(p): sha256(p) for p in metadata_paths}
                        report = process(ROOT, output, old_index, capture_index, signature)
                        if before != {str(p): sha256(p) for p in metadata_paths}: raise ValueError('Kaynak metadata degismis')
                        result = {'status': 'COMPLETED', 'images': report['images'], 'crops': report['crops'],
                                  'variants': len(VARIANTS), 'source_files_unchanged': True,
                                  'baseline_and_alpha1_original11_metrics_equal': True, 'original11C_counts_equal': True,
                                  'training_started': False, 'inference_started': False, 'official_val_used': False,
                                  'parameter_search': True, 'scope': report['scope']}
                except (Exception, KeyboardInterrupt) as exc:
                    traceback.print_exc()
                    result = {'status': 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'ERROR',
                              'error_type': type(exc).__name__, 'error': str(exc),
                              'resume': 'Rerun 11D; completed source summaries are reused'}
                write_json(output / 'result.json', result); write_json(output / 'progress.json', result)
                print('SONUC:', result['status'], flush=True)
        print('GONDERILECEK RAPOR:', bundle(output), flush=True)
        return result['status'] == 'COMPLETED'


def main():
    if '--report-only' in sys.argv:
        print(bundle(ROOT / 'experiments/score_ablation_v1_seed42'), flush=True); return 0
    return 0 if run() else 1


if __name__ == '__main__': raise SystemExit(main())
