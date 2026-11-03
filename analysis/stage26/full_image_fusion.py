"""Reproduce stage26 from immutable 19/24/25 ZIPs. CPU arrays only.

Run --inputs DIRECTORY --output DIRECTORY. Input filenames are recorded below.
No model loads, torch imports, inference, training, or old-result mutation.
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, sys, time, zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent / 'engine'))
from hycado_start import dev_metrics as dm, dev_evaluation as de, score_ablation as sa
from hycado_start.stage1 import fingerprint
from hycado_start.common import write_json, utc_now

NAMES = ['baseline', 'rtdetr_full960', 'fixed_grid_a025', 'hybrid_a025',
         'relocated_s0_a025', 'relocated_s1_a025', 'relocated_s2_a025']
VARIANTS = [('full_fusion_a100', 1.0), ('full_fusion_a025', .25)]
KEYS = ['AP', 'AP50', 'AP75', 'AR1', 'AR10', 'AR100', 'AR500']
EXPECTED = {
    '19_val_karsilastirma_rapor.zip': 'cb59c96eb299fa6b1ec0fe2118cbdbe12d262c653e24b14239fb64feb66af6df',
    '24_metrik_dogrulama_rapor.zip': '120c80c9c4f97f608953ea1f7ef5a3d79526deae5df450dd2b21004472621f67',
    '25_veri_model_denetim_rapor.zip': '0358f6a03b0a73d6bd9d275a29db9614639b2afea55978c5da172c6e16385e6e',
    'HyCADO_24_Referans_Dogrulama_v1.zip': '68f8e946b2eba59a19dcf646e744d7c1fab9f7aa8fe5b0ac4d2c91474ee3503d',
    'HyCADO_Adim2_Bilimsel_Karar_2026-11-03.md': 'ef4fe2a2622f2dc6299af09817d696c58994332baf91d02894cf57a540b91823',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def read(z, name):
    return json.loads(z.read(name))


def from_det8(a):
    result = a[:, :6].copy()
    result[:, 2:4] += result[:, :2]
    result[:, 5] -= 1
    return result.tolist()


def to_det8(rows):
    a = dm.xyxy_to_det(rows)
    return np.column_stack([a, np.full((len(a), 2), -1.)])


def fuse(baseline, full, width, height, alpha):
    rows = [list(r) for r in baseline]
    rows += [[*r[:4], r[4] * alpha, r[5]] for r in full]
    return de.classwise_nms(rows, width, height, threshold=.7, limit=1000)


def aggregate(parts, variant):
    denoms = np.sum([p['gt_denominators'] for p in parts], axis=0)
    presence = np.sum([p['presence'] for p in parts], axis=0)
    weights = presence / presence.sum(); ap = np.zeros((10, 10)); ar = np.zeros((10, 10, 4))
    for c in range(1, 11):
        for ti in range(10):
            for j, cap in enumerate([1, 10, 100, 500]):
                scores = []; flags = []
                for p in parts:
                    cls = p[variant + '_classes'][:cap]; keep = cls == c
                    scores.append(p[variant + '_scores'][:cap][keep])
                    flags.append(p[variant + '_flags'][:cap, ti][keep])
                area, recall = dm.precision_area(np.concatenate(scores), np.concatenate(flags), int(denoms[c-1]))
                ar[c-1, ti, j] = recall
                if j == 3:
                    ap[c-1, ti] = area
    return {'AP': float(np.sum(ap.mean(axis=1) * weights)), 'AP50': float(np.sum(ap[:, 0] * weights)),
            'AP75': float(np.sum(ap[:, 5] * weights)),
            **{f'AR{cap}': float(np.sum(ar[:, :, j].mean(axis=1) * weights)) for j, cap in enumerate([1, 10, 100, 500])},
            'class_macro_AP_separate_diagnostic': float(ap[presence > 0].mean()),
            'per_class_AP': ap.mean(axis=1).tolist(), 'presence_weights': weights.tolist(),
            'units': 'percent; differences in percentage points',
            'scope': 'literal pinned VisDrone toolkit aggregation, post-hoc previously examined VAL'}


def match_worker(job):
    image_id, gt, dets, path, token = job
    result = {'gt_denominators': np.array([np.count_nonzero(gt[:, 5] == c) for c in range(1, 11)], dtype=np.int64)}
    result['presence'] = (result['gt_denominators'] > 0).astype(np.int64)
    for variant, (rows, det) in dets.items():
        short = det[:500]; flags = []
        for t in np.linspace(.5, .95, 10):
            fast = sa.fast_flags(short, gt, float(t)); literal = np.zeros(len(short), dtype=np.int8)
            for c in range(1, 11):
                idx = np.flatnonzero(short[:, 5] == c)
                literal[idx] = dm.match_flags(short[idx, :5], gt[gt[:, 5] == c, :5], float(t))
            check(np.array_equal(fast, literal), 'Fast/literal matching difference: ' + str(image_id))
            flags.append(literal)
        result[variant + '_scores'] = short[:, 4]
        result[variant + '_classes'] = short[:, 5].astype(np.int16)
        result[variant + '_flags'] = np.column_stack(flags).astype(np.int8)
        result[variant + '_predictions_xyxy'] = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
    result['binding_token'] = np.array(token)
    temporary = str(path) + '.building.npz'
    np.savez_compressed(temporary, **result)
    os.replace(temporary, path)
    return image_id


def review25(z25, recs):
    r = read(z25, 'result.json'); summary = read(z25, 'data_summary.json'); rows = read(z25, 'image_inventory.json')
    check(r['status'] == 'COMPLETED' and r['audit_scope_passed'] and not r['open_items'], '25 audit not closed')
    check(digest(z25.read('code/provenance_audit.py')) == r['code_sha256'], '25 code identity')
    check(len(rows) == 7019 and len({r['source_image'] for r in rows}) == 7019, '25 source coverage')
    mapping = {r['source_image']: r for r in rows}
    for rec in recs:
        row = mapping[rec['image']]
        check(row['role'] == 'official_val' and row['sha256'] == rec['image_sha256'] and
              row['annotation_sha256'] == rec['annotation_sha256'], '25/24 VAL binding mismatch')
    for field in ['sha256', 'rgb_sha256']:
        groups = {}
        for row in rows:
            groups.setdefault(row[field], []).append(row)
        dup = [g for g in groups.values() if len(g) > 1]
        check(len(dup) == 3, 'Duplicate count differs')
        check(not any(len({r['role'] for r in g}) > 1 for g in dup), 'Cross-role duplicates')
    for model in ['yolo', 'crop', 'full']:
        m = read(z25, 'models/' + model + '.json')
        check(m['status'] == 'PASSED' and m['model']['head_nc'] == 10 and
              m['model']['all_state_tensors_finite'] and not m['CUDA_initialized'], '25 model not passed')
        if model != 'yolo':
            check(m['model']['head_queries'] == 300 and any(x[2] == 'HGStem' for x in m['initial_model_yaml']['backbone']),
                  '25 RT-DETR architecture identity')
    return {'status': 'PASSED', 'images': 7019, 'roles': summary['roles'],
            'VAL_548_identities_exact24': True, 'duplicate_groups_recomputed': 3, 'cross_role_duplicates': 0,
            'RTDETR_backbone': 'HGStem/HGBlock', 'models_passed': ['yolo', 'crop', 'full'],
            'invalid_geometry_rows': 3, 'invalid_positive_target_rows': 1, 'duplicate_target_rows': 4,
            'source_data_reaudited_here': False, 'scene_independence_verified': False}


def main(inputs, output, workers=4):
    output.mkdir(parents=True, exist_ok=True); (output / 'parts').mkdir(exist_ok=True)
    for name, expected in EXPECTED.items():
        check(digest((inputs / name).read_bytes()) == expected, 'Input archive identity: ' + name)
    z24 = zipfile.ZipFile(inputs / '24_metrik_dogrulama_rapor.zip')
    z19 = zipfile.ZipFile(inputs / '19_val_karsilastirma_rapor.zip')
    z25 = zipfile.ZipFile(inputs / '25_veri_model_denetim_rapor.zip')
    reference = zipfile.ZipFile(inputs / 'HyCADO_24_Referans_Dogrulama_v1.zip')
    cfg = read(z24, 'protocol.json'); records = cfg['records']; index = read(z24, 'input_index.json')['records']
    check(len(records) == len(index) == 548 and cfg['names'] == NAMES, '548 image order')
    expected_metrics = read(reference, 'HyCADO_24_Referans_Dogrulama/metric_comparison.json')
    check(read(reference, 'HyCADO_24_Referans_Dogrulama/result.json')['status'] == 'PASSED', 'Reference24 closure')
    p19 = read(z19, 'protocol.json'); m19 = {x['source_image']: x for x in read(z19, 'prediction_index.json')['records']}
    check(p19['records'] == records, '19/24 ordered source identities')
    write_json(output / 'audit25_review.json', review25(z25, records))
    protocol = {'id': 'hycado_full_image_fusion_v1', 'created_utc': utc_now(), 'input_archive_sha256': EXPECTED,
                'score_multipliers_prespecified_in_step2': [1.0, .25], 'NMS_iou': .7, 'max_output': 1000,
                'post_multiplier_score_cutoff': None, 'metric_maxDets': [1, 10, 100, 500],
                'reference_commit': dm.REFERENCE_COMMIT, 'canonical_image_order': [r['image'] for r in records],
                'literal_and_fast_flags_required_equal_for_every_new_image_variant_threshold': True,
                'new_training': False, 'new_GPU_inference': False, 'FPS_claim': False,
                'post_hoc': True, 'independent_test': False, 'script_sha256': digest(Path(__file__).read_bytes())}
    if (output / 'protocol.json').exists():
        old = read_json_local(output / 'protocol.json')
        check({k:v for k,v in old.items() if k != 'created_utc'} ==
              {k:v for k,v in protocol.items() if k != 'created_utc'}, 'Stage26 protocol changed')
        protocol = old
    write_json(output / 'protocol.json', protocol); signature = fingerprint(protocol)
    write_json(output / 'result.json', {'status': 'RUNNING'})
    source_parts = []; metadata = []; futures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (rec, item) in enumerate(zip(records, index)):
            check(item['source_image'] == rec['image'], 'Input index order')
            prefix = item['path'] + '/'; data = z24.read(prefix + 'input.mat')
            check(digest(data) == item['input_sha256'], '24 input MAT hash')
            bindings = read(z24, prefix + 'bindings.json'); binding_bytes = z24.read(prefix + 'bindings.json')
            check(digest(binding_bytes) == item['bindings_sha256'] and
                  bindings['content_sha256'] == fingerprint({k:v for k,v in bindings.items() if k!='content_sha256'}), '24 binding identity')
            annotation = z24.read(prefix + 'annotation.txt')
            check(digest(annotation) == rec['annotation_sha256'] == item['annotation_sha256'], '24 annotation hash')
            mat = loadmat(io.BytesIO(data)); raw = dm.raw_annotation(annotation.decode('utf-8-sig'))
            check(np.array_equal(raw, mat['rawgt']), '24 raw GT exact')
            integral = dm.ignore_integral(raw, rec['width'], rec['height']); gt = dm.prepare_groundtruth(raw, integral)
            check(np.array_equal(gt, mat['python_gt']), '24 GT preprocessing exact')
            baseline = from_det8(mat['raw0'])
            baseline_array = np.asarray(baseline, dtype=np.float64).reshape(-1, 6)
            check(np.array_equal(baseline_array[:, :4], baseline_array[:, :4].astype(np.float32).astype(np.float64)),
                  'YOLO original float32 coordinate reconstruction')
            check(np.array_equal(to_det8(baseline), mat['raw0']), 'YOLO reversible xyxy/xywh mapping')
            p = m19[rec['image']]; b19 = z19.read(p['path'])
            check(digest(b19) == p['sha256'] == bindings['bindings']['source_prediction_sha256']['19'], '19 prediction source hash')
            doc = json.loads(b19)
            check(doc['content_sha256'] == fingerprint({k:v for k,v in doc.items() if k!='content_sha256'}) and
                  doc['protocol_sha256'] == fingerprint(p19) and doc['image_sha256'] == rec['image_sha256'], '19 signed prediction')
            full = doc['rtdetr_full960']; check(np.array_equal(to_det8(full), mat['raw1']), '19 rows exact24')
            source = {'gt_denominators': mat['python_denominators'].reshape(10).astype(np.int64)}
            source['presence'] = (source['gt_denominators'] > 0).astype(np.int64)
            for vi, name in enumerate(NAMES):
                det = dm.outside_ignore(mat['raw' + str(vi)], integral)
                check(np.array_equal(det, mat['python_det' + str(vi)]), '24 detector preprocessing exact')
                source[name + '_scores'] = det[:500, 4]; source[name + '_classes'] = det[:500, 5].astype(np.int16)
                source[name + '_flags'] = mat['python_flags' + str(vi)].astype(np.int8)
            source_parts.append(source)
            dets = {}
            for variant, alpha in VARIANTS:
                rows = fuse(baseline, full, rec['width'], rec['height'], alpha)
                dets[variant] = rows, dm.outside_ignore(dm.xyxy_to_det(rows), integral)
            token = fingerprint({'protocol': signature, 'id': i, 'record': rec, 'source_mat_sha256': item['input_sha256']})
            path = output / 'parts' / f'{i:04d}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as prior:
                    check(str(prior['binding_token']) == token, 'Stage26 cached part binding')
            else:
                futures.append(pool.submit(match_worker, (i, gt, dets, path, token)))
            metadata.append({'id': i, 'source_image': rec['image'], 'image_sha256': rec['image_sha256'],
                             'annotation_sha256': rec['annotation_sha256'], 'source24_mat_sha256': item['input_sha256'],
                             'source19_prediction_sha256': p['sha256'], 'binding_token': token})
            if i == 0 or (i+1) % 25 == 0 or i+1 == len(records):
                print('26 input/NMS prepared', i+1, '/', len(records), flush=True)
        for i, f in enumerate(futures, 1):
            f.result()
            if i % 25 == 0 or i == len(futures):
                print('26 new literal matching completed', i, '/', len(futures), flush=True)
    reproduced = {v: aggregate(source_parts, v) for v in NAMES}
    max_difference = max(abs(reproduced[v][k] - expected_metrics[v]['python'][k]) for v in NAMES for k in KEYS)
    check(max_difference < 1e-9, 'Existing49 metrics no longer reproduced')
    write_json(output / 'source_reproduction.json', {'all_49_metric_differences_below_1e_9_pp': True,
               'max_absolute_difference_pp': max_difference, 'variants': reproduced})
    new_parts = []
    for row in metadata:
        path = output / 'parts' / f"{row['id']:04d}.npz"; row['npz_sha256'] = digest(path.read_bytes())
        with np.load(path, allow_pickle=False) as a:
            check(str(a['binding_token']) == row['binding_token'], 'Final part binding mismatch')
            new_parts.append({k: a[k].copy() for k in a.files})
    metrics = {v: aggregate(new_parts, v) for v, _ in VARIANTS}
    write_json(output / 'metrics.json', metrics); write_json(output / 'part_index.json', {'records': metadata})
    for name, expected in EXPECTED.items():
        check(digest((inputs / name).read_bytes()) == expected, 'Input mutated during run')
    write_json(output / 'result.json', {'status': 'COMPLETED', 'images': 548, 'new_fusion_variants': 2,
               'all_new_matching_flags_literal_fast_exact': True, 'all_existing_49_metrics_reproduced': True,
               'old_inputs_unchanged': True, 'training_started': False, 'GPU_inference': False,
               'new_reference_Octave_execution_performed': False, 'historical24_reference_source_parity': True,
               'bootstrap_new_variants_performed': False, 'runtime_new_variants_measured': False,
               'independent_new_test': False})
    print('26 COMPLETED', json.dumps(metrics), flush=True)


def read_json_local(path):
    return json.loads(path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    main(args.inputs, args.output, args.workers)
