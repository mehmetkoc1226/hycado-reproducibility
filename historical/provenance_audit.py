"""25: read-only CPU model/data audit. Never train, infer, or alter old runs."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sqlite3
import struct
import subprocess
import sys
import tempfile
import traceback
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

OUT = 'provenance_audit_v1'
REPORT = '25_veri_model_denetim_rapor.zip'
SCHEMA = 'hycado_provenance_audit_v1'
OFFICIAL_RT = '6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f'
MODELS = {
    'yolo': {'folder': 'stage1_v1_seed42', 'kind': 'yolo',
             'protocol': '159ea5fec31f25542b18417823cbefb74ed4a88c442050fec64dece7365ab6be',
             'weights': '3b61a12a89a448b849bc965e44f00afa92042a5fc0928b4c4f801e0182f84149',
             'initial': 'yolov9c.pt'},
    'crop': {'folder': 'rtdetr_crop_v1_seed42', 'kind': 'rtdetr',
             'protocol': '6ae4d2ae2c503b18fe467d3c275605d527bd8934e8f6128276daeaa5327491ae',
             'weights': 'fa6e3991b26650e2426a4bbac9253351d47087e2533d3b98d86a2f24b586cbf9',
             'initial': 'rtdetr-l.pt'},
    'full': {'folder': 'rtdetr_full_image_v1_seed42', 'kind': 'rtdetr',
             'protocol': '0137544e585fa48eee80b1bc1713c77f7ca10b75560876420642f9814cf4643f',
             'weights': '0d7f046673037e83accccf8c430f6978799f59557a93f3be4a7ac2ea182eb57c',
             'initial': 'rtdetr-l.pt'},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(root, relative):
    require(isinstance(relative, str), 'Dosya yolu metin olmali')
    p = PurePosixPath(relative)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in relative
            and ':' not in relative, 'Guvenli olmayan kaynak yolu: ' + relative)
    path = (Path(root) / p).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Kaynak yolu klasor disinda')
    return path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def signature(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


class Cache:
    """Atomic per-file recovery, bound to code, decoder version and file stats.

    Reuse assumes no hostile timestamp restoration. This is a recovery cache,
    not an independent proof that cached source bytes were reread on this run.
    """
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS files '
                        '(key TEXT PRIMARY KEY, value TEXT NOT NULL, digest TEXT NOT NULL)')
        self.namespace = fingerprint({'schema': SCHEMA, 'code': sha256(__file__),
                                      'Pillow': version('Pillow'), 'decode': 'RGB,no EXIF transpose'})
        self.reused = self.new = 0

    def get(self, path, kind, producer):
        path = Path(path).resolve(); before = signature(path)
        # Hard-linked copies share results, while each caller checks its own path.
        identity = before if before[1] else [str(path), *before]
        key = fingerprint([self.namespace, kind, identity])
        row = self.db.execute('SELECT value,digest FROM files WHERE key=?', (key,)).fetchone()
        if row:
            require(hashlib.sha256(row[0].encode()).hexdigest() == row[1],
                    '25 devam onbellegi bozuk; kaynaklar degistirilmedi')
            result = json.loads(row[0]); self.reused += 1
        else:
            result = producer(path)
            require(signature(path) == before, 'Okunurken dosya degisti: ' + str(path))
            value = json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False)
            self.db.execute('INSERT OR REPLACE INTO files VALUES (?,?,?)',
                            (key, value, hashlib.sha256(value.encode()).hexdigest()))
            self.db.commit(); self.new += 1
        require(signature(path) == before, 'Denetim sirasinda dosya degisti: ' + str(path))
        return result

    def digest(self, path):
        return self.get(path, 'bytes', lambda p: {'sha256': sha256(p)})['sha256']

    def image(self, path):
        def produce(p):
            from PIL import Image
            with Image.open(p) as im:
                rgb = im.convert('RGB'); width, height = rgb.size
                h = hashlib.sha256(b'HyCADO-RGB-v1\0' + struct.pack('>II', width, height))
                h.update(rgb.tobytes())
                pixels = h.hexdigest()
            return {'sha256': sha256(p), 'rgb_sha256': pixels,
                    'width': width, 'height': height}
        return self.get(path, 'image', produce)

    def close(self):
        self.db.close()


@contextlib.contextmanager
def lock(path):
    with Path(path).open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def protocol_inputs(root, cache, specs=None):
    specs = MODELS if specs is None else specs
    files = {}; protocols = {}

    def checked(path, expected=None):
        digest = cache.digest(path)
        if expected:
            require(digest == expected, 'Kaynak hash farkli: ' + str(path))
        files[path.relative_to(root).as_posix()] = digest
        return read_json(path)

    for name, spec in specs.items():
        folder = root / 'experiments' / spec['folder']
        p = checked(folder / 'protocol.json')
        require(fingerprint(p) == spec['protocol'], name + ': egitim protokolu farkli')
        index = checked(folder / 'checkpoint_index.json')
        require(index['protocol_sha256'] == spec['protocol'] and index['latest']['epoch'] == 80
                and index['latest']['sha256'] == spec['weights'], name + ': son checkpoint farkli')
        result = checked(folder / 'result.json')
        require(result['status'] == 'COMPLETED', name + ': tamamlanmis egitim gerekli')
        for filename, digest in p['code_hashes'].items():
            require(Path(filename).name == filename, 'Kod adi gecersiz')
            path = safe(root, 'hycado_start/' + filename)
            require(cache.digest(path) == digest, 'Dondurulmus kod farkli: ' + filename)
            files[path.relative_to(root).as_posix()] = digest
        for package, expected in p.get('runtime_versions', {}).items():
            require(version(package) == expected, 'Sabit paket surumu farkli: ' + package)
        protocols[name] = p
    prepared = checked(root / 'data/prepared/manifest.json')
    stage = checked(root / 'experiments/stage1_v1_seed42/data/manifest.json')
    require(files['experiments/stage1_v1_seed42/data/manifest.json'] ==
            protocols['yolo']['dataset_manifest_sha256'], 'Stage1 manifest kimligi farkli')
    require(stage['source_manifest_sha256'] == files['data/prepared/manifest.json'],
            'Prepared manifest kaynak bagi farkli')
    crop_root = root / 'experiments/crop_dataset_v1_seed42'
    p8 = checked(crop_root / 'protocol.json'); catalog = protocols['crop']['catalog']
    require(fingerprint(p8) == catalog['source08_protocol_sha256'], '08 protokol kimligi farkli')
    crop_index = checked(crop_root / 'source_index.json', catalog['source_index_sha256'])
    checked(crop_root / 'dataset_summary.json', catalog['summary_sha256'])
    require(crop_index['protocol_sha256'] == fingerprint(p8), '08 indeks kimligi farkli')
    full_root = root / 'experiments/rtdetr_full_image_v1_seed42'
    full_index = checked(full_root / 'data/sample_index.json')
    require(full_index['protocol_sha256'] == protocols['full']['data_protocol_sha256'],
            '18 veri kimligi farkli')
    full_protocol = checked(full_root / 'data/protocol.json')
    require(fingerprint(full_protocol) == protocols['full']['data_protocol_sha256'],
            '18 veri protokolu farkli')
    full_dev = checked(full_root / 'development_plan.json')
    require(fingerprint(full_dev) == protocols['full']['development_plan_sha256'],
            '18 gelistirme plani farkli')
    c12 = checked(root / 'experiments/official_val_a025_v1_seed42/protocol.json')
    require(fingerprint(c12) == 'ae0ae6f46e530d4a2cdd3f699f490016a659a574e2fe3f7da0b3f6b947eafdd2',
            '12 VAL protokolu farkli')
    require(c12['stage1_weights_sha256'] == specs['yolo']['weights'] and
            c12['crop_weights_sha256'] == specs['crop']['weights'] and
            c12['prepared_manifest_sha256'] == files['data/prepared/manifest.json'],
            'VAL model/veri baglari farkli')
    policy = p8['decision']
    require(policy['selected_configuration_id'] == 'c3_near_z3_b10' and
            c12['crop_score_multiplier'] == .25, 'Sabit yontem karari farkli')
    for filename, digest in policy['merge']['source_code_hashes'].items():
        path = safe(root, 'hycado_start/' + filename)
        require(cache.digest(path) == digest, 'Geometri kodu farkli: ' + filename)
        files[path.relative_to(root).as_posix()] = digest
    return {'prepared': prepared, 'stage': stage, 'crop_index': crop_index,
            'full_index': full_index, 'full_protocol': full_protocol, 'full_dev': full_dev,
            'protocols': protocols, 'files': files, 'crop_policy': policy,
            'val_protocol': c12}


def role_map(prepared, stage, expected=(6471, 548, 5824, 647)):
    require(prepared['purpose'] == 'fresh_original_visdrone_conversion' and
            not prepared.get('is_small_fixture'), 'Orijinal prepared manifest gerekli')
    records = prepared['records']; rows = stage['records']
    originals = {r['image']: r for r in records}
    require(len(originals) == len(records), 'Prepared kaynak yolu tekrari')
    require(Counter(r['split'] for r in records) == {'train': expected[0], 'val': expected[1]},
            'Prepared TRAIN/VAL sayilari farkli')
    require(Counter(r['split'] for r in rows) == {'train': expected[2], 'val': expected[3]},
            '5824/647 ic ayrim sayilari farkli')
    by_source = {r['source_image']: r for r in rows}
    require(len(by_source) == len(rows) and set(by_source) ==
            {r['image'] for r in records if r['split'] == 'train'}, 'TRAIN kapsam baglari farkli')
    groups = defaultdict(set); mapping = {}
    for key, r in originals.items():
        require(r['split'] in ('train', 'val') and
                PurePosixPath(key).parts[:2] == ('images', r['split']) and
                len(PurePosixPath(key).parts) == 3, 'Prepared yol/rol farkli')
        if r['split'] == 'val':
            mapping[key] = 'official_val'; continue
        s = by_source[key]
        require(s['source_split'] == 'train' and s['image_sha256'] == r['image_sha256']
                and s['annotation_sha256'] == r['annotation_sha256'], 'Stage kaynak bagi farkli')
        require(PurePosixPath(s['image']).parts == ('images', s['split'], PurePosixPath(key).name),
                'Stage kaynak goruntu yolu farkli')
        mapping[key] = 'training' if s['split'] == 'train' else 'development'
        groups[s['split']].add(s['group'])
    require(not groups['train'] & groups['val'], 'Kayitli grup train/dev kesisimi')
    return mapping, by_source


def duplicate_groups(records, field):
    grouped = defaultdict(list)
    for r in records:
        grouped[r[field]].append(r)
    result = []
    for digest, members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        roles = sorted({r['role'] for r in members})
        result.append({'digest': digest, 'roles': roles, 'cross_role': len(roles) > 1,
                       'annotation_hashes_equal': len({r['annotation_sha256'] for r in members}) == 1,
                       'members': [{'source_image': r['source_image'], 'role': r['role'],
                                    'annotation_sha256': r['annotation_sha256']} for r in members]})
    return result


def validate_inventory(directory, expected):
    actual = {p.name for p in Path(directory).iterdir() if p.is_file() and
              p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}}
    require(actual == set(expected), 'Goruntu listesi eklenmis/silinmis: ' + str(directory))


def audit_data(root, raw_root, output, inputs, cache, expected=(6471, 548, 5824, 647)):
    from .data import parse_annotation
    from .stability import inspect_labels
    from .experiment_data import grouped_split
    prepared = root / 'data/prepared'; stage_root = root / 'experiments/stage1_v1_seed42/data'
    mapping, stage = role_map(inputs['prepared'], inputs['stage'], expected)
    originals = sorted(inputs['prepared']['records'], key=lambda r: r['image'])
    require(inputs['stage'].get('split_seed') == 20261022, 'Ayrim tohumu farkli')
    split_rows = [{**r, 'class_counts': {}} for r in originals if r['split'] == 'train']
    train, dev = grouped_split(split_rows, seed=20261022)
    require({r['image'] for r in train} == {k for k, v in mapping.items() if v == 'training'} and
            {r['image'] for r in dev} == {k for k, v in mapping.items() if v == 'development'},
            'Kayitli ayrim, sabit kuralla yeniden uretilemiyor')
    for r in train + dev:
        require(stage[r['image']]['group'] == r['group'], 'Yeniden uretilen grup kimligi farkli')
    missing_raw = []; rows = []; issues = []; totals = Counter()
    raw_complete = {}
    for split in ('train', 'val'):
        names = [Path(r['image']).name for r in originals if r['split'] == split]
        validate_inventory(prepared / 'images' / split, names)
        raw_dir = raw_root / ('VisDrone2019-DET-' + split)
        raw_complete[split] = (raw_dir / 'images').is_dir() and (raw_dir / 'annotations').is_dir()
        if raw_complete[split]:
            validate_inventory(raw_dir / 'images', names)
    for split in ('train', 'val'):
        validate_inventory(stage_root / 'images' / split,
                           [Path(r['image']).name for r in stage.values() if r['split'] == split])
    for i, r in enumerate(originals, 1):
        image = safe(prepared, r['image']); role = mapping[r['image']]; stem = image.stem
        pixels = cache.image(image)
        require(pixels['sha256'] == r['image_sha256'] and
                (pixels['width'], pixels['height']) == (r['width'], r['height']),
                'Prepared goruntu hash/boyut farkli: ' + r['image'])
        annotation = prepared / 'visdrone_annotations' / r['split'] / (stem + '.txt')
        require(cache.digest(annotation) == r['annotation_sha256'], 'Ham annotation degismis: ' + stem)
        converted = parse_annotation(annotation.read_text(encoding='utf-8-sig'), r['width'], r['height'], source=r['image'])
        label = prepared / 'labels' / r['split'] / (stem + '.txt')
        expected_label = ''.join(str(x[0]) + ' ' + ' '.join(f'{v:.10f}' for v in x[1:]) + '\n'
                                 for x in converted['labels'])
        require(label.read_text(encoding='utf-8') == expected_label, 'Prepared etiket donusumu farkli: ' + stem)
        cleaned, duplicates, n_targets = inspect_labels(expected_label, r['image'])
        totals['invalid_geometry_rows'] += len(converted['issues'])
        totals['invalid_target_rows'] += sum(x['is_target'] for x in converted['issues'])
        totals['duplicate_target_rows'] += len(duplicates)
        if converted['issues'] or duplicates:
            issues.append({'source_image': r['image'], 'role': role,
                           'invalid_geometry': converted['issues'], 'duplicate_targets': duplicates})
        if r['split'] == 'train':
            s = stage[r['image']]
            require(cache.digest(safe(stage_root, s['image'])) == r['image_sha256'], 'Stage goruntusu farkli: ' + stem)
            require(cache.digest(stage_root / 'annotations' / s['split'] / (stem + '.txt')) == r['annotation_sha256'],
                    'Stage annotation farkli: ' + stem)
            s_label = stage_root / 'labels' / s['split'] / (stem + '.txt')
            require(cache.digest(s_label) == s['label_sha256'] and
                    s_label.read_text(encoding='utf-8') == cleaned and n_targets == s['gt_count'],
                    'Egitim etiketi/GT sayisi farkli: ' + stem)
        raw_status = 'VERIFIED'
        raw_dir = raw_root / ('VisDrone2019-DET-' + r['split'])
        raw_image = raw_dir / 'images' / image.name; raw_anno = raw_dir / 'annotations' / (stem + '.txt')
        if raw_image.is_file() and raw_anno.is_file():
            require(cache.digest(raw_image) == pixels['sha256'] and cache.digest(raw_anno) == r['annotation_sha256'],
                    'Orijinal raw/prepared hash farkli: ' + r['image'])
            # Byte identity proves these original and prepared decoded inputs agree.
        else:
            raw_status = 'NOT_AVAILABLE'; missing_raw.append(r['image'])
        rows.append({'source_image': r['image'], 'role': role, 'annotation_sha256': r['annotation_sha256'],
                     **pixels, 'raw_source_status': raw_status})
        if i == 1 or i % 50 == 0 or i == len(originals):
            atomic_json(output / 'progress.json', {'status': 'DATA_AUDIT', 'completed_images': i,
                        'total_images': len(originals), 'cache_reuses': cache.reused, 'new_checks': cache.new})
            print(f'25 veri denetimi: {i}/{len(originals)}; devam kaydi={cache.reused}', flush=True)
    # Source-level crop role and immutable metadata audit: no 287k crop reread.
    crop_root = root / 'experiments/crop_dataset_v1_seed42'; crop_rows = inputs['crop_index']['records']
    require(len(crop_rows) == len(stage) and {r['source_image'] for r in crop_rows} == set(stage),
            '08 parent kapsam farkli')
    eligible = Counter()
    for i, r in enumerate(crop_rows, 1):
        s = stage[r['source_image']]
        require(r['split'] == s['split'] and r['group'] == s['group'], '08 parent rol/grup farkli')
        p = safe(crop_root, r['source_record'])
        require(cache.digest(p) == r['sha256'], '08 parent metadata hash farkli')
        d = read_json(p)
        require(d['content_sha256'] == fingerprint({k: v for k, v in d.items() if k != 'content_sha256'})
                and d['source_image'] == r['source_image'] and d['split'] == r['split']
                and d['group'] == r['group'] and d['protocol_sha256'] == inputs['crop_index']['protocol_sha256']
                and len(d['crops']) == d['crop_count'] == r['crop_count'], '08 parent icerik farkli')
        if any(c['supervision_state'] != 'fully_ignored_no_positive' for c in d['crops']):
            eligible[r['split']] += 1
        if i == 1 or i % 1000 == 0 or i == len(crop_rows):
            print(f'25 crop parent baglari: {i}/{len(crop_rows)}', flush=True)
    require(eligible['train'] == inputs['protocols']['crop']['samples_per_round'], 'Crop egitim kaynak sayisi farkli')
    full_rows = inputs['full_index']['records']; full_by_source = {r['source_image']: r for r in full_rows}
    full_dev = inputs['full_dev']
    require(len(full_by_source) == len(full_rows) and len(full_dev) == 32, '18 parent tekrari/gelistirme sayisi farkli')
    require({r['source_image'] for r in full_rows if r['split'] == 'train'} ==
            {k for k, v in mapping.items() if v == 'training'}, '18 egitim parent kapsami farkli')
    require({r['source_image'] for r in full_dev} ==
            {r['source_image'] for r in full_rows if r['split'] == 'val'}, '18 gelistirme parent kapsami farkli')
    for r in full_rows:
        s = stage[r['source_image']]
        require(r['split'] == s['split'] and r['group'] == s['group'] and r['crop_index'] == 0,
                '18 parent rol/grup farkli')
    byte_groups = duplicate_groups(rows, 'sha256'); pixel_groups = duplicate_groups(rows, 'rgb_sha256')
    cross_role = [g for g in pixel_groups if g['cross_role']]
    summary = {'execution_status': 'COMPLETED', 'data_integrity': 'PASSED', 'images': len(rows),
               'roles': dict(Counter(r['role'] for r in rows)), 'split_reproduced': True,
               'raw_source_files_available_for_all': not missing_raw,
               'missing_raw_images': missing_raw, 'raw_directory_inventory_checked': raw_complete,
               'exact_byte_duplicate_groups': len(byte_groups), 'exact_RGB_duplicate_groups': len(pixel_groups),
               'cross_role_RGB_duplicate_groups': len(cross_role),
               'geometry_and_duplicate_target_counts': dict(totals), 'eligible_crop_parents': dict(eligible),
               'full_image_diagnostic_development_parents': 32,
               'scene_independence_verified': False, 'perceptual_similarity_audit_performed': False,
               'all_crop_PNGs_and_annotations_rehashed': False,
               'all_full_image_training_PNGs_and_annotations_rehashed': False,
               'scope': 'original/prepared full-image and stage1 label integrity; split reproduction; '
                        'exact RGB/byte duplicates; crop/full source-role bindings, not every derived pixel',
               'cache': {'reused_file_results': cache.reused, 'new_file_results': cache.new,
                         'reuse_rule': 'code+Pillow+device/inode+size+mtime_ns+ctime_ns; no hostile stat restoration'}}
    atomic_json(output / 'data_summary.json', summary)
    atomic_json(output / 'duplicate_groups.json', {'bytes': byte_groups, 'RGB': pixel_groups})
    atomic_json(output / 'label_issues.json', issues)
    atomic_json(output / 'image_inventory.json', rows)
    return summary


def serializable(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, type):
        return value.__module__ + '.' + value.__qualname__
    raise TypeError('Model YAML JSON donusumu desteklenmiyor: ' + type(value).__name__)


def model_description(model, torch):
    state = model.state_dict()
    bad = [k for k, v in state.items() if torch.is_tensor(v) and
           (v.is_floating_point() or v.is_complex()) and not bool(torch.isfinite(v).all())]
    require(not bad, 'Model/EMA sonlu olmayan tensorlar: ' + ','.join(bad[:5]))
    head = model.model[-1]
    nc = getattr(head, 'nc', None)
    require(nc == 10, 'Son model head nc=10 olmali; bulunan: ' + str(nc))
    modules = [{'path': n, 'class': type(m).__module__ + '.' + type(m).__name__}
               for n, m in model.named_modules()]
    return {'class': type(model).__module__ + '.' + type(model).__name__,
            'parameters': sum(p.numel() for p in model.parameters()), 'head_nc': nc,
            'head_queries': getattr(head, 'num_queries', None),
            'model_yaml': serializable(model.yaml), 'modules': modules,
            'state_shapes': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in state.items()},
            'all_state_tensors_finite': True}


def inspect_checkpoint(root, spec):
    # Caller runs this in a fresh process with CUDA hidden, config/output local.
    import torch
    require(not torch.cuda.is_initialized(), 'CPU denetiminden once CUDA baslatilmis')
    torch.set_num_threads(2)
    from ultralytics.nn.tasks import RTDETRDetectionModel
    folder = root / 'experiments' / spec['folder']; p = read_json(folder / 'protocol.json')
    index = read_json(folder / 'checkpoint_index.json'); item = index['latest']
    require(fingerprint(p) == spec['protocol'] and index['protocol_sha256'] == spec['protocol']
            and item['epoch'] == 80 and item['sha256'] == spec['weights'], 'Model protokol/checkpoint bagi farkli')
    weights = safe(folder, item['path']); before = signature(weights)
    require(sha256(weights) == spec['weights'], 'Son egitim agirligi hash farkli')
    initial = root / 'work/fresh_official_pretrained' / spec['initial']; initial_before = signature(initial)
    init_hash = sha256(initial); provenance = read_json(initial.with_suffix('.provenance.json'))
    require(init_hash == provenance['sha256'], 'Resmi baslangic kaynagi hash farkli')
    if spec['kind'] == 'rtdetr':
        require(init_hash == OFFICIAL_RT == p['initial_weights_sha256'], 'RT-DETR baslangic kimligi farkli')
    else:
        identity = read_json(folder / 'model_identity.json')
        require(init_hash == identity['official_initial_weights_sha256'], 'YOLO baslangic kimligi farkli')
    require(signature(weights) == before, 'Checkpoint okunurken degisti')
    saved = torch.load(weights, map_location='cpu', weights_only=False)
    require(saved['protocol_sha256'] == spec['protocol'] and len(saved['history']) == 80,
            'Checkpoint ic protokol/tur sayisi farkli')
    import copy
    if spec['kind'] == 'yolo':
        require(saved['epoch'] == 79, 'YOLO checkpoint 0-tabanli epoch=79 olmali')
        model = saved['ema']
        require(isinstance(model, torch.nn.Module), 'YOLO EMA model nesnesi gerekli')
        description = model_description(model, torch)
        train_description = model_description(saved['model'], torch)
        require(description['state_shapes'] == train_description['state_shapes'], 'YOLO model/EMA yapilari farkli')
        initial_saved = torch.load(initial, map_location='cpu', weights_only=False)
        initial_model = initial_saved.get('ema')
        if initial_model is None:
            initial_model = initial_saved['model']
        initial_yaml = serializable(initial_model.yaml)
        strict = 'serialized final EMA; train/EMA state structure equal'
    else:
        require(saved['schema'] == 'hycado_crop_experiment_checkpoint_v1' and saved['round'] == 80,
                'RT-DETR checkpoint semasi/turu farkli')
        pretrained = torch.load(initial, map_location='cpu', weights_only=False)
        initial_model = pretrained.get('ema')
        if initial_model is None:
            initial_model = pretrained['model']
        initial_yaml = serializable(initial_model.yaml)
        model = RTDETRDetectionModel(copy.deepcopy(initial_model.yaml), nc=10, ch=3, verbose=False)
        model.load_state_dict(saved['model'], strict=True)
        model_description(model, torch)  # finite train-state check, no forward
        model.load_state_dict(saved['ema'], strict=True)
        description = model_description(model, torch)
        require(description['head_queries'] == 300, 'RT-DETR sorgu sayisi 300 olmali')
        strict = 'official checkpoint YAML -> nc10 architecture -> strict train and EMA state loads'
    for key in ['backbone', 'head', 'depth_multiple', 'width_multiple', 'scales', 'scale', 'activation']:
        require(description['model_yaml'].get(key) == initial_yaml.get(key),
                'Resmi kaynak ve son model YAML yapisi farkli: ' + key)
    require(signature(weights) == before and sha256(weights) == spec['weights'], 'Model dosyasi denetimde degisti')
    require(signature(initial) == initial_before and sha256(initial) == init_hash,
            'Baslangic dosyasi denetimde degisti')
    require(not torch.cuda.is_initialized(), 'CPU denetiminde CUDA baslatilmis')
    return {'status': 'PASSED', 'checkpoint_sha256': spec['weights'], 'protocol_sha256': spec['protocol'],
            'initial_sha256': init_hash, 'initial_model_yaml': initial_yaml,
            'selected_weights': 'fixed final round80 EMA', 'architecture_check': strict,
            'model': description, 'inference_calls': 0, 'optimizer_steps': 0, 'CUDA_initialized': False,
            'torch': str(torch.__version__), 'ultralytics': version('ultralytics')}


def worker(root, name):
    output = root / 'experiments' / OUT
    config = output / 'ultralytics_config'
    config.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(config)
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    try:
        result = inspect_checkpoint(root, MODELS[name])
    except Exception as exc:
        result = {'status': 'FAILED', 'error': str(exc), 'traceback': traceback.format_exc()}
    atomic_json(output / 'models' / (name + '.json'), result)
    return 0 if result['status'] == 'PASSED' else 1


def run_models(root):
    output = root / 'experiments' / OUT; results = {}
    (output / 'ultralytics_config').mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        path = output / 'models' / (name + '.json'); path.unlink(missing_ok=True)
        print('25 CPU model kimligi: ' + name + ' (egitim/cikarim yok)', flush=True)
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = ''
        env['YOLO_CONFIG_DIR'] = str(output / 'ultralytics_config')
        env['YOLO_AUTOINSTALL'] = 'false'; env['OMP_NUM_THREADS'] = '2'; env['MKL_NUM_THREADS'] = '2'
        command = [sys.executable, '-B', '-m', 'hycado_start.provenance_audit',
                   '--root', str(root), '--worker-model', name]
        code = subprocess.call(command, cwd=root, env=env)
        result = read_json(path) if path.exists() else {'status': 'FAILED', 'error': 'CPU model alt sureci sonuclanamadi', 'exit_code': code}
        if code != 0 and result.get('status') == 'PASSED':
            result = {'status': 'FAILED', 'error': 'Model sureci cikis kodu farkli', 'exit_code': code}
        results[name] = result
        print('25 model ' + name + ': ' + result['status'], flush=True)
    atomic_json(output / 'model_summary.json', {k: {x: v[x] for x in ['status', 'error'] if x in v}
                                              for k, v in results.items()})
    return results


def final_result(data, models):
    open_items = []
    if data.get('data_integrity') != 'PASSED':
        open_items.append('Veri butunluk denetimi tamamlanamadi')
    if not data.get('raw_source_files_available_for_all'):
        open_items.append('Bazi orijinal raw kaynaklar mevcut degil')
    if data.get('cross_role_RGB_duplicate_groups', 0):
        open_items.append('Roller arasinda ayni RGB goruntuler bulundu; etkilenen ornekler incelenmeli')
    if any(v.get('status') != 'PASSED' for v in models.values()) or len(models) != len(MODELS):
        open_items.append('En az bir model kimligi denetimi gecmedi')
    return {'status': 'COMPLETED_WITH_OPEN_ITEMS' if open_items else 'COMPLETED',
            'audit_scope_passed': not open_items, 'open_items': open_items,
            'training_started': False, 'GPU_inference': False, 'source_files_written': False,
            'scene_independence_verified': False, 'manuscript_ready_for_submission': False,
            'note': 'COMPLETED yalniz bu denetimin kapsamidir; sahne bagimsizligi veya yayin onayi degildir'}


def write_readme(output, result):
    text = ('# 25 veri ve model denetimi\n\nDurum: **' + result['status'] + '**\n\n'
            'Bu kosuda egitim ve model cikarimi yapilmadi. Onceki sonuclar degistirilmedi.\n\n'
            'Model ayrintilari models/*.json, veri ozeti data_summary.json, kopya gruplari '
            'duplicate_groups.json ve etiket bulgulari label_issues.json dosyalarindadir.\n\n'
            'Tam byte ve RGB kopyalari denetlendi; sahne/video bagimsizligi ve yakin kopyalar '
            'bu denetimle kanitlanmaz. Tum turetilmis crop PNG dosyalari tekrar okunmadi. '
            'Devam onbellegi kod, Pillow surumu ve dosya stat bilgilerine baglidir.\n\n'
            'Acik maddeler:\n' + ''.join('- ' + x + '\n' for x in result.get('open_items', [])) +
            '\nBir hata veya kopya bulgusu otomatik olarak tum egitimleri tekrarlama talimati degildir.\n')
    (output / '25_OZET.md').write_text(text, encoding='utf-8')


def bundle(output):
    output = Path(output); target = output / REPORT
    completed = (output / 'result.json').is_file() and read_json(output / 'result.json').get('status') in {
        'COMPLETED', 'COMPLETED_WITH_OPEN_ITEMS'}
    names = ['result.json', 'progress.json', 'protocol.json', 'source_bindings.json', 'environment.json',
             '25_OZET.md', 'error.txt', 'console.log']
    if completed:
        names += ['data_summary.json', 'model_summary.json', 'duplicate_groups.json',
                  'label_issues.json', 'image_inventory.json']
    fd, temporary = tempfile.mkstemp(dir=output, suffix='.zip.tmp'); os.close(fd)
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in names:
                p = output / name
                if p.is_file():
                    z.write(p, name)
            if completed:
                for p in sorted((output / 'models').glob('*.json')):
                    z.write(p, 'models/' + p.name)
            z.write(__file__, 'code/provenance_audit.py')
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log
    def write(self, text):
        self.console.write(text); self.log.write(text); self.log.flush()
    def flush(self):
        self.console.flush(); self.log.flush()


def run(root, raw_root, report_only=False):
    root = Path(root).resolve(); raw_root = Path(raw_root).resolve()
    output = root / 'experiments' / OUT
    require(not output.is_symlink(), '25 cikti klasoru symlink olamaz')
    output.mkdir(parents=True, exist_ok=True)
    with lock(output / 'audit.lock'):
        if report_only:
            require((output / 'result.json').is_file(),
                    'Once 25_VERI_MODEL_DENETIMI.cmd calistirilmali; paketlenecek denetim sonucu yok')
            print('GONDERILECEK RAPOR: ' + str(bundle(output)), flush=True)
            return 0
        with (output / 'console.log').open('a', encoding='utf-8') as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                cache = None; code = 0
                print('\n25 VERI/MODEL DENETIMI: ' + now(), flush=True)
                protocol = {'schema': SCHEMA, 'code_sha256': sha256(__file__), 'models': MODELS,
                            'raw_root': str(raw_root), 'training': False, 'inference': False,
                            'decode': 'Pillow.convert(RGB), no EXIF transpose; SHA256 tagged size and pixels'}
                try:
                    prior = output / 'protocol.json'
                    if prior.exists():
                        require(read_json(prior) == protocol, '25 kodu veya raw yolu degismis; onceki denetimle karistirilmadi')
                    # Prior-attempt summaries must never look like fresh evidence.
                    # Recovery DB and the previous ZIP are retained until the new
                    # report is committed; no file in an older experiment is touched.
                    for filename in ['data_summary.json', 'model_summary.json', 'duplicate_groups.json',
                                     'label_issues.json', 'image_inventory.json', 'source_bindings.json',
                                     'error.txt', 'progress.json', '25_OZET.md']:
                        (output / filename).unlink(missing_ok=True)
                    for name in MODELS:
                        (output / 'models' / (name + '.json')).unlink(missing_ok=True)
                    atomic_json(prior, protocol)
                    atomic_json(output / 'environment.json', {'created_utc': now(), 'python': sys.version,
                                'os': platform.platform(), 'packages': {n: version(n) for n in
                                ['torch', 'torchvision', 'ultralytics', 'numpy', 'Pillow', 'opencv-python']}})
                    atomic_json(output / 'result.json', {'status': 'RUNNING', 'training_started': False})
                    cache = Cache(output / 'resume.sqlite3')
                    inputs = protocol_inputs(root, cache)
                    atomic_json(output / 'source_bindings.json', inputs['files'])
                    data = audit_data(root, raw_root, output, inputs, cache)
                    models = run_models(root)
                    # Small immutable protocol/code inputs are freshly rehashed.
                    for name, digest in inputs['files'].items():
                        require(sha256(safe(root, name)) == digest, 'Denetimde kaynak degisti: ' + name)
                    result = final_result(data, models)
                except KeyboardInterrupt:
                    result = {'status': 'INTERRUPTED', 'open_items': ['Ayni 25 CMD ile devam edin'],
                              'training_started': False, 'GPU_inference': False}; code = 130
                except Exception as exc:
                    result = {'status': 'FAILED', 'error': str(exc), 'open_items': [str(exc)],
                              'training_started': False, 'GPU_inference': False}; code = 1
                    (output / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
                    traceback.print_exc()
                finally:
                    if cache:
                        cache.close()
                result.update(created_utc=now(), code_sha256=sha256(__file__))
                atomic_json(output / 'result.json', result); write_readme(output, result)
                print('DENETIM DURUMU: ' + result['status'], flush=True)
                if result.get('open_items'):
                    print('Rapor incelemesi gerekli; otomatik egitim baslatilmadi.', flush=True)
                print('GONDERILECEK RAPOR: ' + str(bundle(output)), flush=True)
                return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--worker-model', choices=sorted(MODELS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    root = args.root.resolve()
    if args.worker_model:
        return worker(root, args.worker_model)
    return run(root, args.raw or root / 'data/raw', args.report_only)


if __name__ == '__main__':
    sys.exit(main())
