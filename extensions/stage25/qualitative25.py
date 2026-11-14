"""Read frozen predictions and export an auditable qualitative figure bundle.

No model/checkpoint imports, inference, training, matching, or metric tuning.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import traceback
import unittest
import zipfile
import zlib

HERE = Path(__file__).resolve().parent
RUN = 'qualitative1280_v1_seed42'
METHODS = ('yolo1280', 'hybrid1280', 'sahi640')
TITLES = ('Ground truth', 'YOLO-1280', 'HyCADO-1280', 'SAHI-640')
NAMES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
         'tricycle', 'awning-tricycle', 'bus', 'motor')
COLORS = ('#D62728', '#E377C2', '#9467BD', '#1F77B4', '#17BECF',
          '#FF7F0E', '#BCBD22', '#8C564B', '#2CA02C', '#F2B701')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False).encode())


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def safe_name(stem):
    need(isinstance(stem, str) and stem and
         all(c.isascii() and (c.isalnum() or c in '_-') for c in stem),
         'Unsafe image stem')
    return stem


def selection(records, groups, spec):
    """Outcome-blind global hash ranking; retain first image of each group."""
    need(len(records) == len(groups['image_to_group']), 'Group coverage differs')
    need(len({r['image'] for r in records}) == len(records), 'Duplicate image')
    ranked = []
    for rec, gid in zip(records, groups['image_to_group']):
        label = groups['labels'][gid]
        key = f"{spec['seed']}|{label}|{rec['image']}"
        ranked.append((digest(key.encode('utf-8')), label, rec))
    answer, seen = [], set()
    for hashed, group, rec in sorted(ranked, key=lambda x: (x[0], x[2]['image'])):
        if group in seen:
            continue
        seen.add(group)
        answer.append({'rank': len(answer) + 1, 'hash': hashed, 'group': group,
                       'role': 'main' if len(answer) < spec['main_random_examples']
                       else 'supplementary', 'record': rec})
        if len(answer) == spec['candidate_pool']:
            break
    need(len(answer) == spec['candidate_pool'], 'Not enough distinct groups')
    return answer


def dense_window(gt, width, height, size=(640, 420)):
    """Native coordinates. Use GT only, never predictions, to choose a window."""
    ww, hh = min(size[0], width), min(size[1], height)
    centers = [(r[0] + r[2] / 2, r[1] + r[3] / 2) for r in gt]
    if not centers:
        x, y = (width - ww) // 2, (height - hh) // 2
        return [x, y, x + ww, y + hh], 0
    count = min(100, len(centers))
    indices = [0] if count == 1 else [i * (len(centers) - 1) // (count - 1)
                                     for i in range(count)]
    candidates = set()
    for index in indices:
        cx, cy = centers[index]
        x = max(0, min(width - ww, math.floor(cx - ww / 2)))
        y = max(0, min(height - hh, math.floor(cy - hh / 2)))
        candidates.add((x, y))
    scored = []
    for x, y in candidates:
        n = sum(x <= cx < x + ww and y <= cy < y + hh for cx, cy in centers)
        scored.append((-n, x, y))
    neg, x, y = min(scored)
    return [x, y, x + ww, y + hh], -neg


def readonly(path):
    path = Path(path).resolve()
    need(path.is_file(), 'Missing source database: ' + str(path))
    # Immutable read-only opens do not create journal or shared-memory files.
    for suffix in ('-wal', '-journal'):
        other = Path(str(path) + suffix)
        need(not other.exists() or other.stat().st_size == 0,
             'Source may have an active writer; finish it before exporting: ' + str(other))
    db = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True, timeout=30)
    try:
        db.execute('PRAGMA query_only=ON')
    except BaseException:
        db.close()
        raise
    return db


def stat(path):
    s = path.stat()
    return [s.st_size, s.st_mtime_ns]


def load_payload(db, rec, expected_digest, method=None):
    if method is None:
        row = db.execute('SELECT digest,payload FROM prediction WHERE stem=?',
                         (rec['stem'],)).fetchone()
    else:
        row = db.execute('SELECT digest,payload FROM prediction WHERE method=? AND stem=?',
                         (method, rec['stem'])).fetchone()
    need(row is not None, 'Frozen prediction missing: ' + str(method) + '/' + rec['stem'])
    plain = zlib.decompress(row[1])
    need(digest(plain) == row[0] == expected_digest,
         'Prediction hash differs: ' + str(method) + '/' + rec['stem'])
    body = json.loads(plain)
    need(body.get('image_sha256') == rec['image_sha256'], 'Prediction image identity differs')
    rows = body.get('detections')
    need(isinstance(rows, list), 'Expected a final detection list')
    # No global candidate cap: the frozen SAHI archive may exceed 1000 rows.
    for r in rows:
        need(isinstance(r, list) and len(r) == 6 and
             all(type(x) in (int, float) and math.isfinite(x) for x in r),
             'Invalid prediction numbers')
        x1, y1, x2, y2, score, cls = r
        need(0 <= x1 < x2 <= rec['width'] and 0 <= y1 < y2 <= rec['height']
             and 0 <= score <= 1 and cls == int(cls) and 0 <= cls < 10,
             'Invalid prediction geometry/class')
    return plain, body


def illustration_data(annotation_bytes, bodies, rec, spec):
    # Same pinned ignore implementation as stages22-24. All coordinates native.
    import numpy as np
    import visdrone_metrics as v
    raw = v.raw_annotation(annotation_bytes.decode('utf-8-sig'))
    need(not np.any(raw[:, 2:4] <= 0), 'Nonpositive annotation dimensions')
    need(not np.any((raw[:, 5] >= 1) & (raw[:, 5] <= 10) & (raw[:, 4] == 0)),
         'Unexpected class-specific ignored GT')
    integral = v.ignore_integral(raw, rec['width'], rec['height'])
    prepared = v.prepare_groundtruth(raw, integral)
    normal = prepared[(prepared[:, 4] == 0) & (prepared[:, 5] >= 1) & (prepared[:, 5] <= 10)]
    gt = [[float(r[0]), float(r[1]), float(r[0] + r[2]), float(r[1] + r[3]),
           1., int(r[5] - 1)] for r in normal]
    # Window uses original annotation order after the pinned normal-GT filter.
    window, target_count = dense_window(normal.tolist(), rec['width'], rec['height'],
                                        tuple(spec['window']['size']))
    answer = {'ground_truth': gt}
    counts = {}
    for method in METHODS:
        rows = bodies[method]['detections']
        array = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
        wh = v.xyxy_to_det(rows)
        # Include an original row index to retain exact archived xyxy values.
        indexed = np.column_stack((wh, np.arange(len(rows))))
        outside = v.outside_ignore(indexed, integral)
        kept = [int(r[-1]) for r in outside if r[4] >= spec['display_score_threshold']]
        answer[method] = [rows[i] for i in kept]
        counts[method] = {'archived': len(array), 'after_ignore': len(outside),
                          'displayed_full_image': len(kept)}
    return {'window_xyxy': window, 'normal_GT_centers_in_window': target_count,
            'detections_xyxy_score_class0': answer, 'counts': counts,
            'raw_annotation_rows': len(raw), 'normal_GT_after_ignore': len(normal)}


def font(size):
    from PIL import ImageFont
    for candidate in ('DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf'):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def visible_box(row, window):
    x, y, right, bottom = window
    a, b, c, d = max(x, row[0]), max(y, row[1]), min(right, row[2]), min(bottom, row[3])
    if a >= c or b >= d:
        return None
    return [a - x, b - y, c - x, d - y]


def draw_crop(image, rows, window):
    from PIL import ImageDraw
    panel = image.crop(tuple(window)).convert('RGB')
    drawer = ImageDraw.Draw(panel)
    w, h = panel.size
    for row in rows:
        box = visible_box(row, window)
        if box is None:
            continue
        a, b, c, d = box
        # Clamp only the painted border to the last pixel, not source boxes.
        drawer.rectangle((min(w - 1, a), min(h - 1, b),
                          min(w - 1, c), min(h - 1, d)),
                         outline=COLORS[int(row[5])], width=2)
    return panel


def render_row(image, example):
    from PIL import Image, ImageDraw
    data = example['illustration']
    board = Image.new('RGB', (4 * 640 + 3 * 12, 420), 'white')
    for j, method in enumerate(('ground_truth',) + METHODS):
        panel = draw_crop(image, data['detections_xyxy_score_class0'][method], data['window_xyxy'])
        board.paste(panel, (j * 652, 0))
    return board


def render_main(rows, destination):
    from PIL import Image, ImageDraw
    # Headers and row labels occupy separate bands, so labels never cover images.
    width, row_height, header, band, footer = 4 * 640 + 3 * 12, 420, 52, 35, 78
    board = Image.new('RGB', (width, header + len(rows) * (row_height + band) + footer), 'white')
    d = ImageDraw.Draw(board)
    for j, title in enumerate(TITLES):
        d.text((j * 652 + 8, 9), title, fill='#17202A', font=font(29))
    for i, (row, label) in enumerate(rows):
        y = header + i * (row_height + band)
        d.text((7, y + 5), label, fill='#303943', font=font(20))
        board.paste(row, (0, y + band))
    y0 = board.height - footer + 9
    for j, name in enumerate(NAMES):
        x = (j % 5) * 515 + 7
        y = y0 + (j // 5) * 32
        d.rectangle((x, y + 5, x + 20, y + 20), fill=COLORS[j])
        d.text((x + 30, y), f'{j + 1}: {name}', fill='#303943', font=font(21))
    board.save(destination, dpi=(300, 300))


def verify_package(base=HERE):
    files = read(base / 'package_sha256.json')
    need(files and 'qualitative25.py' in files and 'selection.json' in files, 'Package manifest incomplete')
    for relative, expected in files.items():
        path = base / relative
        need(path.is_file() and digest(path.read_bytes()) == expected,
             'Package file changed or missing: ' + relative)
    return fingerprint(files)


def checked_references(base=HERE):
    refs = base / 'references'
    manifest = read(refs / 'dataset_manifest.json')
    groups = read(refs / 'bootstrap_groups.json')
    spec = read(base / 'protocol.json')
    records = manifest['records']
    need(len(records) == 1610, 'Expected 1610 test-dev records')
    need(selection(records, groups, spec['qualitative']) == read(base / 'selection.json'),
         'Frozen image selection differs')
    index20 = read(refs / '20_prediction_index.json')
    result24 = read(refs / '24_result.json')
    index24 = read(refs / '24_prediction_index.json')
    need(result24['status'] == 'SAHI640_TESTDEV_COMPLETE', 'Stage24 not complete')
    need(fingerprint(index24) == result24['prediction_freeze_sha256'], 'Stage24 index hash differs')
    need(index24['protocol_sha256'] == result24['signature'], 'Stage24 binding differs')
    stems = {r['stem'] for r in records}
    need({Path(x['image']).stem for x in index20['records']} == stems and
         {x['stem'] for x in index24['records']} == stems, 'Prediction index coverage differs')
    return manifest, spec, index20, result24, index24


def export(project, dataset, destination, package_hash, tests_passed=0):
    from PIL import Image, ImageDraw
    manifest, spec, idx20, result24, idx24 = checked_references()
    selected = read(HERE / 'selection.json')
    ix20 = {Path(x['image']).stem: x for x in idx20['records']}
    ix24 = {x['stem']: x for x in idx24['records']}
    annotation_hashes = {x['stem']: x['sha256'] for x in
                         read(HERE / 'references/annotation_index.json')['records']}
    paths = {'source20': project / 'experiments/testdev1280_extension_v1_seed42/predictions.sqlite3',
             'source24': project / 'experiments/sahi640_testdev_v1_seed42/predictions.sqlite3'}
    need(all(p.is_file() for p in paths.values()), 'Source prediction database missing; keep stages20 and24')
    before = {name: stat(path) for name, path in paths.items()}
    destination.mkdir(parents=True, exist_ok=True)
    signature = fingerprint({'package': package_hash, 'selection': selected,
                             'stage20': idx20['protocol_sha256'], 'stage24': result24['signature']})
    binding = destination / 'binding.json'
    if binding.exists():
        need(read(binding)['signature'] == signature, 'Output belongs to another package; keep it and report this error')
    else:
        write(binding, {'signature': signature})
    # A successful old export remains untouched until a complete replacement is ready.
    with tempfile.TemporaryDirectory(prefix='export25_', dir=destination) as temporary:
        temp = Path(temporary)
        for folder in ('images', 'annotations', 'predictions', 'previews', 'context', 'references'):
            (temp / folder).mkdir()
        summary, main_rows = [], []
        with contextlib.ExitStack() as stack:
            db20 = stack.enter_context(contextlib.closing(readonly(paths['source20'])))
            db24 = stack.enter_context(contextlib.closing(readonly(paths['source24'])))
            need(db20.execute('SELECT signature FROM binding WHERE id=1').fetchall() ==
                 [(idx20['protocol_sha256'],)], 'Stage20 SQLite binding differs')
            need(db24.execute('SELECT signature FROM binding WHERE id=1').fetchall() ==
                 [(result24['signature'],)], 'Stage24 SQLite binding differs')
            need(db24.execute('SELECT COUNT(*) FROM prediction').fetchone()[0] == 1610,
                 'Stage24 predictions are incomplete')
            for item in selected:
                rec, rank = item['record'], item['rank']
                stem = safe_name(rec['stem'])
                photo = dataset / 'images' / (stem + '.jpg')
                annotation = dataset / 'annotations' / rec['annotation']
                raw_image, raw_annotation = photo.read_bytes(), annotation.read_bytes()
                need(digest(raw_image) == rec['image_sha256'], 'Image hash differs: ' + stem)
                need(digest(raw_annotation) == annotation_hashes[stem], 'Annotation hash differs: ' + stem)
                with Image.open(io.BytesIO(raw_image)) as source:
                    need(source.format == 'JPEG' and source.size == (rec['width'], rec['height']),
                         'Image dimensions or format differ: ' + stem)
                    image = source.convert('RGB')
                (temp / 'images' / photo.name).write_bytes(raw_image)
                (temp / 'annotations' / annotation.name).write_bytes(raw_annotation)
                bodies, payload_hashes = {}, {}
                for method in METHODS:
                    expected = ix24[stem]['sha256'] if method == 'sahi640' else ix20[stem]['variant_sha256'][method]
                    plain, body = load_payload(db24 if method == 'sahi640' else db20,
                                               rec, expected, None if method == 'sahi640' else method)
                    if method == 'sahi640':
                        need(body.get('protocol_sha256') == result24['signature'] and
                             body.get('image') == rec['image'] and
                             (body.get('width'), body.get('height')) == (rec['width'], rec['height']) and
                             len(body['detections']) == ix24[stem]['detections'],
                             'Stage24 image binding/count differs')
                    bodies[method], payload_hashes[method] = body, digest(plain)
                    # These bytes are exactly the decompressed frozen payload, not a rewrite.
                    (temp / 'predictions' / (stem + '__' + method + '.json')).write_bytes(plain)
                illustration = illustration_data(raw_annotation, bodies, rec, spec['qualitative'])
                example = dict(item, illustration=illustration, prediction_sha256=payload_hashes,
                               annotation_sha256=annotation_hashes[stem])
                summary.append(example)
                row = render_row(image, example)
                render_main([(row, f"{rank:02d}: {stem}")], temp / 'previews' / f'{rank:02d}_{stem}.png')
                context = image.copy()
                draw = ImageDraw.Draw(context)
                draw.rectangle(tuple(illustration['window_xyxy']), outline='#FFD400', width=5)
                context.thumbnail((960, 640))
                context.save(temp / 'context' / f'{rank:02d}_{stem}.jpg', quality=92)
                if item['role'] == 'main':
                    main_rows.append((row, f"({chr(96 + rank)}) {stem}"))
                print(f'25 Nitel gorsel: {rank}/{len(selected)}; kaynak ve tahmin hashleri PASS', flush=True)
        need({name: stat(path) for name, path in paths.items()} == before,
             'Source database changed during export; stop active writers and run again')
        render_main(main_rows, temp / 'qualitative1280_main_preview.png')
        write(temp / 'examples.json', summary)
        shutil.copyfile(HERE / 'protocol.json', temp / 'protocol.json')
        shutil.copyfile(HERE / 'selection.json', temp / 'selection.json')
        shutil.copyfile(HERE / 'CAPTION_EN.md', temp / 'CAPTION_EN.md')
        shutil.copyfile(HERE / 'qualitative25.py', temp / 'qualitative25.py')
        shutil.copyfile(HERE / 'visdrone_metrics.py', temp / 'visdrone_metrics.py')
        for name in ('24_result.json',):
            shutil.copyfile(HERE / 'references' / name, temp / 'references' / name)
        result = {'status': 'QUALITATIVE_EXPORT_COMPLETE', 'images': len(selected),
                  'main_images': len(main_rows), 'methods': list(METHODS),
                  'tests_passed': tests_passed,
                  'signature': signature, 'package_manifest_sha256': package_hash,
                  'training': False, 'gpu_inference': False, 'new_metric_computation': False,
                  'source_files_opened_readonly': True, 'source_database_stat_unchanged': True,
                  'full_database_rehash': False, 'selected_prediction_payload_hashes_verified': True,
                  'original_image_and_annotation_hashes_verified': True,
                  'image_selection_uses_predictions': False, 'window_selection_uses_GT': True,
                  'class_colors_are_not_TP_FP_labels': True,
                  'versions': {k: importlib.metadata.version(k) for k in ('Pillow', 'numpy')},
                  'source_database_stats': before}
        write(temp / 'result.json', result)
        contents = {p.relative_to(temp).as_posix(): digest(p.read_bytes())
                    for p in sorted(temp.rglob('*')) if p.is_file()}
        write(temp / 'contents_sha256.json', contents)
        pending = destination / '25_nitel_gorsel_rapor.zip.partial'
        with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED, compresslevel=5) as z:
            for p in sorted(temp.rglob('*')):
                if p.is_file():
                    z.write(p, p.relative_to(temp).as_posix())
        with zipfile.ZipFile(pending) as z:
            need(z.testzip() is None, 'Output ZIP CRC failed')
        target = destination / '25_nitel_gorsel_rapor.zip'
        os.replace(pending, target)
        write(destination / 'result.json', result)
        shutil.copyfile(temp / 'qualitative1280_main_preview.png',
                        destination / 'qualitative1280_main_preview.png')
        return target, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('D:/HyCADO_Temiz_Baslangic'))
    parser.add_argument('--dataset', type=Path, help='Existing test-dev directory containing images/ and annotations/')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    try:
        package_hash = verify_package()
        suite = unittest.defaultTestLoader.discover(str(HERE), pattern='test_qualitative25.py')
        tests = unittest.TextTestRunner(verbosity=2).run(suite)
        need(tests.wasSuccessful() and tests.testsRun > 0, 'Self-tests failed')
        checked_references()
        if args.self_test:
            return 0
        project = args.project.resolve()
        manifest = read(HERE / 'references/dataset_manifest.json')
        dataset = args.dataset.resolve() if args.dataset else project / 'experiments/testdev_frozen_v1_seed42' / (
            'dataset_' + manifest['official_zip_sha256'][:16])
        # Existing dependencies only; no pip, environment changes, or torch import.
        import PIL
        import numpy
        destination = project / 'experiments' / RUN
        target, result = export(project, dataset, destination, package_hash, tests.testsRun)
        print('SONUC:', result['status'], flush=True)
        print('GONDERILECEK RAPOR:', target, flush=True)
        print('ONIZLEME:', destination / 'qualitative1280_main_preview.png', flush=True)
        return 0
    except Exception:
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        error = HERE / '25_nitel_gorsel_hata.txt'
        error.write_text(detail, encoding='utf-8')
        print('DURDU. Hata kaydi:', error, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
