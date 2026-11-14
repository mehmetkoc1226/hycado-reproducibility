"""Synthetic fixtures below validate software; they are not research results."""
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import zlib
import qualitative25 as q


class QualitativeTests(unittest.TestCase):
    def setUp(self):
        # Linux allows unlinking an open database, which hid the Windows error.
        # Check actual connection lifetimes on every platform, not just deletion.
        self._opened_connections = []
        original_connect = sqlite3.connect

        def tracked_connect(*args, **kwargs):
            db = original_connect(*args, **kwargs)
            self._opened_connections.append(db)
            return db

        self._connect_patch = patch.object(sqlite3, 'connect', side_effect=tracked_connect)
        self._connect_patch.start()

    def tearDown(self):
        self._connect_patch.stop()
        still_open = 0
        for db in self._opened_connections:
            try:
                db.execute('SELECT 1').fetchone()
                still_open += 1
            except sqlite3.ProgrammingError:
                pass  # Expected: the connection has explicitly been closed.
            finally:
                db.close()
        self.assertEqual(still_open, 0, 'SQLite connections left open; with db only commits/rolls back, it does not close')

    def test_selection_distinct_groups_and_deterministic(self):
        records = [{'image': f'images/{i}.jpg'} for i in range(30)]
        groups = {'labels': ['a', 'b', 'c', 'd', 'e'], 'image_to_group': [i % 5 for i in range(30)]}
        spec = {'seed': 99, 'candidate_pool': 4, 'main_random_examples': 3}
        a = q.selection(records, groups, spec)
        b = q.selection(records[::-1], dict(groups, image_to_group=groups['image_to_group'][::-1]), spec)
        self.assertEqual(a, b)
        self.assertEqual(len({x['group'] for x in a}), 4)
        self.assertEqual([x['role'] for x in a], ['main'] * 3 + ['supplementary'])

    def test_actual_frozen_selection_is_reproducible(self):
        refs = q.HERE / 'references'
        got = q.selection(q.read(refs / 'dataset_manifest.json')['records'],
                          q.read(refs / 'bootstrap_groups.json'), q.read(q.HERE / 'protocol.json')['qualitative'])
        self.assertEqual(got, q.read(q.HERE / 'selection.json'))
        self.assertEqual(len(got), 12)
        self.assertEqual(len({x['group'] for x in got}), 12)

    def test_empty_window_and_small_image(self):
        self.assertEqual(q.dense_window([], 1000, 800), ([180, 190, 820, 610], 0))
        self.assertEqual(q.dense_window([], 200, 150), ([0, 0, 200, 150], 0))

    def test_dense_window_max_count_and_tie(self):
        rows = [[10, 10, 5, 5], [11, 11, 5, 5], [900, 600, 5, 5]]
        self.assertEqual(q.dense_window(rows, 1000, 800), ([0, 0, 640, 420], 2))
        # Equal density resolves by x then y, independent of candidate order.
        a = [[900, 600, 5, 5], [10, 10, 5, 5]]
        self.assertEqual(q.dense_window(a, 1000, 800)[0], [0, 0, 640, 420])

    def test_more_than_100_gt_candidate_centers(self):
        rows = [[float(i % 1000), float(i % 700), 2, 2] for i in range(1001)]
        win, count = q.dense_window(rows, 1200, 800)
        actual = sum(win[0] <= r[0] + 1 < win[2] and win[1] <= r[1] + 1 < win[3] for r in rows)
        self.assertEqual(actual, count)
        self.assertEqual((win[2] - win[0], win[3] - win[1]), (640, 420))

    def test_window_visible_box_clipping(self):
        self.assertEqual(q.visible_box([0, 20, 30, 60], [10, 30, 50, 80]), [0, 0, 20, 30])
        self.assertIsNone(q.visible_box([0, 0, 10, 10], [10, 10, 50, 50]))

    def test_missing_readonly_does_not_create_and_writes_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'source.sqlite3'
            with self.assertRaisesRegex(ValueError, 'Missing source'):
                q.readonly(p)
            self.assertFalse(p.exists())
            with contextlib.closing(sqlite3.connect(p)) as db, db:
                db.execute('CREATE TABLE x(a)')
            before = p.read_bytes()
            with contextlib.closing(q.readonly(p)) as db:
                with self.assertRaises(sqlite3.OperationalError):
                    db.execute('INSERT INTO x VALUES(1)')
            self.assertEqual(before, p.read_bytes())
            Path(str(p) + '-wal').write_bytes(b'active')
            with self.assertRaisesRegex(ValueError, 'active writer'):
                q.readonly(p)

    def test_payload_hash_no_cap_and_zero_area_rejected(self):
        rec = {'stem': 'fixture', 'image_sha256': 'imagehash', 'width': 100, 'height': 100}
        body = {'image_sha256': 'imagehash', 'detections': [[1., 2., 3., 4., .05, 0]] * 1143}
        plain = json.dumps(body).encode()
        with contextlib.closing(sqlite3.connect(':memory:')) as db, db:
            db.execute('CREATE TABLE prediction(stem,digest,payload)')
            db.execute('INSERT INTO prediction VALUES(?,?,?)', ('fixture', q.digest(plain), zlib.compress(plain)))
            got, parsed = q.load_payload(db, rec, q.digest(plain))
            self.assertEqual(got, plain)
            self.assertEqual(len(parsed['detections']), 1143)
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                q.load_payload(db, rec, 'tampered')
            body['detections'] = [[1, 0, 2, 0, .5, 0]]
            plain = json.dumps(body).encode()
            db.execute('UPDATE prediction SET digest=?,payload=?', (q.digest(plain), zlib.compress(plain)))
            with self.assertRaisesRegex(ValueError, 'geometry'):
                q.load_payload(db, rec, q.digest(plain))

    def test_display_threshold_preserves_archive_and_ignores_others(self):
        annotation = b'5,5,4,4,1,1,0,0\n20,20,4,4,1,11,0,0\n'
        bodies = {m: {'detections': [[5, 5, 9, 9, .05, 0], [15, 15, 19, 19, .049, 0]]} for m in q.METHODS}
        original = json.dumps(bodies)
        data = q.illustration_data(annotation, bodies, {'width': 30, 'height': 30},
                                   {'display_score_threshold': .05, 'window': {'size': [640, 420]}})
        self.assertEqual(data['normal_GT_after_ignore'], 1)
        self.assertEqual(data['counts']['hybrid1280']['archived'], 2)
        self.assertEqual(data['counts']['hybrid1280']['displayed_full_image'], 1)
        self.assertEqual(data['detections_xyxy_score_class0']['hybrid1280'][0], [5, 5, 9, 9, .05, 0])
        self.assertEqual(json.dumps(bodies), original)

    def test_pinned_ignore_filter_applies_to_gt_and_predictions(self):
        annotation = b'1,1,10,10,0,0,0,0\n2,2,4,4,1,1,0,0\n20,20,4,4,1,4,0,0\n'
        bodies = {m: {'detections': [[2, 2, 6, 6, .9, 0], [20, 20, 24, 24, .2, 3]]} for m in q.METHODS}
        data = q.illustration_data(annotation, bodies, {'width': 30, 'height': 30},
                                   {'display_score_threshold': .05, 'window': {'size': [640, 420]}})
        self.assertEqual(data['normal_GT_after_ignore'], 1)
        self.assertEqual(data['detections_xyxy_score_class0']['ground_truth'][0][-1], 3)
        self.assertEqual(data['counts']['sahi640']['after_ignore'], 1)
        self.assertEqual(len(data['detections_xyxy_score_class0']['sahi640']), 1)

    def test_package_tampering_and_unsafe_names_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            for name in ['qualitative25.py', 'selection.json']:
                (p / name).write_text('original')
            q.write(p / 'package_sha256.json', {name: q.digest(b'original') for name in ['qualitative25.py', 'selection.json']})
            q.verify_package(p)
            (p / 'selection.json').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                q.verify_package(p)
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            q.safe_name('../escape')

    def test_roundtrip_export_verifies_bytes_and_preserves_databases(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            package, root, dataset = p / 'package', p / 'project', p / 'data'
            for folder in (package / 'references', dataset / 'images', dataset / 'annotations'):
                folder.mkdir(parents=True)
            image_buffer = io.BytesIO()
            Image.new('RGB', (40, 30), '#889999').save(image_buffer, format='JPEG')
            img, ann = image_buffer.getvalue(), b'10,10,5,5,1,4,0,0\n'
            (dataset / 'images/fixture.jpg').write_bytes(img)
            (dataset / 'annotations/fixture.txt').write_bytes(ann)
            rec = {'stem': 'fixture', 'image': 'images/test-dev/fixture.jpg', 'image_sha256': q.digest(img),
                   'width': 40, 'height': 30, 'annotation': 'fixture.txt'}
            selected = [{'rank': 1, 'hash': 'synthetic-unit-test', 'group': 'fixture', 'role': 'main', 'record': rec}]
            spec = {'qualitative': {'display_score_threshold': .05, 'window': {'size': [640, 420]}}}
            q.write(package / 'selection.json', selected)
            q.write(package / 'protocol.json', spec)
            q.write(package / 'references/annotation_index.json', {'records': [{'stem': 'fixture', 'sha256': q.digest(ann)}]})
            q.write(package / 'references/24_result.json', {'signature': 'sig24'})
            for name in ('CAPTION_EN.md', 'qualitative25.py', 'visdrone_metrics.py'):
                (package / name).write_text('synthetic fixture; not a research result')
            old = root / 'experiments/testdev1280_extension_v1_seed42/predictions.sqlite3'
            new = root / 'experiments/sahi640_testdev_v1_seed42/predictions.sqlite3'
            for db_path, sig in ((old, 'sig20'), (new, 'sig24')):
                db_path.parent.mkdir(parents=True)
                with contextlib.closing(sqlite3.connect(db_path)) as db, db:
                    db.execute('CREATE TABLE binding(id INTEGER,signature TEXT)')
                    db.execute('INSERT INTO binding VALUES(1,?)', (sig,))
                    db.execute('CREATE TABLE prediction(method TEXT,stem TEXT,digest TEXT,payload BLOB)')
            plain = json.dumps({'image': rec['image'], 'image_sha256': rec['image_sha256'],
                                'width': 40, 'height': 30, 'protocol_sha256': 'sig24',
                                'detections': [[10, 10, 15, 15, .05, 3]]}).encode()
            h, compressed = q.digest(plain), zlib.compress(plain)
            with contextlib.closing(sqlite3.connect(old)) as db, db:
                for m in q.METHODS[:2]:
                    db.execute('INSERT INTO prediction VALUES(?,?,?,?)', (m, 'fixture', h, compressed))
            with contextlib.closing(sqlite3.connect(new)) as db, db:
                db.execute('INSERT INTO prediction VALUES(?,?,?,?)', ('sahi640', 'fixture', h, compressed))
                # Production checks completed coverage; dummy unused rows have no exported payload.
                for i in range(1609):
                    db.execute('INSERT INTO prediction VALUES(?,?,?,?)', ('sahi640', f'unused{i}', h, compressed))
            idx20 = {'protocol_sha256': 'sig20', 'records': [{'image': rec['image'], 'variant_sha256': {m: h for m in q.METHODS[:2]}}]}
            idx24 = {'records': [{'stem': 'fixture', 'sha256': h, 'detections': 1}]}
            before = (old.read_bytes(), new.read_bytes())
            with contextlib.redirect_stdout(io.StringIO()), patch.object(q, 'HERE', package), patch.object(q, 'checked_references', return_value=(
                    {}, spec, idx20, {'signature': 'sig24'}, idx24)):
                target, result = q.export(root, dataset, root / 'output', 'packagehash', 12)
            self.assertEqual(result['tests_passed'], 12)
            self.assertEqual(result['status'], 'QUALITATIVE_EXPORT_COMPLETE')
            self.assertEqual((old.read_bytes(), new.read_bytes()), before)
            previous_archive = target.read_bytes()
            # An exception while both real-style source handles are open must
            # close them before temporary-file cleanup and keep a completed ZIP.
            with contextlib.redirect_stdout(io.StringIO()), patch.object(q, 'HERE', package), patch.object(q, 'checked_references', return_value=(
                    {}, spec, idx20, {'signature': 'sig24'}, idx24)), patch.object(q, 'illustration_data', side_effect=RuntimeError('simulated rendering failure')):
                with self.assertRaisesRegex(RuntimeError, 'simulated rendering failure'):
                    q.export(root, dataset, root / 'output', 'packagehash', 12)
            self.assertEqual(target.read_bytes(), previous_archive)
            self.assertEqual((old.read_bytes(), new.read_bytes()), before)
            with zipfile.ZipFile(target) as z:
                self.assertIsNone(z.testzip())
                self.assertEqual(z.read('images/fixture.jpg'), img)
                self.assertEqual(z.read('predictions/fixture__sahi640.json'), plain)
                hashes = json.loads(z.read('contents_sha256.json'))
                self.assertTrue(all(q.digest(z.read(n)) == v for n, v in hashes.items()))
                with Image.open(io.BytesIO(z.read('qualitative1280_main_preview.png'))) as preview:
                    self.assertEqual(preview.width, 2596)

    def test_readonly_setup_failure_closes_connection(self):
        class BrokenPragma(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):
                if sql == 'PRAGMA query_only=ON':
                    raise RuntimeError('simulated query_only setup failure')
                return super().execute(sql, *args, **kwargs)

        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'source.sqlite3'
            with contextlib.closing(sqlite3.connect(path)) as db, db:
                db.execute('CREATE TABLE x(a)')
            tracked_connect = sqlite3.connect
            with patch.object(sqlite3, 'connect', side_effect=lambda *a, **k: tracked_connect(*a, factory=BrokenPragma, **k)):
                with self.assertRaisesRegex(RuntimeError, 'query_only setup failure'):
                    q.readonly(path)


if __name__ == '__main__':
    unittest.main()
