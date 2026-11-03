import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from hycado_start import project_backup as b
from hycado_start import frozen_val as v
from hycado_start import dev_metrics as dm
from hycado_start import score_ablation as s


DECISION = {'proposal_generation': {'threshold': .05, 'top_k': 60, 'context_scale': 3.},
            'merge': {'requested_crop_count': 10, 'requested_crop_count_is_hard_cap': False,
                      'policy': {'name': 'near_z3', 'max_gap_ratio': .5, 'max_axis_growth': 3., 'max_pair_hull_ratio': 2.}}}


class BackupTests(unittest.TestCase):
    def test_full_snapshot_hashes_source_preserved_and_env_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'project'; root.mkdir()
            for relative, data in [('hycado_start/a.py', b'code'), ('experiments/a/weights.pt', b'weights'),
                                   ('experiments/raw_crop_capture_v1_seed42/a.npz', b'raw'), ('.venv/env', b'omit')]:
                p = root / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
            original = {p: b.sha256(p) for p in root.rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()): receipt = b.snapshot(root, Path(tmp) / 'backups')
            self.assertEqual(receipt['status'], 'COMPLETED'); self.assertEqual(receipt['file_count'], 3)
            self.assertEqual(original, {p: b.sha256(p) for p in original})
            copy = Path(receipt['backup_directory']) / root.name
            self.assertFalse((copy / '.venv').exists())
            b.write_json(root / 'reports/12_yedek_sonuc.json', receipt)
            b.verify_receipt(root, ['hycado_start/a.py', 'experiments/a/weights.pt'])
            (copy / 'hycado_start/a.py').write_bytes(b'corrupted')
            with self.assertRaises(ValueError): b.verify_receipt(root, ['hycado_start/a.py'])

    def test_source_change_and_missing_backup_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'project'; root.mkdir(); (root / 'a').write_bytes(b'a')
            with self.assertRaises(FileNotFoundError): b.verify_receipt(root, ['a'])
            with contextlib.redirect_stdout(io.StringIO()): receipt = b.snapshot(root, Path(tmp) / 'backups')
            b.write_json(root / 'reports/12_yedek_sonuc.json', receipt); (root / 'a').write_bytes(b'new')
            with self.assertRaises(ValueError): b.verify_receipt(root, ['a'])

    def test_distinct_snapshots_no_overwrite_and_no_nested_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'project'; root.mkdir(); (root / 'a').write_bytes(b'a')
            with contextlib.redirect_stdout(io.StringIO()):
                one = b.snapshot(root, Path(tmp) / 'backups'); two = b.snapshot(root, Path(tmp) / 'backups')
            self.assertNotEqual(one['backup_directory'], two['backup_directory'])
            with self.assertRaises(ValueError): b.snapshot(root, root / 'bad')

    def test_interruption_never_yields_completed_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'project'; root.mkdir(); (root / 'a').write_bytes(b'a')
            with patch.object(b, 'copy_checked', side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt): b.snapshot(root, Path(tmp) / 'backups')
            dirs = list((Path(tmp) / 'backups').iterdir())
            self.assertEqual(len(dirs), 1); self.assertIn('TAMAMLANMADI', dirs[0].name)
            self.assertFalse((dirs[0] / 'backup_manifest.json').exists())

    def test_no_hardlink_backup_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'a'; dst = Path(tmp) / 'b'; src.write_bytes(b'abc')
            b.copy_checked(src, dst); self.assertFalse(src.samefile(dst)); src.write_bytes(b'changed')
            self.assertEqual(dst.read_bytes(), b'abc')
            root = Path(tmp) / 'root'; root.mkdir(); (root / 'link').symlink_to(src)
            with self.assertRaises(ValueError): b.inventory(root)

    def test_source_tree_change_during_copy_does_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'project'; root.mkdir(); (root / 'a').write_bytes(b'a')
            copy = b.copy_checked
            def changed(source, destination):
                digest = copy(source, destination); (root / 'new_file').write_bytes(b'new')
                return digest
            with patch.object(b, 'copy_checked', side_effect=changed), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError): b.snapshot(root, Path(tmp) / 'backups')
            self.assertTrue(all('TAMAMLANMADI' in p.name for p in (Path(tmp) / 'backups').iterdir()))


class ValTests(unittest.TestCase):
    def fixture(self, root):
        prepared = root / 'prepared'; output = root / 'out'; output.mkdir()
        records = []
        for i in range(3):
            path = prepared / 'images/val' / f'image{i}.png'; path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((48, 64, 3), i * 20, dtype=np.uint8)).save(path)
            records.append({'image': f'images/val/image{i}.png', 'width': 64, 'height': 48,
                            'image_sha256': v.sha256(path), 'annotation_sha256': 'not-read-yet'})
        return prepared, output, records

    def base_factory(self, calls, interrupt=None):
        def predict(path):
            calls.append(path.stem)
            if len(calls) == interrupt: raise KeyboardInterrupt()
            rows = [] if path.stem == 'image2' else [[5., 5., 15., 15., .8, 0]]
            return 64, 48, rows, 0
        return lambda: predict

    def query(self, path, crop):
        a = np.zeros((300, 14), dtype=np.float32); a[0, :4] = [.5, .5, .5, .5]; a[0, 4] = .9
        return a

    def test_manifest_identity_count_disjointness_and_no_gt_fields(self):
        train = {'split': 'train', 'image': 'images/train/t.png', 'image_sha256': 'train'}
        val = {'split': 'val', 'image': 'images/val/v.png', 'image_sha256': 'val', 'width': 4, 'height': 4,
               'annotation_sha256': 'gt', 'ignore_boxes_xyxy': ['not passed'], 'gt_count': 999}
        source = {'purpose': 'fresh_original_visdrone_conversion', 'records': [train, val]}
        got = v.val_records(source, {'records': [train]}, expected=(1, 1))
        self.assertNotIn('gt_count', got[0]); self.assertNotIn('ignore_boxes_xyxy', got[0])
        with self.assertRaises(ValueError): v.val_records(source, {'records': [train]})
        source['records'][1]['image_sha256'] = 'train'
        with self.assertRaises(ValueError): v.val_records(source, {'records': [train]}, expected=(1, 1))

    def test_stage1_interruption_resume_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared, output, records = self.fixture(Path(tmp)); calls = []
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt): v.baseline_pass(records, prepared, output, 'sig', self.base_factory(calls, 2))
                first_hash = v.sha256(output / 'baseline/image0.json'); calls = []
                index = v.baseline_pass(records, prepared, output, 'sig', self.base_factory(calls))
            self.assertEqual(calls, ['image1', 'image2']); self.assertEqual(len(index), 3)
            self.assertEqual(first_hash, v.sha256(output / 'baseline/image0.json'))
            (prepared / records[0]['image']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError): v.baseline_pass(records, prepared, output, 'sig', self.base_factory([]))

    def test_crop_resume_zero_proposals_no_GT_and_exact_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared, output, records = self.fixture(Path(tmp))
            with contextlib.redirect_stdout(io.StringIO()):
                base_index = v.baseline_pass(records, prepared, output, 'sig', self.base_factory([]))
            calls = []
            def interrupt(path, crop):
                calls.append(1)
                if len(calls) == 2: raise KeyboardInterrupt()
                return self.query(path, crop)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt): v.crop_pass(records, prepared, output, 'sig', base_index, DECISION, lambda: interrupt)
            first = v.sha256(output / 'sources/image0/prediction.json'); calls = []
            def query(path, crop): calls.append(1); return self.query(path, crop)
            with contextlib.redirect_stdout(io.StringIO()): index = v.crop_pass(records, prepared, output, 'sig', base_index, DECISION, lambda: query)
            self.assertEqual(len(calls), 1); self.assertEqual(first, v.sha256(output / 'sources/image0/prediction.json'))
            self.assertEqual([r['crops'] for r in index], [1, 1, 0])
            self.assertFalse((prepared / 'visdrone_annotations').exists())
            p = v.read_json(output / 'sources/image0/prediction.json')
            self.assertEqual(p['baseline'], [[5., 5., 15., 15., .8, 0]])
            self.assertAlmostEqual(p['hybrid_a025'][1][4], float(np.float32(.9)) * .25)
            with np.load(output / 'sources/image2/raw_crop_queries.npz', allow_pickle=False) as z:
                self.assertEqual(z['raw_queries'].shape, (0, 300, 14))
            self.assertFalse(list((output / 'sources').glob('*/crop_*.png')))

    def test_measure_matches_frozen_original_evaluator_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared, output, records = self.fixture(Path(tmp))
            with contextlib.redirect_stdout(io.StringIO()):
                base = v.baseline_pass(records, prepared, output, 'sig', self.base_factory([]))
                index = v.crop_pass(records, prepared, output, 'sig', base, DECISION, lambda: self.query)
            for r in records:
                p = prepared / 'visdrone_annotations/val' / (Path(r['image']).stem + '.txt'); p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('5,5,10,10,1,1,0,0\n', encoding='utf-8'); r['annotation_sha256'] = v.sha256(p)
            with self.assertRaises(ValueError): v.measure(records, prepared, output, 'sig', index[:-1])
            with contextlib.redirect_stdout(io.StringIO()): report = v.measure(records, prepared, output, 'sig', index)
            images = []
            for r, item in zip(records, index):
                p = v.read_json(output / item['path']); raw = dm.raw_annotation('5,5,10,10,1,1,0,0\n')
                images.append({'gt': dm.prepare_groundtruth(raw, None), **{n: dm.xyxy_to_det(p[n]) for n in ['baseline', 'hybrid_a025']}})
            for name in ['baseline', 'hybrid_a025']:
                with contextlib.redirect_stdout(io.StringIO()): expected = dm.evaluate(images, name)
                s.compare_metrics(report['variants'][name], expected)
                self.assertEqual(len(list((output / 'detections_txt' / name).glob('*.txt'))), 3)

    def test_report_excludes_weights_raw_images_and_queries(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); v.write_json(p / 'result.json', {'status': 'ERROR'})
            for n in ['model.pt', 'raw.npz', 'photo.png']: (p / n).write_bytes(b'x')
            with zipfile.ZipFile(v.bundle(p)) as z:
                self.assertEqual(z.namelist(), ['result.json']); self.assertIsNone(z.testzip())


if __name__ == '__main__': unittest.main()
