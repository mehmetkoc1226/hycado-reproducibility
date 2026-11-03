import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from hycado_start import score_ablation as s
from hycado_start import dev_metrics as old


def fixture():
    baseline = [[2., 0., 12., 10., .8, 0.], [40., 40., 50., 50., .6, 1.]]
    candidates = np.array(baseline + [[3.5, 0., 13.5, 10., .9, 0.], [60., 60., 70., 70., .002, 0.]])
    origins = np.array([[0, -1, 0], [0, -1, 1], [1, 0, 1], [1, 0, 2]], dtype=np.int32)
    original = s.classwise_nms(candidates.tolist(), 100, 100)
    raw = old.raw_annotation('0,0,10,10,1,1,0,0\n40,40,10,10,1,2,0,0\n70,70,10,10,0,1,0,0\n')
    return baseline, candidates, origins, original, raw


class ScoreAblationTests(unittest.TestCase):
    def test_matching_equivalence_random_ties_ignore(self):
        rng = np.random.default_rng(42)
        for _ in range(40):
            gt = np.column_stack([rng.integers(0, 7, (22, 2)), rng.integers(1, 15, (22, 2)),
                                  rng.integers(0, 2, 22), rng.integers(1, 4, 22)]).astype(float)
            det = np.column_stack([rng.integers(0, 7, (35, 2)), rng.integers(1, 15, (35, 2)),
                                   np.sort(rng.choice([.2, .5, .9], 35))[::-1], rng.integers(1, 4, 35)]).astype(float)
            for threshold in s.THRESHOLDS:
                actual = s.fast_flags(det, gt, float(threshold))
                for cls in range(1, 4):
                    ids = np.flatnonzero(det[:, 5] == cls)
                    np.testing.assert_array_equal(actual[ids], old.match_flags(det[ids, :5], gt[gt[:, 5] == cls, :5], float(threshold)))

    def test_full_metrics_equivalent_original_evaluator(self):
        base, a, o, original, raw = fixture()
        parts = [s.analyze_parent(raw, 100, 100, a, o, base, original)] * 2
        integral = old.ignore_integral(raw, 100, 100)
        gt = old.prepare_groundtruth(raw, integral)
        variants = s.fusion_variants(a, o, 100, 100, base, original)
        for name in s.VARIANTS:
            images = [{'gt': gt, name: old.outside_ignore(old.xyxy_to_det(variants[name]), integral)}] * 2
            with contextlib.redirect_stdout(io.StringIO()): expected = old.evaluate(images, name)
            s.compare_metrics(s.aggregate(parts, name), expected)

    def test_dense_global_caps_and_unequal_class_presence(self):
        rng = np.random.default_rng(123); parts = []; images = []
        for number in range(2):
            gt = np.column_stack([rng.integers(0, 30, (18, 2)), rng.integers(3, 15, (18, 2)),
                                  rng.integers(0, 2, 18), rng.integers(1, 3, 18)]).astype(float)
            if number: gt[:, 5] = 1
            det = np.column_stack([rng.integers(0, 30, (520, 2)), rng.integers(3, 15, (520, 2)),
                                   np.sort(rng.choice([.1, .2, .5, .9], 520))[::-1], rng.integers(1, 3, 520)]).astype(float)
            images.append({'gt': gt, 'baseline': det})
            parts.append({'gt_denominators': np.array([np.count_nonzero(gt[:, 5] == c) for c in range(1, 11)]),
                          'presence': np.array([np.any(gt[:, 5] == c) for c in range(1, 11)], dtype=np.int64),
                          'baseline_scores': det[:500, 4], 'baseline_classes': det[:500, 5].astype(np.int8),
                          'baseline_flags': np.column_stack([s.fast_flags(det[:500], gt, float(t)) for t in s.THRESHOLDS])})
        with contextlib.redirect_stdout(io.StringIO()): expected = old.evaluate(images, 'baseline')
        s.compare_metrics(s.aggregate(parts, 'baseline'), expected)

    def test_multiplication_changes_nms_before_matching_and_preserves_sources(self):
        base, a, o, original, _ = fixture(); before = a.copy()
        result = s.fusion_variants(a, o, 100, 100, base, original)
        self.assertEqual(result['hybrid_a100'], original)
        self.assertEqual(result['hybrid_a050'][0], base[0])
        self.assertNotIn(base[0], result['hybrid_a100'])
        self.assertTrue(any(abs(row[4] - .0002) < 1e-15 for row in result['hybrid_a010']))
        np.testing.assert_array_equal(a, before)
        self.assertEqual(result['baseline'], base)

    def test_global_cap_recomputed_after_rescoring(self):
        baseline = [[0., 0., 1., 1., .5, 0.]]
        crops = [[float(i + 2), 0., float(i + 3), 1., .9, 0.] for i in range(1001)]
        a = np.array(baseline + crops); o = np.array([[0, -1, 0]] + [[1, i, 0] for i in range(1001)])
        original = s.classwise_nms(a.tolist(), 2000, 10)
        result = s.fusion_variants(a, o, 2000, 10, baseline, original)
        self.assertEqual(len(result['hybrid_a100']), 1000)
        self.assertNotIn(baseline[0], result['hybrid_a100'])
        self.assertEqual(result['hybrid_a050'][0], baseline[0])
        self.assertEqual(len(result['hybrid_a050']), 1000)

    def test_replay_mismatch_rejected(self):
        base, a, o, original, _ = fixture()
        with self.assertRaises(ValueError): s.fusion_variants(a, o, 100, 100, base, original[:-1])

    def test_cache_roundtrip_identity_corruption_and_no_overwrite(self):
        base, a, o, original, raw = fixture(); data = s.analyze_parent(raw, 100, 100, a, o, base, original)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'item'; bindings = {'source': 'a'}
            s.save_cache(p, data, bindings); loaded = s.load_cache(p, bindings)
            for key in data: np.testing.assert_array_equal(data[key], loaded[key])
            with self.assertRaises(FileExistsError): s.save_cache(p, data, bindings)
            with self.assertRaises(ValueError): s.load_cache(p, {'source': 'b'})
            with (p / 'matches.npz').open('ab') as f: f.write(b'corrupt')
            with self.assertRaises(ValueError): s.load_cache(p, bindings)

    def test_interruption_resume_and_source_immutability(self):
        base, a, o, original, raw = fixture()
        gt_text = '0,0,10,10,1,1,0,0\n40,40,10,10,1,2,0,0\n70,70,10,10,0,1,0,0\n'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'experiments/development_eval_v1_seed42'
            capture = root / 'experiments/raw_crop_capture_v1_seed42'; output = root / 'out'
            old_index = {'records': []}; index = {'records': []}; hashes = {}
            for number, nc in [(1, 1), (2, 2)]:
                stem = 'image' + str(number); image = 'images/train/' + stem + '.jpg'
                old_path = source / 'predictions' / (stem + '.json'); old_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {'protocol_sha256': s.SOURCE_PROTOCOL, 'source_image': image, 'width': 100, 'height': 100,
                           'baseline': base, 'hycado': original, 'all_crops_processed': nc,
                           'source_crop_document_sha256': 'cropdoc' + str(number)}
                payload['content_sha256'] = s.fingerprint(payload); s.write_json(old_path, payload)
                gt_path = source / 'reference_annotations' / (stem + '.txt'); gt_path.parent.mkdir(exist_ok=True)
                gt_path.write_text(gt_text, encoding='utf-8')
                folder = capture / 'sources' / stem; folder.mkdir(parents=True)
                np.savez_compressed(folder / 'queries_and_trace.npz', candidate_rows=a, candidate_origins=o)
                meta = {'protocol_sha256': s.CAPTURE_SIGNATURE, 'source_image': image, 'crop_count': nc,
                        'source_crop_document_sha256': payload['source_crop_document_sha256'],
                        'old11_prediction_sha256': s.sha256(old_path), 'archive_sha256': s.sha256(folder / 'queries_and_trace.npz'),
                        'diagnostics': {'original11_exactly_reproduced': True}}
                meta['content_sha256'] = s.fingerprint(meta); s.write_json(folder / 'capture.json', meta)
                old_index['records'].append({'source_image': image, 'path': old_path.relative_to(source).as_posix(),
                    'sha256': s.sha256(old_path), 'annotation_sha256': s.sha256(gt_path), 'width': 100, 'height': 100})
                index['records'].append({'source_image': image, 'path': (folder / 'capture.json').relative_to(capture).as_posix(),
                                         'sha256': s.sha256(folder / 'capture.json'), 'archive_sha256': meta['archive_sha256']})
            data = s.analyze_parent(raw, 100, 100, a, o, base, original)
            s.write_json(source / 'metrics.json', {'baseline': s.aggregate([data, data], 'baseline'),
                                                  'hycado': s.aggregate([data, data], 'hybrid_a100')})
            counts = {t: {old_name: s.aggregate([data, data], name)['counts_at_fixed_max500'][t]
                          for name, old_name in [('baseline', 'baseline'), ('crop_only', 'crop_only_fixed_NMS'), ('hybrid_a100', 'hybrid_original_11')]}
                      for t in ['0.5', '0.75']}
            s.write_json(capture / 'capture_summary.json', {'counts_at_fixed_max500': counts})
            output.mkdir()
            hashes = {p: s.sha256(p) for p in root.rglob('*') if p.is_file()}
            original_analyze = s.analyze_parent; calls = []
            def interrupt(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2: raise KeyboardInterrupt()
                return original_analyze(*args, **kwargs)
            with patch.object(s, 'analyze_parent', side_effect=interrupt), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt): s.process(root, output, old_index, index, 'signature', expected_counts=(2, 3))
            committed = s.sha256(output / 'sources/image1/matches.npz')
            with patch.object(s, 'analyze_parent', wraps=original_analyze) as resumed, contextlib.redirect_stdout(io.StringIO()):
                report = s.process(root, output, old_index, index, 'signature', expected_counts=(2, 3))
                self.assertEqual(resumed.call_count, 1)
            self.assertEqual(report['images'], 2); self.assertEqual(report['crops'], 3)
            self.assertEqual(committed, s.sha256(output / 'sources/image1/matches.npz'))
            self.assertEqual(hashes, {p: s.sha256(p) for p in hashes})

    def test_report_contains_no_npz_or_model(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); s.write_json(p / 'result.json', {'status': 'ERROR'})
            (p / 'secret.npz').write_bytes(b'raw'); (p / 'weights.pt').write_bytes(b'model')
            with zipfile.ZipFile(s.bundle(p)) as z:
                self.assertEqual(z.namelist(), ['result.json']); self.assertIsNone(z.testzip())


if __name__ == '__main__': unittest.main()
