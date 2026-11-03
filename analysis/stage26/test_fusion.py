import importlib.util
from pathlib import Path
import unittest, tempfile, numpy as np, contextlib, io

spec = importlib.util.spec_from_file_location('fusion26', Path(__file__).with_name('full_image_fusion.py'))
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


class FusionTests(unittest.TestCase):
    def test_score_scale_no_cutoff_and_original_preserved(self):
        base = [[0., 0., 10., 10., .2, 0]]
        full = [[0., 0., 10., 10., .4, 0], [12., 12., 14., 14., .0011, 1]]
        scaled = f.fuse(base, full, 20, 20, .25)
        self.assertEqual(scaled[0], base[0]); self.assertIn(.000275, [r[4] for r in scaled])
        self.assertEqual(full[0][4], .4)
        self.assertEqual(f.fuse(base, full, 20, 20, 1.)[0][4], .4)

    def test_empty_and_classwise(self):
        self.assertEqual(f.fuse([], [], 20, 20, .25), [])
        a = [[0., 0., 10., 10., .2, 0]]; b = [[0., 0., 10., 10., .8, 1]]
        self.assertEqual(len(f.fuse(a, b, 20, 20, 1.)), 2)

    def test_flags_and_aggregation_equal_literal_evaluate(self):
        rng = np.random.default_rng(26); images = []; parts = []
        with tempfile.TemporaryDirectory() as d:
            for i in range(5):
                gt = []; dt = []
                for c in range(1, 11):
                    # Presence varies across images, testing class multiplicity.
                    if i % c == 0:
                        gt += [[c*20., 0., 10., 10., 0., c], [c*20., 20., 10., 10., 1., c]]
                    dt += [[c*20., 0., 10., 10., .5 if i%2 else .7, c],
                           [c*20., 20., 10., 10., .001, c], [c*20., 20., 10., 10., .001, c]]
                gt = np.asarray(gt).reshape(-1, 6); dt = np.asarray(dt).reshape(-1, 6)
                dt = dt[np.argsort(-dt[:, 4], kind='stable')]
                images.append({'gt': gt, 'toy': dt})
                path = Path(d) / f'{i}.npz'
                f.match_worker((i, gt, {'toy': ([], dt)}, path, 'fixture'))
                with np.load(path) as a:
                    parts.append({k: a[k].copy() for k in a.files})
            with contextlib.redirect_stdout(io.StringIO()):
                expected = f.dm.evaluate(images, 'toy')
            actual = f.aggregate(parts, 'toy')
            for k in f.KEYS + ['class_macro_AP_separate_diagnostic']:
                self.assertAlmostEqual(actual[k], expected[k], places=12)


if __name__ == '__main__':
    unittest.main()
