"""Regression tests for spatial coverage and restartable frozen-model inference."""
import copy
import json
import random
import tempfile
import unittest
import zipfile
from pathlib import Path

from hycado_start.common import sha256, write_json
from hycado_start.stage1 import fingerprint
from hycado_start.proposal_geometry import (canonical_detections, coverage_counts,
                                           load_targets, select_regions, union_area)
from hycado_start import proposals


class GeometryTests(unittest.TestCase):
    def test_invalid_detection_values_do_not_silently_pass(self):
        for row in [[0,0,1,1,float('nan'),0],[0,0,1,1,.5,10],[2,0,1,1,.5,0], [0,0,1,1,.5,.5]]:
            with self.assertRaises(ValueError):
                canonical_detections([row], 100, 100)

    def test_original_coordinates_clipping_zero_area_and_tie_order(self):
        rows, n = canonical_detections([[90,90,120,110,.7,1], [0,0,0,8,.8,0],
                                        [1,1,4,4,.7,0]], 100, 100)
        self.assertEqual(n, 1)
        self.assertEqual(rows, [[1.,1.,4.,4.,.7,0],[90.,90.,100,100,.7,1]])

    def test_context_geometry_is_native_image_not_resized_coordinates(self):
        rows, _ = canonical_detections([[40,20,60,40,.6,0]], 200, 100)
        self.assertEqual(select_regions(rows,200,100,.6,10,2),[(30,10,70,50)])

    def test_half_open_pixel_crop_rounding_and_image_boundary(self):
        rows, _ = canonical_detections([[0.2,0.4,8.8,9.9,.8,0]],10,10)
        self.assertEqual(select_regions(rows,10,10,.1,1,3),[(0,0,10,10)])

    def test_top_k_does_not_discard_cached_low_rank_detections(self):
        rows, _ = canonical_detections([[i,0,i+1,1,.9-i*.01,0] for i in range(5)],20,20)
        self.assertEqual(len(select_regions(rows,20,20,.1,2,1)),2)
        self.assertEqual(len(rows),5)

    def test_exact_crop_duplicates_removed_but_overlap_kept(self):
        rows, _ = canonical_detections([[0,0,10,10,.9,0],[0,0,10,10,.8,1],[5,0,15,10,.7,0]],20,20)
        self.assertEqual(select_regions(rows,20,20,.1,3,1),[(0,0,10,10),(5,0,15,10)])

    def test_no_detection_means_no_oracle_fallback_crop(self):
        self.assertEqual(select_regions([],100,100,.1,60,3),[])
        c = coverage_counts([[0,0,10,10,0]],[])
        self.assertEqual(c['all'],{'targets':1,'center':0,'ioa50':0,'ioa90':0,'ioa100':0})

    def test_spatial_coverage_is_class_agnostic(self):
        c = coverage_counts([[0,0,10,10,4]],[(0,0,10,10)])
        self.assertEqual(c['classes']['4']['ioa90'],1)
        self.assertEqual(c['classes']['0']['targets'],0)

    def test_two_half_crops_are_not_one_complete_crop(self):
        c = coverage_counts([[0,0,10,10,0]],[(0,0,5,10),(5,0,10,10)])
        self.assertEqual(c['all']['ioa50'],1)
        self.assertEqual(c['all']['ioa90'],0)
        self.assertEqual(union_area([(0,0,5,10),(5,0,10,10)]),100)

    def test_union_area_against_independent_pixel_set_oracle(self):
        rng = random.Random(41)
        for _ in range(50):
            rects = []
            for _ in range(rng.randrange(10)):
                a,b = sorted(rng.sample(range(11),2)); c,d = sorted(rng.sample(range(11),2))
                rects.append((a,c,b,d))
            pixels = {(x,y) for a,c,b,d in rects for x in range(a,b) for y in range(c,d)}
            self.assertEqual(union_area(rects),len(pixels))

    def test_empty_targets_small_boundary_and_class_counts(self):
        self.assertEqual(coverage_counts([],[(0,0,1,1)])['all']['targets'],0)
        c = coverage_counts([[0,0,32,32,0],[0,0,31,32,1]],[(0,0,32,32)])
        self.assertEqual(c['small_native_area_lt_1024']['targets'],1)
        self.assertEqual(c['all']['ioa100'],2)

    def test_labels_to_original_pixels_and_invalid_boxes(self):
        self.assertEqual(load_targets('4 .5 .5 .2 .4\n',100,50),[[40.,15.,60.,35.,4]])
        for text in ['10 .5 .5 .2 .2','0 .5 .5 0 .1','0 1 .5 .5 .5','0 nan .5 .2 .2']:
            with self.assertRaises(ValueError):
                load_targets(text,100,100)


class CacheCoverageTests(unittest.TestCase):
    def test_manifest_rejects_official_val_and_group_hash_overlap(self):
        records = [{'split':'train' if i<5824 else 'val', 'source_split':'train',
                    'source_image':f'images/train/{i}.jpg',
                    'image':f"images/{'train' if i<5824 else 'val'}/{i}.jpg",
                    'group':str(i), 'image_sha256':str(i)} for i in range(6471)]
        manifest={'records':records,'official_val_used_for_training_or_selection':False}
        proposals.validate_manifest(manifest)
        for field, value in [('source_split','val'),('group','0'),('image_sha256','0')]:
            edited=copy.deepcopy(manifest);edited['records'][-1][field]=value
            with self.assertRaises(ValueError):
                proposals.validate_manifest(edited)

    def fixture(self, root):
        records = []
        for split, stem in [('train','one'),('val','two')]:
            image = f'images/{split}/{stem}.jpg'
            path = root/'data'/image;path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(b'image')
            label = root/'data/labels'/split/(stem+'.txt');label.parent.mkdir(parents=True,exist_ok=True)
            label.write_text('4 .5 .5 .2 .2\n')
            records.append({'image':image,'source_image':f'images/train/{stem}.jpg','split':split,
                            'image_sha256':sha256(path),'gt_count':1})
        output = root/'output';output.mkdir()
        return records, output

    def test_atomic_cache_resume_with_no_model_loading_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);records,out = self.fixture(root)
            calls = []
            def predict(path):
                calls.append(path)
                return 100,100,[[40,40,60,60,.9,4],[0,0,5,5,.001,0]],0
            first = proposals.cache_predictions(records,root/'data',out,'id',lambda:predict)
            self.assertEqual(first['new_this_invocation'],2)
            def fail():
                self.fail('Valid cache must not load a model')
            second = proposals.cache_predictions(records,root/'data',out,'id',fail)
            self.assertEqual(second['reused_this_invocation'],2)
            self.assertEqual(len(calls),2)
            path = proposals.prediction_path(out,records[0]);body = json.loads(path.read_text())
            body['detections'][0][4]=.2;write_json(path,body)
            with self.assertRaisesRegex(ValueError,'hash'):
                proposals.cache_predictions(records,root/'data',out,'id',fail)

    def test_interrupted_prediction_continues_from_last_atomic_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records,out=self.fixture(root)
            count=0
            def interrupted(path):
                nonlocal count
                count+=1
                if count==2:raise InterruptedError('synthetic interruption')
                return 100,100,[],0
            with self.assertRaises(InterruptedError):
                proposals.cache_predictions(records,root/'data',out,'id',lambda:interrupted)
            self.assertTrue(proposals.prediction_path(out,records[0]).exists())
            calls=[]
            def resume(path):calls.append(path);return 100,100,[],0
            report=proposals.cache_predictions(records,root/'data',out,'id',lambda:resume)
            self.assertEqual(report['new_this_invocation'],1)
            self.assertEqual(report['reused_this_invocation'],1)
            self.assertEqual(len(calls),1)

    def test_coverage_uses_development_only_and_never_writes_crop_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records,out=self.fixture(root)
            proposals.cache_predictions(records,root/'data',out,'id',lambda:lambda _: (100,100,[[40,40,60,60,.9,4]],0))
            cfg={'coverage':{'thresholds':[.05,.95],'top_k':[1],'context_scales':[1.,2.]}}
            report=proposals.measure_coverage(records,root/'data',out,'id',cfg)
            self.assertEqual(report['development_images'],1)
            self.assertEqual(report['targets'],1)
            rows=json.loads((out/'coverage_summary.json').read_text())['configurations']
            self.assertEqual(rows[0]['coverage_ioa90'],1)
            self.assertEqual(rows[-1]['coverage_ioa90'],0)
            self.assertEqual(rows[-1]['zero_crop_images'],1)
            self.assertEqual(rows[0]['mean_union_fraction'],.04)
            self.assertIsNone(report['auto_selected_configuration'])
            self.assertFalse((out/'labels').exists())
            self.assertNotIn('one.jpg',(out/'coverage_per_image.jsonl').read_text())

    def test_wrong_protocol_and_source_identity_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records,out=self.fixture(root)
            proposals.cache_predictions(records,root/'data',out,'id',lambda:lambda _: (100,100,[],0))
            body=json.loads(proposals.prediction_path(out,records[0]).read_text())
            with self.assertRaisesRegex(ValueError,'kimligi'):
                proposals.validate_cache(body,records[0],'different')

    def test_bundle_contains_no_weights_predictions_images_or_training_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'weights').mkdir();(out/'weights/a.pt').write_bytes(b'x')
            (out/'predictions').mkdir();(out/'predictions/x.json').write_text('{}')
            write_json(out/'result.json',{'status':'COMPLETED'})
            with zipfile.ZipFile(proposals.make_report(out)) as z:
                self.assertEqual(z.namelist(),['result.json'])

    def test_error_bundle_excludes_previous_success_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            write_json(out/'result.json',{'status':'ERROR'})
            write_json(out/'coverage_summary.json',{'old_success':True})
            with zipfile.ZipFile(proposals.make_report(out)) as z:
                self.assertNotIn('coverage_summary.json',z.namelist())


if __name__ == '__main__':
    unittest.main()
