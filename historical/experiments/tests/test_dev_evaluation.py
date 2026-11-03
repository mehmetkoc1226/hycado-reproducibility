"""Development inference/metric tests; synthetic fixtures, no real predictions."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hycado_start import dev_metrics as m
from hycado_start import dev_evaluation as e
from hycado_start.common import write_json, sha256
from hycado_start.stage1 import fingerprint


class MetricsTests(unittest.TestCase):
    def test_exact_positive_duplicate_wrong_class_empty(self):
        gt=np.array([[10,10,10,10,0,1]],dtype=float)
        det=np.array([[10,10,10,10,.9,1],[10,10,10,10,.8,1]],dtype=float)
        flags=m.match_flags(det[:,:5],gt[:,:5],.5)
        np.testing.assert_array_equal(flags,[1,0])
        self.assertEqual(m.precision_area(det[:,4],flags,1),(100.,100.))
        self.assertEqual(m.precision_area(np.empty(0),np.empty(0),1),(0.,0.))
        metrics=m.evaluate([{'gt':gt,'a':det}],'a')
        self.assertEqual(metrics['AP'],100.)
        bad=det.copy();bad[:,5]=2
        self.assertEqual(m.evaluate([{'gt':gt,'a':bad}],'a')['AP'],0.)

    def test_positive_precedes_ignore_and_ignore_can_match_repeatedly(self):
        gt=np.array([[0,0,20,20,1],[0,0,10,10,0]],dtype=float)
        dt=np.array([[0,0,10,10,.9],[0,0,10,10,.8],[1,1,4,4,.7]],dtype=float)
        np.testing.assert_array_equal(m.match_flags(dt,gt,.5),[1,-1,-1])

    def test_greedy_tie_chooses_last_equal_gt(self):
        gt=np.array([[0,0,10,10,0],[0,0,10,10,0]],dtype=float)
        dt=np.array([[0,0,10,10,.9],[0,0,10,10,.8]],dtype=float)
        np.testing.assert_array_equal(m.match_flags(dt,gt,.5),[1,1])

    def test_cap_is_across_classes_not_per_class(self):
        gt=np.array([[0,0,10,10,0,1]],dtype=float)
        dt=np.array([[0,0,10,10,.99,2],[0,0,10,10,.9,1]],dtype=float)
        result=m.evaluate([{'gt':gt,'a':dt}],'a')
        self.assertEqual(result['AR1'],0.)
        self.assertEqual(result['AR10'],100.)

    def test_reference_aggregation_preserves_image_multiplicity(self):
        g1=np.array([[0,0,10,10,0,1]],dtype=float)
        g2=np.array([[0,0,10,10,0,2]],dtype=float)
        d=np.array([[0,0,10,10,.9,1]],dtype=float)
        result=m.evaluate([{'gt':g1,'a':d},{'gt':g1,'a':d},{'gt':g2,'a':np.empty((0,6))}],'a')
        self.assertAlmostEqual(result['AP'],100*2/3)
        self.assertEqual(result['class_macro_AP_separate_diagnostic'],50.)

    def test_reference_denominator_includes_ignored_gt(self):
        gt=np.array([[0,0,10,10,0,1],[20,20,10,10,1,1]],dtype=float)
        dt=np.array([[0,0,10,10,.9,1]],dtype=float)
        self.assertEqual(m.evaluate([{'gt':gt,'a':dt}],'a')['AP'],50.)

    def test_pixel_ignore_union_and_half_boundary(self):
        raw=np.array([[1,1,5,10,0,0,0,0],[3,1,3,10,0,0,0,0]],dtype=float)
        integral=m.ignore_integral(raw,20,20)
        dt=np.array([[1,1,10,10,.9,1],[12,12,4,4,.8,1]],dtype=float)
        kept=m.outside_ignore(dt,integral)
        self.assertEqual(len(kept),1);self.assertEqual(kept[0,0],12)

    def test_raw_annotation_keeps_other_score0_duplicates_and_invalid_geometry(self):
        raw=m.raw_annotation('0,0,0,1,1,1,0,0\n1,1,2,3,0,4,0,0\n1,1,2,3,1,11,0,0,\n')
        self.assertEqual(raw.shape,(3,8));self.assertEqual(raw[0,2],0)
        with self.assertRaises(ValueError):m.raw_annotation('0,0,1,1,nan,1,0,0')


class InferenceGeometryTests(unittest.TestCase):
    def test_square_resize_inverse_and_translation(self):
        query=np.zeros((1,14));query[0,:4]=[.5,.5,.5,.5];query[0,7]=.9
        crop={'source_crop_xyxy':[100,200,500,400]}
        self.assertEqual(e.map_queries(query,crop),[[200.,250.,400.,350.,.9,3.]])

    def test_query_clipping_best_class_and_strict_confidence(self):
        q=np.zeros((3,14));q[:,:4]=[.1,.9,.6,.6]
        q[0,4:6]=[.5,.6];q[1,4]=.001;q[2,13]=.9
        rows=e.map_queries(q,{'source_crop_xyxy':[0,0,100,50]})
        self.assertEqual(len(rows),2);self.assertEqual(rows[0][5],1);self.assertEqual(rows[1][5],9)
        self.assertEqual(rows[0][:4],[0.,30.000000000000004,40.,50.])
        q[0,0]=np.nan
        with self.assertRaises(ValueError):e.map_queries(q,{'source_crop_xyxy':[0,0,100,50]})

    def test_classwise_nms_ties_cross_class_and_cap(self):
        rows=[[0,0,10,10,.9,0],[0,0,10,10,.8,0],[0,0,10,10,.9,1],[20,20,30,30,.7,0]]
        result=e.classwise_nms(rows,40,40)
        self.assertEqual(len(result),3);self.assertEqual([r[5] for r in result[:2]],[0,1])
        self.assertEqual(len(e.classwise_nms(rows,40,40,limit=1)),1)

    def test_cache_signature_detects_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'a.json'
            p={'protocol_sha256':'recipe','source_image':'src','width':10,'height':10,'baseline':[],'hycado':[]}
            p['content_sha256']=fingerprint(p);write_json(path,p)
            self.assertEqual(e.cached_payload(path,'recipe','src')['hycado'],[])
            p['hycado']=[[0,0,1,1,.9,0]];write_json(path,p)
            with self.assertRaises(ValueError):e.cached_payload(path,'recipe','src')

    def test_export_preserves_native_geometry_and_official_class_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'a.txt';e.export_txt(path,[[10,20,30,60,.9,9]])
            v=list(map(float,path.read_text().strip().split(',')))
            self.assertEqual(v,[10,20,20,40,.9,10,-1,-1])


class CacheRecoveryTests(unittest.TestCase):
    def test_interrupted_inference_reuses_completed_images_and_keeps_all_crops(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'new_eval';out.mkdir()
            data=root/'experiments/stage1_v1_seed42/data'
            crops=root/'experiments/crop_dataset_v1_seed42'
            pred=root/'experiments/stage1_predictions_v1_seed42'
            sources={'index':{}};parents=[];cfg={'seed':42,'test_recipe':True}
            for i in range(2):
                stem=f'example_{i}';image=data/f'images/val/{stem}.jpg';image.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(100,80),(40,50,60)).save(image)
                ann=data/f'annotations/val/{stem}.txt';ann.parent.mkdir(parents=True,exist_ok=True)
                ann.write_text('10,10,20,20,1,1,0,0\n')
                record={'source_image':f'images/train/{stem}.jpg','split':'val','image':f'images/val/{stem}.jpg',
                        'image_sha256':sha256(image),'annotation_sha256':sha256(ann)}
                p={'protocol_sha256':e.P06,'source_image':record['source_image'],'split':'val',
                   'image_sha256':record['image_sha256'],'width':100,'height':80,'detections':[[10.,10.,30.,30.,.8,0]]}
                p['content_sha256']=fingerprint(p);path=pred/f'predictions/val/{stem}.json';write_json(path,p)
                sources['index'][record['source_image']]={'sha256':sha256(path)}
                folder=crops/f'items/val/{stem}';folder.mkdir(parents=True)
                options=[]
                for j in range(2):
                    path=folder/f'crop_{j}.png';Image.new('RGB',(50,40),(j,0,0)).save(path)
                    options.append({'image':path.name,'image_sha256':sha256(path),'width':50,'height':40,
                                    'crop_index':j,'source_crop_xyxy':[0,0,50,40],
                                    'supervision_state':'fully_ignored_no_positive' if j else 'positive'})
                parents.append((record,{'width':100,'height':80,'crops':options},folder))
            immutable={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            attempts=[]
            def interrupted_factory(*args):
                def predict(path,crop):
                    attempts.append(str(path))
                    if len(attempts)==3:raise RuntimeError('synthetic GPU stop')
                    return [[10.,10.,30.,30.,.9,0]]
                return predict
            with self.assertRaisesRegex(RuntimeError,'GPU stop'):
                e.infer(root,out,sources,cfg,None,None,parents,predictor_factory=interrupted_factory)
            cache=out/'predictions/example_0.json';first=cache.read_bytes()
            resumed=[]
            def resume_factory(*args):
                def predict(path,crop):
                    resumed.append(crop['supervision_state']);return [[10.,10.,30.,30.,.9,0]]
                return predict
            index=e.infer(root,out,sources,cfg,None,None,parents,predictor_factory=resume_factory)
            self.assertEqual(len(index),2);self.assertEqual(first,cache.read_bytes())
            self.assertIn('fully_ignored_no_positive',resumed)
            self.assertEqual(len(resumed),2)  # only the second source image is replayed.
            self.assertTrue(all(p.read_bytes()==b for p,b in immutable.items()))
            metrics=e.measure(out,index,fingerprint(cfg))
            self.assertEqual(metrics['baseline']['AP'],100.)
            self.assertEqual(metrics['hycado']['AP'],100.)
            with __import__('zipfile').ZipFile(e.bundle(out)) as z:
                self.assertIn('metrics.json',z.namelist())
                self.assertTrue(all(not n.endswith(('.pt','.png','.jpg')) for n in z.namelist()))


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('ultralytics'),
                     'Optional CPU RT-DETR API and upstream decoder check')
class NativeModelTests(unittest.TestCase):
    def test_real_rtdetr_960_output_and_upstream_coordinate_decoder(self):
        import torch
        from types import SimpleNamespace
        from ultralytics.nn.tasks import RTDETRDetectionModel
        from ultralytics.models.rtdetr.predict import RTDETRPredictor
        torch.set_num_threads(2);torch.manual_seed(42)
        model=RTDETRDetectionModel('rtdetr-l.yaml',nc=10,ch=3,verbose=False).eval()
        with torch.inference_mode():prediction=model.predict(torch.zeros(1,3,960,960))[0]
        self.assertEqual(tuple(prediction.shape),(1,300,14))
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        # Independent native Ultralytics decoder against our mapping, fixed inputs.
        prediction=prediction.clone()
        prediction.zero_();prediction[0,0,:4]=torch.tensor([.5,.5,.4,.2]);prediction[0,0,7]=.9
        predictor=object.__new__(RTDETRPredictor)
        predictor.args=SimpleNamespace(conf=.001,classes=None,max_det=300)
        predictor.model=SimpleNamespace(names={i:str(i) for i in range(10)})
        predictor.batch=(['sample.png'],None,None)
        expected=predictor.postprocess((prediction,None),torch.zeros(1,3,960,960),[np.zeros((333,1001,3),dtype=np.uint8)])[0].boxes.data.numpy()
        actual=e.map_queries(prediction[0].numpy(),{'source_crop_xyxy':[0,0,1001,333]})
        np.testing.assert_allclose(actual,expected,rtol=0,atol=5e-5)


if __name__=='__main__':unittest.main()
