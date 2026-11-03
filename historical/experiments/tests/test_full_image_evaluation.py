import contextlib,copy,io,sys,tempfile,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from hycado_start import full_image_evaluation as e


def query(cls=0):
    q=np.zeros((300,14),dtype=np.float32)
    q[:2,:4]=[.15,.15,.1,.1];q[0,4+cls]=.8;q[1,4+cls]=.7
    return q


class EvaluationTests(unittest.TestCase):
    def test_decode_full_coordinates_keeps_overlapping_queries_and_scores(self):
        q=query();rows=e.decode(q,100,100)
        self.assertEqual(len(rows),2)  # NMS must not remove duplicate set predictions.
        self.assertAlmostEqual(rows[0][4],float(q[0,4]));self.assertEqual(rows[0][5],0)
        np.testing.assert_allclose(rows[0][:4],[10,10,20,20],atol=1e-5)
        self.assertEqual(rows,e.r.old.canonical_detections(e.r.old.map_queries(q,{'source_crop_xyxy':[0,0,100,100]}),100,100)[0])
        for bad in [q.astype(np.float64),q[:2],q.copy()]:
            if bad.shape==q.shape and bad.dtype==q.dtype:bad[0,4]=float('nan')
            with self.assertRaises(ValueError):e.decode(bad,100,100)

    def records(self,root,n=2):
        records=[]
        for i in range(n):
            path=root/'images/val'/f'{i}.png';path.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(100,100)).save(path)
            records.append({'image':f'images/val/{i}.png','width':100,'height':100,'image_sha256':e.sha256(path)})
        return records

    def test_prediction_cache_integrity_and_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rec=self.records(root,1)[0];folder=root/'prediction'
            e.save_prediction(folder,query(),rec,'sig');self.assertEqual(len(e.checked_prediction(folder,rec,'sig')['rtdetr_full960']),2)
            with self.assertRaises(ValueError):e.checked_prediction(folder,rec,'changed')
            (folder/'queries.npz').write_bytes(b'bad')
            with self.assertRaises(ValueError):e.checked_prediction(folder,rec,'sig')

    def test_interrupted_inference_reuses_whole_images_without_GT_or_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records=self.records(root,3);out=root/'out';out.mkdir();calls=[0]
            def interrupted(model,path,record):
                calls[0]+=1
                if calls[0]==2:raise InterruptedError('interrupted image')
                return query()
            with patch.object(e,'predict_path',side_effect=interrupted),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(InterruptedError):e.inference(records,root,out,'sig',lambda:object())
            self.assertFalse((out/'prediction_index.json').exists())
            prior=e.sha256(out/'predictions/0/prediction.json')
            with patch.object(e,'predict_path',return_value=query()) as predict,contextlib.redirect_stdout(io.StringIO()):
                index=e.inference(records,root,out,'sig',lambda:object());self.assertEqual(predict.call_count,2)
            self.assertEqual(prior,e.sha256(out/'predictions/0/prediction.json'));self.assertEqual(len(index),3)
            with patch.object(e,'predict_path',side_effect=AssertionError('must reuse cache')):
                with contextlib.redirect_stdout(io.StringIO()):e.inference(records,root,out,'sig',lambda:(_ for _ in ()).throw(AssertionError('no new model')))

    def metric_fixture(self,root):
        from hycado_start import dev_metrics as m
        records=self.records(root,2);out=root/'out';out.mkdir();base=root/'reference';base.mkdir();oldindex=[];images=[]
        for i,rec in enumerate(records):
            anno=root/'visdrone_annotations/val'/f'{i}.txt';anno.parent.mkdir(parents=True,exist_ok=True);anno.write_text(f'10,10,10,10,1,{i+1},0,0\n')
            rec['annotation_sha256']=e.sha256(anno)
            baseline=[[10.,10.,20.,20.,.9,i]] if i==0 else []
            hybrid=[[10.,10.,20.,20.,.9,i],[60.,60.,70.,70.,.1,i]]
            ref=base/f'{i}.json';e.r.val.signed_json(ref,{'protocol_sha256':e.r.P12,'source_image':rec['image'],
                'image_sha256':rec['image_sha256'],'width':100,'height':100,'baseline':baseline,'hybrid_a025':hybrid})
            oldindex.append({'source_image':rec['image'],'path':ref.name,'sha256':e.sha256(ref)})
            images.append({'gt':m.prepare_groundtruth(m.raw_annotation(anno.read_text()),None),
                           'baseline':m.xyxy_to_det(baseline),'hybrid_a025':m.xyxy_to_det(hybrid)})
        e.write_json(base/'prediction_index.json',{'records':oldindex})
        with contextlib.redirect_stdout(io.StringIO()):metrics={name:m.evaluate(images,name) for name in ['baseline','hybrid_a025']}
        for name,tp,fp in [('baseline',1,0),('hybrid_a025',2,2)]:
            metrics[name]['counts_at_fixed_max500']={t:{'TP':tp,'FP':fp,'ignored_detections':0} for t in ['0.5','0.75']}
        e.write_json(base/'metrics.json',{'variants':metrics})
        with patch.object(e,'predict_path',side_effect=lambda model,path,rec:query(int(path.stem))),contextlib.redirect_stdout(io.StringIO()):
            index=e.inference(records,root,out,'sig',lambda:object())
        return records,out,base,index

    def test_same_metric_three_variants_matches_independent_reference_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records,out,base,index=self.metric_fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):result=e.measure(records,root,base,out,'sig',index)
            self.assertAlmostEqual(result['variants']['baseline']['AP'],50.)
            self.assertAlmostEqual(result['variants']['rtdetr_full960']['AP'],100.)
            self.assertAlmostEqual(result['variants']['hybrid_a025']['AP'],100.)
            self.assertEqual(result['variants']['rtdetr_full960']['counts_at_fixed_max500']['0.5'],{'TP':2,'FP':2,'ignored_detections':0})
            with self.assertRaises(ValueError):e.measure(records,root,base,out,'sig',index[:1])
            old=e.read_json(base/'metrics.json');old['variants']['baseline']['AP']=0;e.write_json(base/'metrics.json',old)
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):e.measure(records,root,base,out,'sig',index)

    def test_loader_selects_EMA80_and_rejects_crop_or_incomplete_checkpoint(self):
        from hycado_start import crop_training_v2 as engine
        selected=[]
        class Model:
            def load_state_dict(self,state,strict):selected.append((state,strict))
            def float(self):return self
            def eval(self):return self
            def to(self,device):return self
            def parameters(self):return []
            def state_dict(self):return {}
        saved={'schema':'hycado_crop_experiment_checkpoint_v1','protocol_sha256':e.P18,'round':80,
               'history':[{'optimizer_updates':728}]*79+[{'optimizer_updates':708}], 'ema':{'EMA':'chosen'},'model':{'raw':'wrong'}}
        torch=SimpleNamespace(load=lambda *a,**k:saved,set_num_threads=lambda n:None,
           backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True,deterministic=False)),use_deterministic_algorithms=lambda *a,**k:None)
        cv=SimpleNamespace(setNumThreads=lambda n:None)
        with patch.dict(sys.modules,{'torch':torch,'cv2':cv}),patch.object(engine,'seed_all'),patch.object(engine,'build_model',return_value=Model()):
            e.load_model(Path('weights'),Path('official'),'cpu');self.assertEqual(selected,[({'EMA':'chosen'},True)])
            saved['round']=79
            with self.assertRaises(ValueError):e.load_model(Path('weights'),Path('official'),'cpu')

    def test_report_backup_contains_predictions_not_images_models_or_raw_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);code=root/'hycado_start/full_image_evaluation.py';code.parent.mkdir();code.write_bytes(b'code')
            records,out,base,index=self.metric_fixture(root)
            report=e.bundle(out);e.backup(root,out)
            with zipfile.ZipFile(report) as z:
                self.assertIsNone(z.testzip());self.assertEqual(len([n for n in z.namelist() if n.endswith('prediction.json')]),2)
                self.assertFalse(any(n.endswith(('.png','.pt','.npz')) for n in z.namelist()))


if __name__=='__main__':unittest.main()
