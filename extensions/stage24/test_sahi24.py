import contextlib,io,json,sqlite3,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import common24 as h
import toolkit_core as t
import visdrone_metrics as v
import coco_core as c
import uncertainty_core as u
import p2_sahi_exact as p2

class SahiTests(unittest.TestCase):
    def test_factory_preserves_original_arguments_and_class_names(self):
        calls=[]
        model=types.SimpleNamespace(model=types.SimpleNamespace(names=p2.EXPECTED_NAMES))
        auto=types.SimpleNamespace(from_pretrained=lambda **kw:(calls.append(kw),model)[1])
        with patch.dict('sys.modules',{'sahi':types.SimpleNamespace(AutoDetectionModel=auto)}):
            self.assertIs(p2.sahi_factory(Path('frozen.pt')),model)
            self.assertEqual(calls,[{'model_type':'ultralytics','model_path':'frozen.pt','confidence_threshold':.001,'device':'cuda:0','image_size':800}])
            model.model.names=['wrong']
            with self.assertRaisesRegex(ValueError,'class names'):p2.sahi_factory(Path('frozen.pt'))

    def test_prediction_wrapper_preserves_slice_fusion_and_RGB_settings(self):
        from PIL import Image
        calls=[];sync=[]
        box=types.SimpleNamespace(bbox=types.SimpleNamespace(to_xyxy=lambda:[1,2,11,12]),score=types.SimpleNamespace(value=.7),category=types.SimpleNamespace(id=3))
        def sliced(rgb,model,**kwargs):
            calls.append((rgb.copy(),model,kwargs));return types.SimpleNamespace(object_prediction_list=[box])
        with tempfile.TemporaryDirectory() as f,patch.dict('sys.modules',{
            'torch':types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:sync.append(True))),
            'sahi.predict':types.SimpleNamespace(get_sliced_prediction=sliced)}):
            path=Path(f)/'rgb.png';Image.new('RGB',(80,60),(255,0,0)).save(path)
            w,hh,rows,dropped=p2.sahi_predict('model',path,640)
            self.assertEqual((w,hh,dropped),(80,60,0));self.assertEqual(rows,[[1.,2.,11.,12.,.7,3]])
            self.assertEqual(calls[0][0][0,0].tolist(),[255,0,0]);self.assertEqual(len(sync),2)
            self.assertEqual(calls[0][2],{'slice_height':640,'slice_width':640,'overlap_height_ratio':.2,'overlap_width_ratio':.2,'perform_standard_pred':True,'postprocess_type':'NMS','postprocess_match_metric':'IOU','postprocess_match_threshold':.5,'postprocess_class_agnostic':False,'verbose':0})

    def test_long_SAHI_prediction_is_preserved_and_hash_checked(self):
        rec={'image':'images/test-dev/dense.jpg','stem':'dense','image_sha256':'a'*64,'width':100,'height':100}
        rows=[[1.,1.,2.,2.,.5,0] for _ in range(1018)]
        with tempfile.TemporaryDirectory() as f,contextlib.closing(h.state(Path(f)/'state.sqlite3','sig')) as db:
            h.save_prediction(db,rec,'sig',rows,0)
            self.assertEqual(h.load_prediction(db,rec,'sig')['detections'],rows)
            with db:db.execute("UPDATE prediction SET digest='changed'")
            with self.assertRaisesRegex(ValueError,'hash differs'):h.load_prediction(db,rec,'sig')

    def test_interrupt_resume_does_not_repeat_committed_inference(self):
        with tempfile.TemporaryDirectory() as f:
            out=Path(f);data=out/'dataset';(data/'images').mkdir(parents=True)
            records=[]
            for name in ['one','two']:
                image=data/'images'/(name+'.jpg');image.write_bytes(name.encode())
                records.append({'image':'images/test-dev/'+name+'.jpg','stem':name,'image_sha256':h.sha(image),'width':100,'height':100})
            calls=[]
            def failing(image):
                calls.append(image.stem)
                if image.stem=='two':raise RuntimeError('simulated interruption')
                return 100,100,[[1,1,5,5,.5,0]],0
            with contextlib.closing(h.state(out/'pred.sqlite3','sig')) as db,contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):h.infer(db,records,data,'sig',out,lambda:failing)
                self.assertEqual(db.execute('SELECT COUNT(*) FROM prediction').fetchone()[0],1)
            def resumed(image):calls.append('resume_'+image.stem);return 100,100,[[1,1,5,5,.5,0]],0
            with contextlib.closing(h.state(out/'pred.sqlite3','sig')) as db,contextlib.redirect_stdout(io.StringIO()):
                freeze=h.infer(db,records,data,'sig',out,lambda:resumed)
                self.assertTrue(freeze['all_predictions_committed']);self.assertFalse(freeze['annotations_read_by_infer'])
            self.assertEqual(calls,['one','two','resume_two']);self.assertFalse((data/'annotations').exists())

    def test_binding_mismatch_closes_rejected_SQLite_connection(self):
        with tempfile.TemporaryDirectory() as f:
            path=Path(f)/'cache.sqlite3'
            with contextlib.closing(h.state(path,'one')):pass
            with self.assertRaisesRegex(ValueError,'different inputs'):h.state(path,'two')
            path.rename(path.with_name('preserved.sqlite3'))

    def test_invalid_geometry_and_class_are_not_silently_accepted(self):
        from proposal_geometry import canonical_detections
        for row in [[5,1,2,3,.5,0],[1,1,2,3,float('nan'),0],[1,1,2,3,.5,10]]:
            with self.assertRaises(ValueError):canonical_detections([row],100,100)
        rows,dropped=canonical_detections([[1,0,2,0,.5,0],[1,1,2,2,.5,0]],100,100)
        self.assertEqual((len(rows),dropped),(1,1))

    def test_fast_toolkit_matcher_equals_original_reference(self):
        rng=np.random.default_rng(24)
        for _ in range(6):
            gt=np.array([[float(j*12),5.,10.,10.,int(j%6==0),int(j%3+1)] for j in range(12)])
            det=np.array([[float(j*12+rng.uniform(-2,2)),5.,10.,10.,.9-j*.01,int(j%3+1)] for j in range(15)])
            for threshold in [.5,.75,.95]:
                actual=t.fast_flags(det,gt,threshold);expected=np.zeros(len(det),dtype=np.int8)
                for cat in range(1,11):
                    ids=np.flatnonzero(det[:,5]==cat);expected[ids]=v.match_flags(det[ids,:5],gt[gt[:,5]==cat,:5],threshold)
                np.testing.assert_array_equal(actual,expected)

    def test_original_toolkit_aggregation_and_global_caps(self):
        images=[];parts=[]
        for i in range(4):
            gt=np.array([[1.,1.,10.,10.,0.,1.]]+([[20.,20.,20.,20.,0.,2.]] if i%2 else []))
            det=np.array([[1.,1.,10.,10.,.9,1.],[70.,70.,4.,4.,.8,1.],[20.,20.,20.,20.,.7,2.]])
            images.append({'gt':gt,'sahi640':det})
            g=np.array([np.count_nonzero(gt[:,5]==cat) for cat in range(1,11)])
            parts.append({'gt_denominators':g,'presence':(g>0).astype(int),'sahi640_scores':det[:,4],
                          'sahi640_classes':det[:,5].astype(np.int8),'sahi640_flags':np.column_stack([t.fast_flags(det,gt,float(th)) for th in t.THRESHOLDS])})
        actual=t.aggregate(parts,'sahi640')
        with contextlib.redirect_stdout(io.StringIO()):expected=v.evaluate(images,'sahi640')
        for key in ['AP','AP50','AP75','AR1','AR10','AR100','AR500','class_macro_AP_separate_diagnostic']:self.assertAlmostEqual(actual[key],expected[key],places=10)

    def test_COCO_and_bootstrap_accumulation_preserve_ties(self):
        from pycocotools.cocoeval import COCOeval
        rec={'id':1,'width':100,'height':100};g=[[1,1,10,10,1]];d=[[1,1,10,10,.5,1],[30,30,5,5,.5,1]]
        entries={cat:{} for cat in range(1,11)}
        for e in c.one_image(rec,g,d):
            x,hh,payload=c.encode(e)
            if x['area_index']==0:entries[x['category_id']][1]=c.decode(hh,payload)
        engine=[[u.compile_category(entries[cat],[1]) for cat in range(1,11)]]
        actual=u.evaluate(engine,np.array([2]))[0]
        images=[{'id':1,'width':100,'height':100},{'id':2,'width':100,'height':100}]
        ga=[{'id':i,'image_id':i,'category_id':1,'bbox':[1,1,10,10],'area':100,'iscrowd':0} for i in [1,2]]
        da=[]
        for i in [1,2]:
            for row in d:da.append({'id':len(da)+1,'image_id':i,'category_id':1,'bbox':row[:4],'area':row[2]*row[3],'score':row[4],'iscrowd':0})
        ev=COCOeval(c.coco_object(images,ga),c.coco_object(images,da),'bbox');ev.params.maxDets=[1,10,100]
        with contextlib.redirect_stdout(io.StringIO()):ev.evaluate();ev.accumulate()
        p=ev.eval['precision'];rr=ev.eval['recall'];expected=[c.mean(p[:,:,:,0,2]),c.mean(p[0,:,:,0,2]),c.mean(p[5,:,:,0,2]),c.mean(rr[:,:,0,2])]
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)

    def test_frozen_23_baseline_draws_are_complete_and_paired(self):
        import run_sahi24 as runner
        values=runner.baseline_draws();self.assertEqual(values.shape,(1000,2,4));self.assertTrue(np.isfinite(values).all())
        # The retained differences reproduce the previously verified primary CI.
        diff=values[:,1,0]-values[:,0,0]
        np.testing.assert_allclose(np.quantile(diff,[.025,.975],method='linear'),[.16128724430974808,.8375766091324609],atol=1e-10,rtol=0)

    def test_native_match_cache_tampering_is_rejected(self):
        import run_sahi24 as runner
        part={'gt_denominators':np.array([1]+[0]*9)}
        digest,payload=runner.pack_part(part);np.testing.assert_array_equal(runner.load_part(digest,payload)['gt_denominators'],part['gt_denominators'])
        with self.assertRaises(ValueError):runner.load_part('wrong',payload)

    def test_prediction_identity_cannot_be_reused_for_other_image(self):
        rec={'image':'images/test-dev/a.jpg','stem':'a','image_sha256':'a'*64,'width':100,'height':100}
        with tempfile.TemporaryDirectory() as f,contextlib.closing(h.state(Path(f)/'state.sqlite3','sig')) as db:
            h.save_prediction(db,rec,'sig',[],0)
            with self.assertRaisesRegex(ValueError,'identity'):h.load_prediction(db,{**rec,'image_sha256':'b'*64},'sig')

    def test_new_SAHI_bootstrap_is_paired_and_resumes_existing_draws(self):
        import run_sahi24 as runner
        import resampling as res
        records=[{'id':i,'image':str(i)+'.jpg','stem':str(i),'image_sha256':str(i)*64,'width':100,'height':100} for i in [1,2]]
        groups=res.make_groups(records);entries={cat:{} for cat in range(1,11)}
        for rec in records:
            for e in c.one_image(rec,[[1,1,10,10,1]],[[1,1,10,10,.9,1]]):
                x,digest,payload=c.encode(e)
                if x['area_index']==0:entries[x['category_id']][rec['id']]=c.decode(digest,payload)
        fake_old=np.tile(np.array([[20.,30.,21.,40.],[20.5,30.5,21.5,40.5]])[None,:,:],(1000,1,1))
        original_read=runner.read
        def fixtures(path):
            if Path(path).name=='bootstrap_groups.json':return groups
            if Path(path).name=='23_point.json':return {'coco':fake_old[0].tolist()}
            return original_read(path)
        coco={'100':{'AP':100.,'AP50':100.,'AP75':100.,'AR':100.}}
        with tempfile.TemporaryDirectory() as f,patch.object(runner,'read',fixtures),patch.object(runner,'baseline_draws',lambda:fake_old),contextlib.redirect_stdout(io.StringIO()):
            out=Path(f)
            with contextlib.closing(h.state(out/'state.sqlite3','sig')) as db:
                first=runner.uncertainty(db,records,coco,lambda cat,area:entries[cat],out)
                self.assertEqual(db.execute('SELECT COUNT(*) FROM bootstrap').fetchone()[0],1000)
            counter=[];original_evaluate=u.evaluate
            def counted(*args):counter.append(True);return original_evaluate(*args)
            with contextlib.closing(h.state(out/'state.sqlite3','sig')) as db,patch.object(u,'evaluate',counted):
                second=runner.uncertainty(db,records,coco,lambda cat,area:entries[cat],out)
                self.assertEqual(len(counter),1)  # Full-data point gate only; every draw reused.
                self.assertEqual(first,second)
                v=second['differences_pp']['hybrid1280_minus_sahi640']['AP']
                for key in ['point','lower95','upper95']:self.assertAlmostEqual(v[key],-79.5)

if __name__=='__main__':unittest.main()
