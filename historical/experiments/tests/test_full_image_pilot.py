import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from PIL import Image
from hycado_start import full_image_pilot as p
from hycado_start.data import parse_annotation
from hycado_start.stability import inspect_labels


def fixture(root,ntrain=15,ndev=3,text=None):
    data=root/'source';records=[];prepared=[]
    if text is None:text='5,7,10,12,1,1,0,0\n30,30,10,10,1,11,0,0\n50,30,10,10,0,2,0,0\n5,7,10,12,1,1,0,0\n'
    rng=np.random.default_rng(3)
    for i in range(ntrain+ndev):
        role='train' if i<ntrain else 'val';name=f'{i}.jpg';path=data/'images'/role/name;path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(rng.integers(0,256,(80,100,3),dtype=np.uint8)).save(path)
        anno=data/'annotations'/role/(str(i)+'.txt');anno.parent.mkdir(parents=True,exist_ok=True);anno.write_text(text)
        labels=parse_annotation(text,100,80)['labels']
        converted=''.join(str(r[0])+' '+' '.join(f'{v:.10f}' for v in r[1:])+'\n' for r in labels)
        clean,_,count=inspect_labels(converted,'test');lab=data/'labels'/role/(str(i)+'.txt');lab.parent.mkdir(parents=True,exist_ok=True);lab.write_text(clean)
        record={'source_image':'images/train/'+name,'source_split':'train','split':role,'group':f'g{i}',
            'image':f'images/{role}/{name}','image_sha256':p.sha256(path),'annotation_sha256':p.sha256(anno),
            'label_sha256':p.sha256(lab),'gt_count':count,'width':100,'height':80}
        records.append(record);prepared.append({'split':'train','image':record['source_image'],'width':100,'height':80,
                            'image_sha256':record['image_sha256'],'annotation_sha256':record['annotation_sha256']})
    return data,records,{'records':prepared}


class FullImageTests(unittest.TestCase):
    def test_plan_is_deterministic_TRAIN_only_and_includes_maximum_GT(self):
        records=[{'source_image':str(i),'split':'train' if i<30 else 'val','gt_count':i} for i in range(35)]
        a=p.select_plan(records,representative=4,stress_count=2,development=2)
        self.assertEqual(a,p.select_plan(records[::-1],representative=4,stress_count=2,development=2))
        reps=[r['source_image'] for phase in a for r in phase['representative']];self.assertEqual(len(set(reps)),12)
        self.assertEqual([r['gt_count'] for r in a[0]['stress']],[29,28])
        self.assertTrue(all(r['split']=='train' for phase in a for k in ['representative','stress'] for r in phase[k]))
        self.assertTrue(all(r['split']=='val' for phase in a for r in phase['development']))

    def test_catalog_joins_original_dimensions_and_checks_annotation_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,records,prepared=fixture(Path(tmp),1,1)
            native=[{k:v for k,v in r.items() if k not in ['width','height']} for r in records]
            with contextlib.redirect_stdout(io.StringIO()):result=p.catalog(data,{'records':native},prepared)
            self.assertEqual(result,records)
            (data/'annotations/train/0.txt').write_text('bad')
            with self.assertRaises(ValueError):p.catalog(data,{'records':records},prepared)

    def test_full_frame_pixels_targets_ignores_and_native_loader(self):
        from hycado_start.crop_loader import load_sample
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,records,_=fixture(root,1,0);out=root/'pilot';item=p.materialize(data,out,records[0],'fixed')
            ann=p.read_json(out/item['annotation']);self.assertEqual(ann['source_crop_xyxy'],[0,0,100,80])
            self.assertEqual(len(ann['positive_targets']),1);self.assertEqual(len(ann['ignore_regions']),2)
            self.assertEqual(ann['positive_targets'][0]['box_xyxy'],[5.,7.,15.,19.])
            self.assertEqual(ann['sample_kind'],'whole_image')
            with Image.open(data/records[0]['image']) as image:expected=np.array(image.convert('RGB'))
            with Image.open(out/item['image']) as image:np.testing.assert_array_equal(np.array(image),expected)
            batch=load_sample(out,item,128,'cpu');self.assertEqual(tuple(batch['img'].shape),(1,3,128,128))
            self.assertEqual(tuple(batch['bboxes'].shape),(1,4));self.assertEqual(tuple(batch['ignore_boxes'][0].shape),(2,4))

    def test_empty_and_fully_ignored_frames_remain_whole_and_loadable(self):
        from hycado_start.crop_loader import load_sample
        for text,state in [('', 'clean_background'),('0,0,100,80,0,0,0,0\n','fully_ignored_no_positive')]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);data,records,_=fixture(root,1,0,text=text);out=root/'pilot';item=p.materialize(data,out,records[0],'fixed')
                self.assertEqual(item['supervision_state'],state);self.assertEqual(item['positive_targets'],0)
                self.assertEqual(tuple(load_sample(out,item,128,'cpu')['bboxes'].shape),(0,4))

    def test_materialization_resume_corruption_rejection_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,records,_=fixture(root,1,0);out=root/'pilot'
            before={x:x.read_bytes() for x in data.rglob('*') if x.is_file()}
            with patch.object(Image.Image,'save',side_effect=OSError('interrupted image write')):
                with self.assertRaises(OSError):p.materialize(data,out,records[0],'fixed')
            self.assertFalse(list(out.rglob('full.png')));self.assertFalse(list(out.rglob('*_building_*')))
            item=p.materialize(data,out,records[0],'fixed');self.assertEqual(p.materialize(data,out,records[0],'fixed'),item)
            self.assertTrue(all(x.read_bytes()==b for x,b in before.items()))
            (out/item['image']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):p.materialize(data,out,records[0],'fixed')

    def test_actual_rtdetr_full_frame_loader_forward_backward(self):
        import torch
        from ultralytics.nn.tasks import RTDETRDetectionModel
        from hycado_start.crop_loader import load_sample
        from hycado_start.crop_criterion import IgnoreAwareRTDETRLoss,forward_loss
        torch.set_num_threads(2);torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,records,_=fixture(root,1,0);out=root/'pilot';item=p.materialize(data,out,records[0],'fixed')
            batch=load_sample(out,item,128,'cpu');model=RTDETRDetectionModel('rtdetr-l.yaml',nc=10,verbose=False).train()
            criterion=IgnoreAwareRTDETRLoss(10);loss,_=forward_loss(model,batch,criterion);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertGreater(criterion.audit['denoising_branch_calls'],0)
            self.assertTrue(all(x.grad is None or torch.isfinite(x.grad).all() for x in model.parameters()))
            model.eval()
            with torch.no_grad():loss,_=forward_loss(model,batch,criterion)
            self.assertTrue(torch.isfinite(loss))
            # Real model training forward also handles no positive GT + full ignore.
            model.train();model.zero_grad(set_to_none=True)
            empty={'img':batch['img'],'cls':torch.empty(0,dtype=torch.long),'bboxes':torch.empty(0,4),
                   'batch_idx':torch.empty(0,dtype=torch.long),'ignore_boxes':[torch.tensor([[0.,0.,1.,1.]])]}
            loss,_=forward_loss(model,empty,criterion);loss.backward();self.assertTrue(torch.isfinite(loss))

    def test_prepared_plan_native_items_are_identical_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,records,_=fixture(root);out=root/'pilot'
            selection=p.select_plan(records,representative=4,stress_count=2,development=2)
            with patch.object(p,'select_plan',return_value=selection),contextlib.redirect_stdout(io.StringIO()):
                a=p.prepare_plan(data,out,records,{'fixed':'data'})
                b=p.prepare_plan(data,out,records,{'fixed':'data'})
                self.assertEqual(a,b);self.assertEqual(sum(len(x['representative'])+len(x['stress']) for x in a),18)
                with self.assertRaises(ValueError):p.prepare_plan(data,out,records,{'changed':'data'})

    def test_no_CUDA_run_fails_before_PNG_or_training_and_preserves_old_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'experiments/rtdetr_crop_v1_seed42/keep.txt';old.parent.mkdir(parents=True);old.write_bytes(b'preserved')
            code=root/'hycado_start/full_image_pilot.py';code.parent.mkdir();code.write_bytes(b'code')
            with patch.object(p,'ROOT',root),patch.object(p,'sources',return_value=(root,{},root/'official.pt',{})),\
                 patch.object(p,'system_info',return_value={'cuda_available':False}),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertFalse(p.run())
            output=root/'experiments'/p.OUT;self.assertEqual(p.read_json(output/'result.json')['status'],'ERROR')
            self.assertEqual(old.read_bytes(),b'preserved');self.assertFalse((output/'data').exists())

    def test_reused_engine_recovers_optimizer_ema_rng_with_real_whole_frame_samples(self):
        import torch
        from hycado_start import crop_training_v2 as engine
        from hycado_start.crop_loader import load_sample
        cfg={'seed':42,'amp':False,'imgsz':32,'effective_batch':3,'lr':.0001,'weight_decay':.0001,
             'betas':[.9,.999],'gradient_clip_norm':10.,'ema_decay':.9999,'ema_tau':2000,'warmup_timing_samples_per_phase':1}
        def model(*args):return torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*32*32,5),torch.nn.Dropout(.2),torch.nn.Linear(5,1))
        def loss(model,batch,criterion):
            value=(model(batch['img'])-.3).square().mean();return value,{'loss':value}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,records,_=fixture(root,5,2);out=root/'pilot'
            items=[p.materialize(data,out,r,'fixed') for r in records]
            plan=[{'phase':i,'representative':items[:4],'stress':items[4:5],'development':items[5:]} for i in range(1,4)]
            full=root/'full';resume=root/'resume';full.mkdir();resume.mkdir();calls=[0]
            def interrupted(*args):
                calls[0]+=1
                if calls[0]==10:raise InterruptedError('phase 2 interrupted')
                return load_sample(*args)
            with patch.object(engine,'build_model',side_effect=model),patch('hycado_start.crop_criterion.forward_loss',side_effect=loss),contextlib.redirect_stdout(io.StringIO()):
                engine.train_core(out,full,None,cfg,plan,device='cpu')
                with patch.object(engine,'load_sample',side_effect=interrupted):
                    with self.assertRaises(InterruptedError):engine.train_core(out,resume,None,cfg,plan,device='cpu')
                self.assertEqual(p.read_json(resume/'checkpoint_index.json')['latest']['epoch'],1)
                engine.train_core(out,resume,None,cfg,plan,device='cpu')
            def saved(folder):return torch.load(folder/p.read_json(folder/'checkpoint_index.json')['latest']['path'],map_location='cpu',weights_only=False)
            a,b=saved(full),saved(resume)
            for key in ['model','ema']:
                for name in a[key]:torch.testing.assert_close(a[key][name],b[key][name],rtol=0,atol=0)
            for i in a['optimizer']['state']:
                for key in a['optimizer']['state'][i]:torch.testing.assert_close(a['optimizer']['state'][i][key],b['optimizer']['state'][i][key],rtol=0,atol=0)
            torch.testing.assert_close(a['rng']['torch'],b['rng']['torch'],rtol=0,atol=0)
            self.assertEqual(a['ema_updates'],b['ema_updates'])

    def test_projection_uses_representative_training_only_and_reports_limits(self):
        history=[{'representative_timed_samples':2,'mean_seconds_per_training_sample':1.,'peak_cuda_reserved_gib':3.,'peak_cuda_allocated_gib':2.},
                 {'representative_timed_samples':4,'mean_seconds_per_training_sample':2.,'peak_cuda_reserved_gib':4.,'peak_cuda_allocated_gib':3.}]
        result=p.projection(history);self.assertAlmostEqual(result['mean_training_seconds_per_sample'],10/6)
        self.assertAlmostEqual(result['candidate_80_round_training_only_hours'],5824*80*10/6/3600)
        self.assertEqual(result['maximum_observed_reserved_gib'],4.);self.assertFalse(result['long_training_started'])

    def test_error_bundle_excludes_stale_success_projection_pixels_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);p.write_json(out/'result.json',{'status':'ERROR'});p.write_json(out/'budget_projection.json',{'stale':True})
            (out/'weights.pt').write_bytes(b'not included');(out/'image.png').write_bytes(b'not included')
            with zipfile.ZipFile(p.bundle(out)) as z:
                self.assertIsNone(z.testzip());self.assertNotIn('budget_projection.json',z.namelist())
                self.assertFalse(any(x.endswith(('.pt','.png')) for x in z.namelist()))


if __name__=='__main__':unittest.main()
