"""Torch CPU tests for ignore gradients, immutable data and phase recovery."""
import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from ultralytics.models.utils.loss import RTDETRDetectionLoss
    from hycado_start.crop_criterion import IgnoreAwareRTDETRLoss,ignored_queries,criterion_self_check
    from hycado_start import crop_training as ct
    TORCH=True
except ImportError:TORCH=False

from hycado_start import crop_loader as cl,crop_data as cd
from hycado_start.common import sha256,write_json
from hycado_start.stage1 import fingerprint
from test_crop_data import fixture,clean_text


@unittest.skipUnless(TORCH,'Torch/Ultralytics required for crop training tests')
class IgnoreGradientTests(unittest.TestCase):
    def test_main_aux_denoising_positive_priority_gradient(self):
        check=criterion_self_check()
        self.assertEqual(check['audit']['blocked_query_branch_instances'],6)

    def test_no_ignore_matches_stock_losses_and_gradients_with_and_without_gt(self):
        for n in [0,1]:
            torch.manual_seed(23)
            boxes=torch.rand(3,1,5,4)*.4+.2
            left=torch.randn(3,1,5,10,requires_grad=True);right=left.detach().clone().requires_grad_()
            batch={'cls':torch.tensor([3]*n,dtype=torch.long),'bboxes':torch.tensor([[.5,.5,.2,.2]]*n).reshape(-1,4),
                   'gt_groups':[n],'ignore_boxes':[torch.empty(0,4)]}
            stock=RTDETRDetectionLoss(nc=10,use_vfl=True)((boxes,left),batch)
            custom=IgnoreAwareRTDETRLoss()((boxes,right),batch)
            self.assertEqual(stock.keys(),custom.keys())
            for key in stock:torch.testing.assert_close(stock[key],custom[key],rtol=0,atol=0)
            sum(stock.values()).backward();sum(custom.values()).backward()
            torch.testing.assert_close(left.grad,right.grad,rtol=0,atol=0)

    def test_unmatched_inside_ignore_zero_outside_keeps_negative_gradient(self):
        boxes=torch.tensor([[[[.2,.2,.1,.1],[.8,.8,.1,.1]]]])
        scores=torch.zeros((1,1,2,10),requires_grad=True)
        batch={'cls':torch.empty(0,dtype=torch.long),'bboxes':torch.empty(0,4),'gt_groups':[0],
               'ignore_boxes':[torch.tensor([[0.,0.,.5,.5]])]}
        sum(IgnoreAwareRTDETRLoss()((boxes,scores),batch).values()).backward()
        self.assertEqual(scores.grad[0,0,0].abs().sum(),0)
        self.assertGreater(scores.grad[0,0,1].abs().sum(),0)

    def test_ioa_uses_prediction_area_and_batch_local_regions(self):
        boxes=torch.tensor([[[.5,.5,1.,1.],[.1,.1,.1,.1]],[[.5,.5,1.,1.],[.1,.1,.1,.1]]])
        masks=ignored_queries(boxes,[torch.tensor([[0.,0.,.5,1.]]),torch.empty(0,4)])
        self.assertEqual(masks.tolist(),[[True,True],[False,False]])
        self.assertEqual(ignored_queries(boxes[:1],[torch.tensor([[0.,0.,.49,1.]])]).tolist(),[[False,True]])

    def test_real_rtdetr_forward_backward_and_ema_style_eval_path(self):
        from ultralytics.nn.tasks import RTDETRDetectionModel
        from hycado_start.crop_criterion import forward_loss
        torch.set_num_threads(2);torch.manual_seed(42)
        # 128 provides at least 300 encoder tokens for the unmodified decoder.
        model=RTDETRDetectionModel('rtdetr-l.yaml',nc=10,verbose=False).train()
        batch={'img':torch.rand(1,3,128,128),'cls':torch.tensor([3]),
               'bboxes':torch.tensor([[.5,.5,.2,.2]]),'batch_idx':torch.tensor([0]),
               'ignore_boxes':[torch.tensor([[.1,.1,.4,.4]])]}
        criterion=IgnoreAwareRTDETRLoss();loss,_=forward_loss(model,batch,criterion)
        loss.backward();self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(criterion.audit['denoising_branch_calls'],0)
        model.eval()
        with torch.no_grad():dev,_=forward_loss(model,batch,criterion)
        self.assertTrue(torch.isfinite(dev))


def crop_fixture(root,all_ignored=False):
    stage,s6,s7,out,records,sources,cfg=fixture(root)
    if all_ignored:
        record=records[0];raw='0,0,100,100,0,0,0,0\n'
        ann=stage/'data/annotations/train/example_0.txt';ann.write_text(raw)
        lab=stage/'data/labels/train/example_0.txt';lab.write_text(clean_text(raw))
        record.update(annotation_sha256=sha256(ann),label_sha256=sha256(lab),gt_count=0)
    cfg.update(code_hashes={n:sha256(Path(cd.__file__).with_name(n)) for n in ['crop_data.py','crop_labels.py']},training_started=False)
    digest=fingerprint(cfg)
    files,_=cd.process_sources(records,stage/'data',s6,s7,out,sources,cfg,digest)
    cd.finalize_dataset(files,out,sources,cfg,digest)
    write_json(out/'protocol.json',cfg);write_json(out/'result.json',{'status':'COMPLETED','official_val_used':False})
    return out,digest


class CatalogAndSamplingTests(unittest.TestCase):
    def test_catalog_counts_and_train_dev_group_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,digest=crop_fixture(Path(tmp));groups,info=cl.load_catalog(out,digest)
            self.assertEqual(info['total_crops'],{'train':3,'val':6})
            self.assertEqual(info['eligible_source_images'],{'train':1,'val':2})
            self.assertEqual(len(cl.source_round(groups['train'],0)),1)
            index=cl.read_json(out/'source_index.json');entry=index['records'][2]
            path=out/entry['source_record'];doc=cl.read_json(path);doc['group']='0'
            doc.pop('content_sha256');doc['content_sha256']=fingerprint(doc);write_json(path,doc)
            entry['group']='0';entry['sha256']=sha256(path);write_json(out/'source_index.json',index)
            with self.assertRaisesRegex(ValueError,'sizintisi'):cl.load_catalog(out,digest)

    def test_fully_ignored_excluded_without_changing_08_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,digest=crop_fixture(Path(tmp),True);groups,info=cl.load_catalog(out,digest)
            self.assertEqual(info['fully_ignored_crops_excluded']['train'],3)
            self.assertEqual(info['eligible_source_images']['train'],0)
            self.assertEqual(info['total_crops']['train'],3)
            self.assertTrue((out/'items/train/example_0/crop_0000.png').exists())

    def test_source_balanced_cycle_visits_every_crop_before_repetition(self):
        groups=[{'source_image':'x','crops':[{'crop_index':i} for i in range(3)]},
                {'source_image':'y','crops':[{'crop_index':5}]}]
        visits=[cl.source_round(groups,r) for r in range(4)]
        # Each round includes both sources regardless of their crop counts.
        self.assertTrue(all(len(v)==2 for v in visits))
        indexes=[next(v['crop_index'] for v in rows if v['crop_index']!=5) for rows in visits]
        self.assertEqual(set(indexes[:3]),{0,1,2});self.assertEqual(indexes[0],indexes[3])
        self.assertEqual(visits,[cl.source_round(groups,r) for r in range(4)])

    def test_pilot_representative_sources_disjoint_and_development_separate(self):
        groups={r:[{'source_image':f'{r}{i}','crops':[{'source_image':f'{r}{i}','crop_index':j,
                    'positive_targets':i+j,'width':10+i,'height':20,'ignored_boxes_by_reason':{'x':j}} for j in range(2)]}
                   for i in range(20)] for r in ['train','val']}
        plan=cl.pilot_plan(groups,phases=3,representative=4,stress_count=4,development=2)
        ids=[x['source_image'] for p in plan for x in p['representative']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(plan[0]['stress'],plan[2]['stress'])
        self.assertEqual(plan[0]['development'],plan[2]['development'])
        self.assertTrue(all(x['source_image'].startswith('val') for p in plan for x in p['development']))

    def test_safe_path_rejects_escape(self):
        for bad in ['../x','/x','D:/x','items\\x']:
            with self.assertRaises(ValueError):cl.safe_path(Path('/tmp'),bad)

    @unittest.skipUnless(TORCH,'Torch required for tensor loader')
    def test_actual_png_loader_normalization_and_hash_rejection(self):
        import cv2,numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            out,digest=crop_fixture(Path(tmp));groups,_=cl.load_catalog(out,digest)
            item=cl.source_round(groups['train'],0)[0];batch=cl.load_sample(out,item,64,'cpu')
            self.assertEqual(tuple(batch['img'].shape),(1,3,64,64))
            with Image.open(out/item['image']) as im:expected=cv2.resize(np.array(im.convert('RGB')),(64,64),interpolation=cv2.INTER_LINEAR)
            np.testing.assert_array_equal((batch['img'][0].permute(1,2,0).numpy()*255).round().astype('uint8'),expected)
            self.assertEqual(tuple(batch['bboxes'].shape),(item['positive_targets'],4))
            path=out/item['image'];path.write_bytes(path.read_bytes()+b'bad')
            with self.assertRaisesRegex(ValueError,'hash'):cl.load_sample(out,item,64,'cpu')


@unittest.skipUnless(TORCH,'Torch required for optimizer/checkpoint recovery')
class EngineRecoveryTests(unittest.TestCase):
    def test_interrupted_phase_resume_matches_uninterrupted_optimizer_ema_and_rng(self):
        cfg={'seed':42,'amp':False,'imgsz':4,'effective_batch':3,'lr':.0001,'weight_decay':.0001,
             'betas':[.9,.999],'gradient_clip_norm':10.,'ema_decay':.9999,'ema_tau':2000,
             'warmup_timing_samples_per_phase':1}
        plan=[{'phase':p,'representative':[{'id':i,'supervision_state':'positive'} for i in range(6)],
               'stress':[{'id':6+i,'supervision_state':'positive'} for i in range(2)],
               'development':[{'id':i,'supervision_state':'positive'} for i in range(2)]} for p in range(1,4)]
        def model(*args):return torch.nn.Sequential(torch.nn.Linear(4,5),torch.nn.Dropout(.2),torch.nn.Linear(5,1))
        def sample(root,item,res,device):return {'img':torch.ones(1,4)*(item['id']+1)/10}
        def loss(model,batch,criterion):
            value=(model(batch['img'])-.3).square().mean();return value,{'loss':value}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);full=root/'full';resume=root/'resume';full.mkdir();resume.mkdir()
            with patch.object(ct,'build_model',side_effect=model),patch.object(ct,'load_sample',side_effect=sample),\
                    patch('hycado_start.crop_criterion.forward_loss',side_effect=loss):
                ct.train_core(root,full,None,cfg,plan,device='cpu')
                calls=0
                def stop(*args):
                    nonlocal calls
                    calls+=1
                    if calls==11:raise InterruptedError('synthetic phase 2 interruption')
                    return sample(*args)
                with patch.object(ct,'load_sample',side_effect=stop):
                    with self.assertRaises(InterruptedError):ct.train_core(root,resume,None,cfg,plan,device='cpu')
                self.assertEqual(cl.read_json(resume/'checkpoint_index.json')['latest']['epoch'],1)
                ct.train_core(root,resume,None,cfg,plan,device='cpu')
            def saved(path):
                name=cl.read_json(path/'checkpoint_index.json')['latest']['path']
                return torch.load(path/name,map_location='cpu',weights_only=False)
            a,b=saved(full),saved(resume)
            for key in ['model','ema']:
                for name in a[key]:torch.testing.assert_close(a[key][name],b[key][name],rtol=0,atol=0)
            self.assertEqual(a['ema_updates'],9);self.assertEqual(a['ema_updates'],b['ema_updates'])
            for index in a['optimizer']['state']:
                for key in a['optimizer']['state'][index]:
                    torch.testing.assert_close(a['optimizer']['state'][index][key],b['optimizer']['state'][index][key],rtol=0,atol=0)
            torch.testing.assert_close(a['rng']['torch'],b['rng']['torch'],rtol=0,atol=0)
            # A missing/stale history sidecar is reconstructed from the committed checkpoint.
            (resume/'history.json').unlink()
            ct.train_core(root,resume,None,cfg,plan,device='cpu')
            self.assertEqual(len(cl.read_json(resume/'history.json')),3)

    def test_error_bundle_excludes_stale_success_projection_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);write_json(out/'result.json',{'status':'ERROR'})
            write_json(out/'budget_projection.json',{'stale':True});(out/'bad.pt').write_bytes(b'weights')
            with zipfile.ZipFile(ct.make_bundle(out)) as z:
                self.assertNotIn('budget_projection.json',z.namelist());self.assertFalse(any(n.endswith('.pt') for n in z.namelist()))


if __name__=='__main__':unittest.main()
