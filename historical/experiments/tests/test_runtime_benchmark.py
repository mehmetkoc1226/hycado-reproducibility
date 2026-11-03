import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from hycado_start import runtime_benchmark as r
from hycado_start import crop_capture as c


DECISION = {'proposal_generation': {'threshold': .05, 'top_k': 60, 'context_scale': 3.},
            'merge': {'requested_crop_count': 10, 'policy': {'name': 'near_z3', 'max_gap_ratio': .5,
                     'max_axis_growth': 3., 'max_pair_hull_ratio': 2.}}}


class RuntimeTests(unittest.TestCase):
    def test_in_memory_preprocess_and_predictions_exact_old_PNG_route(self):
        import torch
        rng = np.random.default_rng(42)
        full = Image.fromarray(rng.integers(0, 256, (61, 79, 3), dtype=np.uint8)); rect = (3, 7, 49, 53)
        rgb = np.array(full.crop(rect)); inputs = []
        class Model:
            def predict(self, x):
                inputs.append(x.detach().cpu().numpy().copy())
                out = torch.zeros((1, 300, 14)); out[0, 0, :3] = x[0, :, 31, 55]
                return out
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'crop.png'; full.crop(rect).save(p, compress_level=1)
            old = c.predict_queries(Model(), p, {'width':46, 'height':46, 'image_sha256':r.sha256(p)}, device='cpu')
            new = r.direct_queries(Model(), rgb, device='cpu')
        np.testing.assert_array_equal(inputs[0], inputs[1]); np.testing.assert_array_equal(old, new)

    def test_pipeline_geometry_scores_and_no_prediction_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'image.png'; Image.fromarray(np.zeros((48,64,3), dtype=np.uint8)).save(path)
            record = {'width':64, 'height':48}; calls = []; sync = []
            baseline = [[5.,5.,15.,15.,.8,0]]
            def one(p): calls.append('stage1'); return 64,48,baseline,0
            def two(rgb):
                calls.append('crop'); self.assertEqual(rgb.shape, (25,25,3))
                a=np.zeros((300,14),dtype=np.float32); a[0,:4]=[.5,.5,.5,.5]; a[0,4]=.9; return a
            times,b,hybrid,rects = r.run_image(record,path,one,two,DECISION,True,lambda:sync.append(1))
            self.assertEqual(calls,['stage1','crop']); self.assertEqual(rects,[[0,0,25,25]])
            self.assertEqual(b,baseline); self.assertEqual(hybrid[0],baseline[0])
            self.assertAlmostEqual(hybrid[1][4],float(np.float32(.9))*.25)
            self.assertEqual(times['crop_count'],1); self.assertGreaterEqual(times['total_ms'],sum(times[k] for k in r.COMPONENTS if k!='total_ms'))
            self.assertGreaterEqual(len(sync),3)

    def test_zero_crop_and_baseline_skip_crop_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'image.png'; Image.new('RGB',(64,48)).save(path)
            record={'width':64,'height':48}; one=lambda p:(64,48,[],0)
            def forbidden(rgb): raise AssertionError('Crop model must not be called')
            for hybrid in [False,True]:
                times,base,result,rects=r.run_image(record,path,one,forbidden,DECISION,hybrid,lambda:None)
                self.assertEqual(result,[]); self.assertEqual(rects,[]); self.assertEqual(times['crop_count'],0)

    def phase(self,config,pass_id,name):
        rows=[]
        for rec in r.pass_order(config['records'],pass_id):
            rows.append({'source_image':rec['image'],'stage1_ms':10.,'geometry_ms':0.,'crop_pipeline_ms':20. if name=='hybrid_a025' else 0.,
                         'merge_ms':0.,'total_ms':30. if name=='hybrid_a025' else 10., 'crop_count':1 if name=='hybrid_a025' else 0,
                         'peak_allocated_mib':200. if name=='hybrid_a025' else 100.,'peak_reserved_mib':300.,'exact_original12_output':True})
        body={'status':'COMPLETED','protocol_sha256':r.fingerprint(config),'pass_id':pass_id,'variant':name,'rows':rows}
        body['content_sha256']=r.fingerprint(body); return body

    def test_phase_resume_rejects_incomplete_or_changed_signature_and_time(self):
        cfg={'records':[{'image':'a'},{'image':'b'}], 'expected_crop_count':2}
        good=self.phase(cfg,0,'hybrid_a025'); r.validate_phase(good,cfg,0,'hybrid_a025')
        for changed in ['missing','hash','time','method']:
            import copy
            p=copy.deepcopy(good)
            if changed=='missing': p['rows'].pop()
            elif changed=='hash': p['rows'][0]['total_ms']=11
            elif changed=='time': p['rows'][0]['total_ms']=-1
            else: p['rows'][0]['exact_original12_output']=False
            if changed!='hash':p['content_sha256']=r.fingerprint({k:v for k,v in p.items() if k!='content_sha256'})
            with self.assertRaises(ValueError): r.validate_phase(p,cfg,0,'hybrid_a025')

    def test_summary_uses_total_time_for_throughput_and_three_paired_passes(self):
        cfg={'records':[{'image':'a'},{'image':'b'}], 'expected_crop_count':2}
        phases=[self.phase(cfg,p,n) for p,n in r.PHASES]
        s=r.summarize(phases)
        self.assertEqual(s['variants']['baseline']['mean_ms'],10.)
        self.assertAlmostEqual(s['variants']['baseline']['throughput_images_per_second'],100.)
        self.assertAlmostEqual(s['variants']['hybrid_a025']['throughput_images_per_second'],1000/30)
        self.assertEqual(s['hybrid_to_baseline_latency_ratio'],3.)
        self.assertEqual(len(s['paired_passes']),3)
        self.assertTrue(all(p['hybrid_to_baseline_latency_ratio']==3. for p in s['paired_passes']))

    def test_paired_order_is_complete_deterministic_and_changes_by_pass(self):
        rows=[{'image':str(i)} for i in range(30)]
        self.assertEqual(r.pass_order(rows,0),r.pass_order(rows[::-1],0))
        self.assertEqual(len({x['image'] for x in r.pass_order(rows,0)}),30)
        self.assertNotEqual(r.pass_order(rows,0),r.pass_order(rows,1))

    def test_small_backup_copies_only_new_code_report_and_retains_previous_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project'; code=root/'hycado_start/runtime_benchmark.py';code.parent.mkdir(parents=True);code.write_bytes(b'v1')
            data=root/'data/huge';data.parent.mkdir();data.write_bytes(b'not copied')
            out=root/'experiments/runtime';out.mkdir(parents=True);r.write_json(out/'result.json',{'status':'ERROR'})
            target=Path(r.small_backup(root,out,report=True));code.write_bytes(b'v2');r.small_backup(root,out,report=True)
            copies=list(target.rglob('runtime_benchmark.py'))
            self.assertEqual(len(copies),2);self.assertEqual({p.read_bytes() for p in copies},{b'v1',b'v2'})
            self.assertFalse(list(target.rglob('huge')))

    def test_worker_stdout_is_streamed_and_nonzero_exit_returned(self):
        with contextlib.redirect_stdout(io.StringIO()) as log:
            code=r.child_process([sys.executable,'-u','-c',"print('progress',flush=True);raise SystemExit(3)"],Path.cwd())
        self.assertEqual(code,3);self.assertIn('progress',log.getvalue())

    def test_mocked_GPU_worker_rejects_changed_output_then_commits_full_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir();base=root/'reference';base.mkdir()
            records=[];index=[];baseline=[[5.,5.,15.,15.,.8,0]]
            query=np.zeros((300,14),dtype=np.float32);query[0,:4]=[.5,.5,.5,.5];query[0,4]=.9
            rows=r.old.map_queries(query,{'source_crop_xyxy':[0,0,25,25]})
            for row in rows:row[4]*=.25
            hybrid=r.old.classwise_nms(baseline+rows,64,48)
            for i in range(2):
                path=root/'data/prepared/images/val'/f'{i}.png';path.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(64,48)).save(path)
                rec={'image':f'images/val/{i}.png','width':64,'height':48,'image_sha256':r.sha256(path)};records.append(rec)
                ref=base/f'{i}.json'
                r.val.signed_json(ref,{'protocol_sha256':r.P12,'source_image':rec['image'],'image_sha256':rec['image_sha256'],
                         'width':64,'height':48,'baseline':baseline,'hybrid_a025':hybrid,'crop_rects':[[0,0,25,25]]})
                index.append({'source_image':rec['image'],'path':ref.name,'sha256':r.sha256(ref)})
            r.write_json(base/'prediction_index.json',{'records':index})
            weights=[]
            for n in ['w1','w2','official']:
                p=root/n;p.write_bytes(n.encode());weights.append(p)
            config={'records':records,'warmup_images':1,'expected_crop_count':2,'code_sha256':r.sha256(r.__file__),'gpu':'fake','driver':'fakeD'}
            r.write_json(out/'protocol.json',config)
            job={'output':str(out),'pass_id':0,'variant':'hybrid_a025','protocol_sha256':r.fingerprint(config)};r.write_json(out/'job.json',job)
            cfg12={'stage1_inference':{},'crop_policy':DECISION,'stage1_weights_sha256':r.sha256(weights[0]),'crop_weights_sha256':r.sha256(weights[1])}
            cuda=SimpleNamespace(is_available=lambda:True,synchronize=lambda:None,reset_peak_memory_stats=lambda:None,
                                 max_memory_allocated=lambda:200*1024**2,max_memory_reserved=lambda:300*1024**2)
            bad=[True]
            def stage1(path):
                if bad[0] and path.stem=='1':return 64,48,[],0
                return 64,48,baseline,0
            with patch.dict('sys.modules',{'torch':SimpleNamespace(cuda=cuda)}),patch.object(r,'ROOT',root),\
                 patch.object(r,'source',return_value=(cfg12,base,*weights)),\
                 patch.object(r,'system_info',return_value={'gpu':'fake','nvidia_smi':'fake, 0, 0, fakeD'}),\
                 patch.object(r.proposals,'load_ema',return_value=object()),patch.object(r.proposals,'make_predictor',return_value=stage1),\
                 patch.object(r.crop_capture,'load_model',return_value=object()),patch.object(r,'direct_queries',return_value=query),\
                 patch.object(r,'backend_state',return_value={}),patch.object(r,'restore_backend'),\
                 contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(r.worker(out/'job.json'),1)
                self.assertFalse((out/'pass_1_hybrid_a025.json').exists())
                self.assertTrue((out/'phase_error.json').exists())
                bad[0]=False;self.assertEqual(r.worker(out/'job.json'),0)
            phase=r.read_json(out/'pass_1_hybrid_a025.json');r.validate_phase(phase,config,0,'hybrid_a025')
            self.assertEqual(len(phase['rows']),2)


if __name__=='__main__':unittest.main()
