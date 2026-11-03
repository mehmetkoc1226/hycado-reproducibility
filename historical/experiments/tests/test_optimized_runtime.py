import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from hycado_start import optimized_runtime as o

DECISION = {'proposal_generation':{'threshold':.05,'top_k':60,'context_scale':3.},
            'merge':{'requested_crop_count':10,'policy':{'name':'near_z3','max_gap_ratio':.5,
                       'max_axis_growth':3.,'max_pair_hull_ratio':2.}}}


class OptimizedTests(unittest.TestCase):
    def fixture(self, root, count=2):
        base = root/'reference'; base.mkdir(); records=[]; index=[]
        baseline=[[5.,5.,15.,15.,.8,0]]
        query=np.zeros((300,14),dtype=np.float32); query[0,:4]=[.5,.5,.5,.5]; query[0,4]=.9
        rows=o.r.old.map_queries(query,{'source_crop_xyxy':[0,0,25,25]})
        for row in rows: row[4]*=.25
        hybrid=o.r.old.classwise_nms(baseline+rows,64,48)
        for i in range(count):
            path=root/'data/prepared/images/val'/f'{i}.png';path.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(64,48)).save(path)
            rec={'image':f'images/val/{i}.png','width':64,'height':48,'image_sha256':o.sha256(path)};records.append(rec)
            folder=base/str(i);folder.mkdir()
            np.savez_compressed(folder/'raw_crop_queries.npz',raw_queries=query[None],crop_rects=np.array([[0,0,25,25]],dtype=np.int32))
            archive=o.sha256(folder/'raw_crop_queries.npz')
            o.r.val.signed_json(folder/'prediction.json',{'protocol_sha256':o.r.P12,'source_image':rec['image'],
              'image_sha256':rec['image_sha256'],'width':64,'height':48,'baseline':baseline,'hybrid_a025':hybrid,
              'crop_rects':[[0,0,25,25]],'crop_count':1,'raw_archive_sha256':archive})
            index.append({'source_image':rec['image'],'path':f'{i}/prediction.json','sha256':o.sha256(folder/'prediction.json'),
                          'crops':1,'raw_archive_sha256':archive})
        o.write_json(base/'prediction_index.json',{'records':index})
        return base,records,index,baseline,query

    def test_integrated_pipeline_same_pixels_scores_and_order_as_13(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base,recs,index,baseline,q=self.fixture(root,1);calls=[]
            def predict(rgb): calls.append(rgb.copy());return q.copy()
            args=(recs[0],root/'data/prepared'/recs[0]['image'],lambda p:(64,48,baseline,0),predict,DECISION,True,lambda:None)
            a=o.r.run_image(*args);b=o.run_image(*args)
            self.assertEqual(a[1:],b[1:]);np.testing.assert_array_equal(calls[0],calls[1]);self.assertEqual(b[0]['crop_count'],1)
            self.assertLessEqual(sum(b[0][k] for k in o.r.COMPONENTS if k!='total_ms'),b[0]['total_ms']+.01)
            self.assertEqual(o.run_image(*args[:5],False,lambda:None)[2],baseline)

    def test_profile_split_preserves_model_input_and_raw_query_on_CPU(self):
        import torch
        inputs=[]
        class Model:
            def predict(self,x):
                inputs.append(x.numpy().copy());v=torch.zeros((1,300,14));v[0,0,:3]=x[0,:,25,40];return v
        rgb=np.random.default_rng(4).integers(0,256,(37,49,3),dtype=np.uint8)
        original=o.r.direct_queries(Model(),rgb,device='cpu')
        actual,detail=o.profile_queries(Model(),rgb,device='cpu')
        np.testing.assert_array_equal(inputs[0],inputs[1]);np.testing.assert_array_equal(actual,original)
        self.assertEqual(set(detail),{'resize_ms','transfer_normalize_ms','forward_ms','query_to_cpu_ms'})
        self.assertTrue(all(x>=0 for x in detail.values()))

    def test_profile_geometry_mapping_and_strict_raw_final_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base,recs,index,baseline,q=self.fixture(root,1)
            expected=o.read_json(base/'0/prediction.json');path=root/'data/prepared'/recs[0]['image']
            predict=lambda rgb:(q.copy(),{k:0. for k in ['resize_ms','transfer_normalize_ms','forward_ms','query_to_cpu_ms']})
            row=o.profile_image(path,expected,q[None],predict)
            self.assertTrue(row['raw_queries_exact12']);self.assertTrue(row['final_output_exact12']);self.assertEqual(row['crop_count'],1)
            self.assertLessEqual(sum(row[k] for k in o.PROFILE_COMPONENTS),row['diagnostic_wall_ms']+.02)
            bad=q.copy();bad[0,4]=.8
            with self.assertRaisesRegex(ValueError,'ham sorgusu'):o.profile_image(path,expected,bad[None],predict)
            expected['hybrid_a025']=[]
            with self.assertRaisesRegex(ValueError,'Tani ciktisi'):o.profile_image(path,expected,q[None],predict)

    def test_profile_selection_coverage_hash_timing_and_count_validation(self):
        cfg={'records':[{'image':str(i)} for i in range(40)],'profile_expected_crops':32}
        self.assertEqual(o.profile_order(cfg['records']),o.profile_order(cfg['records'][::-1]))
        self.assertEqual(len(o.profile_order(cfg['records'])),32)
        rows=[{'source_image':x['image'],**{k:1. for k in o.PROFILE_COMPONENTS},'diagnostic_wall_ms':9.,
               'crop_count':1,'raw_queries_exact12':True,'final_output_exact12':True} for x in o.profile_order(cfg['records'])]
        p={'status':'COMPLETED','protocol_sha256':o.fingerprint(cfg),'rows':rows};p['content_sha256']=o.fingerprint(p)
        o.validate_profile(p,cfg)
        for kind in ['hash','missing','order','negative','nan','components','crops','equality']:
            bad=copy.deepcopy(p)
            if kind=='hash':bad['rows'][0]['extract_ms']=2
            elif kind=='missing':bad['rows'].pop()
            elif kind=='order':bad['rows'].reverse()
            elif kind=='negative':bad['rows'][0]['extract_ms']=-1
            elif kind=='nan':bad['rows'][0]['extract_ms']=float('nan')
            elif kind=='components':bad['rows'][0]['forward_ms']=10
            elif kind=='crops':bad['rows'][0]['crop_count']=2
            else:bad['rows'][0]['raw_queries_exact12']=False
            if kind!='hash':bad['content_sha256']=o.fingerprint({k:v for k,v in bad.items() if k!='content_sha256'})
            with self.assertRaises(ValueError):o.validate_profile(bad,cfg)

    def test_diagnostic_times_excluded_from_benchmark_summary(self):
        phases=[]
        for p,v in o.r.PHASES:
            rows=[{'source_image':str(i),'total_ms':10. if v=='baseline' else 30.,'stage1_ms':10.,
                  'geometry_ms':0.,'crop_pipeline_ms':15. if v!='baseline' else 0.,'merge_ms':5. if v!='baseline' else 0.,
                  'crop_count':1 if v!='baseline' else 0,'peak_allocated_mib':200.,'peak_reserved_mib':300.} for i in range(2)]
            phases.append({'pass_id':p,'variant':v,'rows':rows})
        profile={'rows':[{k:1000000. for k in o.PROFILE_COMPONENTS}|{'crop_count':1} for i in range(32)]}
        s=o.summary(phases,profile,{'variants':{'hybrid_a025':{'mean_ms':40.}}})
        self.assertEqual(s['variants']['hybrid_a025']['mean_ms'],30.)
        self.assertEqual(s['hybrid_to_baseline_latency_ratio'],3.)
        self.assertEqual(s['crop_diagnostics']['images'],32)
        self.assertTrue(s['crop_diagnostics']['excluded_from_end_to_end_statistics'])
        self.assertAlmostEqual(sum(s['crop_diagnostics']['share_of_summed_components_percent'].values()),100)
        self.assertAlmostEqual(s['historical13_context']['old_to_new_ratio'],40/30)

    def test_worker_rejects_mismatch_then_commits_valid_benchmark_and_profile(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir();base,recs,index,baseline,q=self.fixture(root,2)
            weights=[]
            for name in ['w1','w2','official']:
                path=root/name;path.write_bytes(name.encode());weights.append(path)
            cfg={'records':recs,'warmup_images':1,'expected_crop_count':2,'profile_expected_crops':2,
                 'code_sha256':o.sha256(o.__file__),'upstream_metadata_sha256':{},'gpu':'fake','driver':'driver'}
            o.write_json(out/'protocol.json',cfg)
            cfg12={'stage1_inference':{},'crop_policy':DECISION,'stage1_weights_sha256':o.sha256(weights[0]),'crop_weights_sha256':o.sha256(weights[1])}
            bad=[True]
            def stage1(path):return (64,48,[],0) if bad[0] and path.stem=='1' else (64,48,baseline,0)
            with contextlib.ExitStack() as stack:
                for name,value in [('is_available',True),('synchronize',None),('reset_peak_memory_stats',None),
                                   ('max_memory_allocated',200*1024**2),('max_memory_reserved',300*1024**2)]:
                    stack.enter_context(patch.object(torch.cuda,name,return_value=value))
                for obj,name,value in [(o,'ROOT',root),(o,'sources',(cfg12,base,*weights,{})),
                    (o,'system_info',{'gpu':'fake','nvidia_smi':'fake, 0, 0, driver'}),
                    (o.r.proposals,'load_ema',object()),(o.r.proposals,'make_predictor',stage1),
                    (o.r.crop_capture,'load_model',object()),(o.r,'direct_queries',q),
                    (o.r,'backend_state',{}),(o,'profile_queries',(q,{k:0. for k in ['resize_ms','transfer_normalize_ms','forward_ms','query_to_cpu_ms']}))]:
                    if name=='ROOT': stack.enter_context(patch.object(obj,name,value))
                    else:stack.enter_context(patch.object(obj,name,return_value=value))
                stack.enter_context(patch.object(o.r,'restore_backend'))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()));stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                job={'kind':'benchmark','output':str(out),'protocol_sha256':o.fingerprint(cfg),'pass_id':0,'variant':'hybrid_a025'}
                o.write_json(out/'job.json',job)
                self.assertEqual(o.worker(out/'job.json'),1);self.assertFalse((out/'pass_1_hybrid_a025.json').exists())
                bad[0]=False;self.assertEqual(o.worker(out/'job.json'),0)
                o.r.validate_phase(o.read_json(out/'pass_1_hybrid_a025.json'),cfg,0,'hybrid_a025')
                job['kind']='profile';o.write_json(out/'job.json',job)
                self.assertEqual(o.worker(out/'job.json'),0)
                o.validate_profile(o.read_json(out/'crop_profile.json'),cfg)
                archive=base/'1/raw_crop_queries.npz';archive.write_bytes(b'corrupt')
                (out/'crop_profile.json').unlink()
                self.assertEqual(o.worker(out/'job.json'),1);self.assertFalse((out/'crop_profile.json').exists())

    def test_small_backup_only_new_small_files_and_bundle_atomic(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';code=root/'hycado_start/optimized_runtime.py';code.parent.mkdir(parents=True);code.write_bytes(b'v1')
            other=root/'data/huge';other.parent.mkdir();other.write_bytes(b'must not copy')
            out=root/'experiments/15';out.mkdir(parents=True);o.write_json(out/'result.json',{'status':'ERROR'})
            report=o.bundle(out);target=Path(o.small_backup(root,out,True))
            next(target.rglob('optimized_runtime.py')).write_bytes(b'corrupt')
            o.small_backup(root,out)
            self.assertEqual(next(target.rglob('optimized_runtime.py')).read_bytes(),b'v1')
            code.write_bytes(b'v2');o.small_backup(root,out)
            self.assertEqual({x.read_bytes() for x in target.rglob('optimized_runtime.py')},{b'v1',b'v2'})
            self.assertFalse(list(target.rglob('huge')))
            with zipfile.ZipFile(report) as z:self.assertIsNone(z.testzip());self.assertEqual(z.namelist(),['result.json'])
            self.assertFalse(list(out.glob('*.tmp')))

    def test_orchestrator_rejects_negative_native_exit_and_resumes_whole_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';code=root/'hycado_start/optimized_runtime.py';code.parent.mkdir(parents=True);code.write_bytes(b'code')
            base=root/'experiments'/o.r.val.OUT;base.mkdir(parents=True)
            recs=[{'image':f'images/val/{i}.jpg'} for i in range(548)]
            index=[{'source_image':x['image'],'crops':19 if i<193 else 18} for i,x in enumerate(recs)]
            crop_counts={x['source_image']:x['crops'] for x in index}
            o.write_json(base/'prediction_index.json',{'records':index})
            for name in o.r.FILES12:
                if name!='prediction_index.json':o.write_json(base/name,{'fixed':name})
            hist=root/'experiments'/o.r.OUT;hist.mkdir();o.write_json(hist/'summary.json',{'variants':{'hybrid_a025':{'mean_ms':60.}}})
            weights=[]
            for name in ['w1','w2','official']:
                path=root/name;path.write_bytes(name.encode());weights.append(path)
            cfg12={'records':recs,'runtime_versions':{}}
            output=root/'experiments'/o.OUT;calls=[];stop=[True]
            def child(command,cwd):
                job=o.read_json(Path(command[-1]));calls.append(job)
                if stop[0] and len(calls)==3:return -1073741819
                cfg=o.read_json(output/'protocol.json');diag=job['kind']=='profile';rows=[]
                order=o.profile_order(recs) if diag else o.r.pass_order(recs,job['pass_id'])
                for rec in order:
                    if diag:
                        row={k:1. for k in o.PROFILE_COMPONENTS};row.update(diagnostic_wall_ms=9.,crop_count=crop_counts[rec['image']],raw_queries_exact12=True,final_output_exact12=True)
                    else:
                        hybrid=job['variant']=='hybrid_a025'
                        row={'stage1_ms':10.,'geometry_ms':0.,'crop_pipeline_ms':20. if hybrid else 0.,'merge_ms':5. if hybrid else 0.,
                             'total_ms':35. if hybrid else 10.,'crop_count':crop_counts[rec['image']] if hybrid else 0,
                             'peak_allocated_mib':200.,'peak_reserved_mib':300.,'exact_original12_output':True}
                    rows.append({'source_image':rec['image'],**row})
                body={'status':'COMPLETED','protocol_sha256':o.fingerprint(cfg),'rows':rows}
                if not diag:body.update(pass_id=job['pass_id'],variant=job['variant'])
                body['content_sha256']=o.fingerprint(body)
                filename='crop_profile.json' if diag else f'pass_{job["pass_id"]+1}_{job["variant"]}.json'
                o.write_json(output/filename,body);return 0
            with patch.object(o,'ROOT',root),patch.object(o,'sources',return_value=(cfg12,base,*weights,{})),\
                 patch.object(o,'system_info',return_value={'cuda_available':True,'gpu':'fake','nvidia_smi':'fake, 0, 0, driver'}),\
                 patch.object(o.r,'child_process',side_effect=child),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertFalse(o.run());self.assertEqual(o.read_json(output/'result.json')['status'],'ERROR')
                self.assertIn('-1073741819',o.read_json(output/'result.json')['error'])
                committed={x.name:o.sha256(x) for x in output.glob('pass_*.json')};self.assertEqual(len(committed),2)
                stop[0]=False;calls.clear();self.assertTrue(o.run());self.assertEqual(len(calls),5)
                self.assertEqual({name:o.sha256(output/name) for name in committed},committed)
                result=o.read_json(output/'result.json');self.assertEqual(result['measured_image_calls'],3288)
                calls.clear();self.assertTrue(o.run());self.assertEqual(calls,[])
            self.assertEqual(o.read_json(output/'summary.json')['variants']['hybrid_a025']['crops_per_pass'],10057)


if __name__=='__main__':unittest.main()
