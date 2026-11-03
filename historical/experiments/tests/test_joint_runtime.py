import contextlib,copy,io,sys,tempfile,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from hycado_start import joint_runtime as j


class JointTests(unittest.TestCase):
    def config(self):return {'records':[{'image':'a'},{'image':'b'}],'crop_counts':{'a':2,'b':1}}
    def phase(self,cfg,p,v):
        rows=[{'source_image':r['image'],'total_ms':{'baseline':10.,'rtdetr_full960':20.,'hybrid_a025':40.}[v],
            'crop_count':cfg['crop_counts'][r['image']] if v=='hybrid_a025' else 0,
            'peak_allocated_mib':100.,'peak_reserved_mib':200.,'exact_frozen_output':True} for r in j.r.pass_order(cfg['records'],p)]
        body={'status':'COMPLETED','pass_id':p,'variant':v,'protocol_sha256':j.fingerprint(cfg),'rows':rows};body['content_sha256']=j.fingerprint(body);return body

    def test_balanced_positions_and_same_image_orders(self):
        for v in j.VARIANTS:self.assertEqual(sorted([i%3 for i,(_,n) in enumerate(j.PHASES) if n==v]),[0,1,2])
        self.assertEqual(len(set(j.PHASES)),9)
        recs=[{'image':str(i)} for i in range(100)]
        self.assertEqual(j.r.pass_order(recs,0),j.r.pass_order(recs[::-1],0))

    def test_phase_rejects_incomplete_corrupt_output_timing_memory_and_crop_counts(self):
        cfg=self.config();good=self.phase(cfg,0,'hybrid_a025');j.validate_phase(good,cfg,0,'hybrid_a025')
        for kind in ['hash','missing','order','time','memory','output','crop']:
            p=copy.deepcopy(good)
            if kind=='missing':p['rows'].pop()
            elif kind=='order':p['rows'].reverse()
            elif kind=='time':p['rows'][0]['total_ms']=-1
            elif kind=='memory':p['rows'][0]['peak_reserved_mib']=1
            elif kind=='output':p['rows'][0]['exact_frozen_output']=False
            else:p['rows'][0]['crop_count']=50
            if kind!='hash':p['content_sha256']=j.fingerprint({k:v for k,v in p.items() if k!='content_sha256'})
            with self.assertRaises(ValueError):j.validate_phase(p,cfg,0,'hybrid_a025')

    def test_summary_three_way_throughput_ratios_and_frozen_AP(self):
        cfg=self.config();phases=[self.phase(cfg,p,v) for p,v in j.PHASES]
        metrics={'variants':{v:{k:3. for k in ['AP','AP50','AP75','AR500']} for v in j.VARIANTS}}
        s=j.summarize(phases,metrics)
        self.assertEqual(s['variants']['baseline']['throughput_images_per_second'],100.)
        self.assertEqual(s['variants']['rtdetr_full960']['throughput_images_per_second'],50.)
        self.assertEqual(s['variants']['hybrid_a025']['throughput_images_per_second'],25.)
        self.assertEqual(s['variants']['hybrid_a025']['measured_image_calls'],6)
        self.assertTrue(all(x['hybrid_to_YOLO_ratio']==4 and x['hybrid_to_RTDETR_ratio']==2 for x in s['paired_passes']))
        with self.assertRaises(ValueError):j.summarize(phases[:-1],metrics)

    def test_full_timing_includes_path_and_decoder_without_reference_cache(self):
        q=np.zeros((300,14),np.float32);q[0,:4]=[.5,.5,.2,.2];q[0,4]=.9;calls=[]
        with patch.object(j.e,'predict_path',side_effect=lambda *args:(calls.append('new inference') or q)):
            times,result=j.run_full({'width':100,'height':100},Path('input.jpg'),object(),lambda:calls.append('sync'))
        self.assertEqual(calls,['sync','new inference','sync']);self.assertEqual(times['crop_count'],0)
        self.assertGreater(times['total_ms'],0);self.assertEqual(len(result),1)

    def test_worker_all_three_methods_and_failure_does_not_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'out';output.mkdir();base=root/'12';base.mkdir();folder=root/'19';folder.mkdir()
            j.write_json(folder/'protocol.json',{'reference':'19'});signature=j.fingerprint({'reference':'19'})
            records=[];oldindex=[];newindex=[];baseline=[[1.,1.,5.,5.,.8,0]];rects=[[0,0,5,5],[2,2,8,8]]
            q=np.zeros((300,14),np.float32);q[0,:4]=[.5,.5,.2,.2];q[0,4]=.9
            for i in range(2):
                path=root/'data/prepared/images/val'/f'{i}.png';path.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(100,100)).save(path)
                rec={'image':f'images/val/{i}.png','width':100,'height':100,'image_sha256':j.sha256(path)};records.append(rec)
                old=base/f'{i}.json';j.r.val.signed_json(old,{'protocol_sha256':j.r.P12,'source_image':rec['image'],'image_sha256':rec['image_sha256'],
                    'width':100,'height':100,'baseline':baseline,'hybrid_a025':baseline,'crop_rects':rects})
                oldindex.append({'source_image':rec['image'],'path':old.name,'sha256':j.sha256(old)})
                dest=folder/f'{i}';j.e.save_prediction(dest,q,rec,signature)
                newindex.append({'source_image':rec['image'],'path':f'{i}/prediction.json','sha256':j.sha256(dest/'prediction.json')})
            j.write_json(base/'prediction_index.json',{'records':oldindex});j.write_json(folder/'prediction_index.json',{'records':newindex})
            cfg={'records':records,'crop_counts':{r['image']:2 for r in records},'warmup_images':1,'gpu':'fake','driver':'driver','code_sha256':j.sha256(j.__file__)}
            j.write_json(output/'protocol.json',cfg);bad=[True]
            def pipeline(*args,**kwargs):
                hybrid=args[5];result=[] if bad[0] else baseline
                return {'total_ms':10.,'crop_count':2 if hybrid else 0},baseline,result,rects if hybrid else []
            cuda=SimpleNamespace(is_available=lambda:True,synchronize=lambda:None,reset_peak_memory_stats=lambda:None,
                max_memory_allocated=lambda:100*1024**2,max_memory_reserved=lambda:200*1024**2)
            with patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=cuda)}),patch.object(j,'ROOT',root),\
                patch.object(j,'source',return_value=({'stage1_inference':{},'crop_policy':{}},base,folder,*[root/x for x in ['w1','w2','full','official']],{})),\
                patch.object(j,'system_info',return_value={'gpu':'fake','nvidia_smi':'fake,0,0,driver'}),\
                patch.object(j.r.proposals,'load_ema',return_value=object()),patch.object(j.r.proposals,'make_predictor',return_value=object()),\
                patch.object(j.r.crop_capture,'load_model',return_value=object()),patch.object(j.e,'load_model',return_value=object()),\
                patch.object(j.e,'predict_path',return_value=q),patch.object(j.r,'backend_state',return_value={}),patch.object(j.r,'restore_backend'),\
                patch.object(j.optimized,'run_image',side_effect=pipeline),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                job={'output':str(output),'protocol_sha256':j.fingerprint(cfg),'pass_id':0,'variant':'baseline'};j.write_json(output/'job.json',job)
                self.assertEqual(j.worker(output/'job.json'),1);self.assertFalse((output/'pass_1_baseline.json').exists())
                bad[0]=False
                for v in j.VARIANTS:
                    job['variant']=v;j.write_json(output/'job.json',job);self.assertEqual(j.worker(output/'job.json'),0)
                    j.validate_phase(j.read_json(output/f'pass_1_{v}.json'),cfg,0,v)

    def test_backup_and_atomic_report_only_small_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';code=root/'hycado_start/joint_runtime.py';code.parent.mkdir(parents=True);code.write_bytes(b'code')
            output=root/'out';output.mkdir();j.write_json(output/'result.json',{'status':'ERROR'});(output/'model.pt').write_bytes(b'not copied')
            report=j.bundle(output);j.backup(root,output)
            with zipfile.ZipFile(report) as z:self.assertIsNone(z.testzip());self.assertEqual(z.namelist(),['result.json'])
            self.assertFalse(list((root.parent/'HyCADO_Yedekleri').rglob('model.pt')))

    def test_parent_native_failure_and_resume_preserve_completed_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';code=root/'hycado_start/joint_runtime.py';code.parent.mkdir(parents=True);code.write_bytes(b'code')
            base=root/'12';base.mkdir();folder=root/'19';folder.mkdir()
            records=[{'image':str(i)} for i in range(548)]
            j.write_json(base/'prediction_index.json',{'records':[{'source_image':x['image'],'crops':19 if i<193 else 18} for i,x in enumerate(records)]})
            j.write_json(folder/'metrics.json',{'variants':{v:{k:3. for k in ['AP','AP50','AP75','AR500']} for v in j.VARIANTS}})
            cfg12={'records':records,'runtime_versions':{},'stage1_weights_sha256':'w1','crop_weights_sha256':'w2'}
            output=root/'experiments'/j.OUT;calls=[];fail=[True]
            def child(command,cwd):
                job=j.read_json(command[-1]);calls.append(job)
                if fail[0] and len(calls)==3:return -1073741819
                cfg=j.read_json(output/'protocol.json');p,v=job['pass_id'],job['variant']
                j.write_json(output/f'pass_{p+1}_{v}.json',self.phase(cfg,p,v));return 0
            with patch.object(j,'ROOT',root),patch.object(j,'source',return_value=(cfg12,base,folder,root/'w1',root/'w2',root/'full',root/'official',{})),\
                patch.object(j,'system_info',return_value={'cuda_available':True,'gpu':'fake','nvidia_smi':'fake,0,0,driver'}),\
                patch.object(j.r,'child_process',side_effect=child),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertFalse(j.run());self.assertEqual(j.read_json(output/'result.json')['status'],'ERROR')
                before={p.name:j.sha256(p) for p in output.glob('pass_*.json')};self.assertEqual(len(before),2)
                fail[0]=False;calls.clear();self.assertTrue(j.run());self.assertEqual(len(calls),7)
                self.assertEqual({name:j.sha256(output/name) for name in before},before)
                self.assertEqual(j.read_json(output/'result.json')['reused_completed_phases'],2)


if __name__=='__main__':unittest.main()
