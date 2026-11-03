import contextlib,copy,io,random,sys,tempfile,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from hycado_start import position_control as p


def query(cls=0):
    a=np.zeros((300,14),np.float32);a[0,:4]=[.5,.5,.5,.5];a[0,4+cls]=.8
    return a


class PositionTests(unittest.TestCase):
    def inputs(self,records,refs=None):
        return [{'source_image':x['image'],'image_sha256':x.get('image_sha256','im'),'width':x['width'],'height':x['height'],
            'reference12_sha256':refs[x['image']] if refs else 'ref','crop_rects':[[0,0,20,20],[10,10,40,40]]} for x in records]

    def config(self,records,plan):return {'records':records,'geometry_plan_sha256':p.fingerprint(plan),'code_sha256':p.sha256(p.__file__),'gpu':'fake','driver':'driver'}

    def records(self,root,n=2):
        result=[]
        for i in range(n):
            path=root/'images/val'/f'{i}.png';path.parent.mkdir(parents=True,exist_ok=True)
            yy,xx=np.mgrid[:100,:100];pixels=np.stack([xx,yy,np.full_like(xx,i)],axis=2).astype(np.uint8);Image.fromarray(pixels).save(path)
            result.append({'image':f'images/val/{i}.png','width':100,'height':100,'image_sha256':p.sha256(path)})
        return result

    def seed_doc(self,cfg,plan,seed):
        rows=[{'source_image':x['source_image'],'crops':len(x['original_rects']),'path':f'predictions/s{seed}/{Path(x["source_image"]).stem}/prediction.json',
            'sha256':'meta','raw_archive_sha256':'raw'} for x in plan['records']]
        doc={'status':'COMPLETED','position_seed':seed,'protocol_sha256':p.fingerprint(cfg),'records':rows};doc['content_sha256']=p.fingerprint(doc);return doc

    def test_uniform_rejection_bounds_and_global_rng_unchanged(self):
        state=random.getstate()
        for maximum in [0,1,2,37,10000]:
            a=[p.uniform_coordinate(maximum,{'test':i}) for i in range(100)]
            self.assertTrue(all(0<=x<=maximum for x in a));self.assertEqual(a,[p.uniform_coordinate(maximum,{'test':i}) for i in range(100)])
        self.assertEqual(random.getstate(),state)
        with patch.object(p,'fingerprint',side_effect=['f'*64,'0'*64]) as hashed:
            self.assertEqual(p.uniform_coordinate(2,'rejection'),0);self.assertEqual(hashed.call_count,2)
        for bad in [-1,1.5,True]:
            with self.assertRaises(ValueError):p.uniform_coordinate(bad,{})

    def test_relocation_exact_count_ordered_size_area_bounds_and_fullframe_case(self):
        original=[[0,0,20,30],[75,50,100,100],[0,0,100,100],[0,0,100,1]]
        results=[]
        for seed in p.SEEDS:
            new=p.relocate(original,100,100,'images/val/test.jpg',seed);results.append(new)
            self.assertEqual(new,p.relocate(original,100,100,'images/val/test.jpg',seed));self.assertEqual(len(new),len(original))
            self.assertEqual([(b[2]-b[0],b[3]-b[1]) for b in new],[(b[2]-b[0],b[3]-b[1]) for b in original])
            self.assertEqual(new[2],original[2]);self.assertTrue(all(0<=x<x2<=100 and 0<=y<y2<=100 for x,y,x2,y2 in new))
        self.assertNotEqual(results[0],results[1]);self.assertEqual(p.relocate([],100,100,'empty',0),[])
        for rects in [[[0,0,101,20]],[[0,0,0,10]],[[0.,0,20,20]]]:
            with self.assertRaises(ValueError):p.relocate(rects,100,100,'bad',0)

    def test_plan_is_predeclared_stable_and_rejects_incomplete_coverage(self):
        records=[{'image':f'images/val/{i}.png','width':100,'height':100} for i in range(2)];inputs=self.inputs(records);original=copy.deepcopy(inputs)
        plan=p.make_plan(inputs,expected=(2,4));self.assertEqual(plan,p.make_plan(inputs[::-1],expected=(2,4)));self.assertEqual(inputs,original)
        self.assertEqual(plan['location_seeds'],[0,1,2]);self.assertEqual(plan['crops_per_seed'],4);self.assertEqual(plan['total_new_crop_forwards'],12)
        for seed in p.SEEDS:
            area=sum((b[2]-b[0])*(b[3]-b[1]) for row in plan['records'] for b in row[p.variant(seed)])
            self.assertEqual(area,plan['source_pixel_area_per_seed'])
        for bad in [inputs[:1],inputs+inputs[:1]]:
            with self.assertRaises(ValueError):p.make_plan(bad,expected=(2,4))

    def test_native_RGB_crop_inference_uses_planned_pixel_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rec=self.records(root,1)[0];rects=[[10,20,30,50],[50,60,90,80]];calls=[]
            def fresh(model,rgb):calls.append(rgb);return query()
            with patch.object(p.r,'direct_queries',side_effect=fresh):raw=p.infer_image(object(),root/rec['image'],rec,rects)
            self.assertEqual(raw.shape,(2,300,14));self.assertEqual(raw.dtype,np.float32)
            self.assertEqual([x.shape for x in calls],[(30,20,3),(20,40,3)])
            self.assertEqual(calls[0][0,0].tolist(),[10,20,0]);self.assertEqual(calls[1][0,0].tolist(),[50,60,0])
            with patch.object(p.r,'direct_queries',side_effect=AssertionError('zero crops must not infer')):
                self.assertEqual(p.infer_image(None,root/rec['image'],rec,[]).shape,(0,300,14))

    def test_atomic_prediction_replay_NMS_guard_and_corrupt_cache_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(p.g.j.optimized.n,'fast_nms',side_effect=p.r.old.classwise_nms):
            root=Path(tmp);rec=self.records(root,1)[0];plan=p.make_plan(self.inputs([rec]),expected=(1,2));cfg=self.config([rec],plan);row=plan['records'][0]
            folder=root/'out/predictions/s0/0';raw=np.stack([query()]*2)
            with patch.object(p.g.j.optimized.n,'fast_nms',return_value=[]),self.assertRaises(ValueError):p.save_prediction(folder,rec,cfg,row,0,[],raw)
            self.assertFalse(folder.exists());p.save_prediction(folder,rec,cfg,row,0,[],raw)
            doc=p.checked_prediction(folder,rec,cfg,row,0,[]);self.assertEqual(doc['crop_count'],2);self.assertTrue(doc['independent_NMS_replay_exact'])
            with self.assertRaises(ValueError):p.checked_prediction(folder,rec,cfg,row,1,[])
            changed=copy.deepcopy(row);changed['reference12_sha256']='other'
            with self.assertRaises(ValueError):p.checked_prediction(folder,rec,cfg,changed,0,[])
            (folder/'queries.npz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):p.checked_prediction(folder,rec,cfg,row,0,[])
            self.assertFalse(list((root/'out/staging').iterdir()))

    def test_worker_interrupt_resume_skips_completed_images_and_lazy_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';out=root/'out';out.mkdir(parents=True);base=root/'12';base.mkdir();records=self.records(root/'data/prepared');index=[];refs={};baseline=[[60.,60.,70.,70.,.9,1]]
            for rec in records:
                path=base/(Path(rec['image']).stem+'.json');p.r.val.signed_json(path,{'protocol_sha256':p.r.P12,'source_image':rec['image'],'image_sha256':rec['image_sha256'],
                    'width':100,'height':100,'baseline':baseline,'hybrid_a025':baseline,'crop_rects':[[0,0,20,20],[10,10,40,40]]})
                refs[rec['image']]=p.sha256(path);index.append({'source_image':rec['image'],'path':path.name,'sha256':p.sha256(path),'crops':2})
            p.write_json(base/'prediction_index.json',{'records':index});plan=p.make_plan(self.inputs(records,refs),expected=(2,4));cfg=self.config(records,plan)
            p.write_json(out/'protocol.json',cfg);p.write_json(out/'geometry_plan.json',plan);job={'output':str(out),'position_seed':0,'protocol_sha256':p.fingerprint(cfg)};p.write_json(out/'job.json',job)
            attempts=[0];fail=[True];forward=[];actualinfer=p.infer_image
            def interrupted(*args):
                attempts[0]+=1
                if fail[0] and attempts[0]==2:raise InterruptedError('simulated restart')
                return actualinfer(*args)
            with patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True))}),patch.object(p,'ROOT',root),\
                patch.object(p,'source',return_value=({},base,root/'19',root/'weights',root/'official',{})),\
                patch.object(p,'system_info',return_value={'gpu':'fake','nvidia_smi':'fake,0,0,driver'}),patch.object(p.r.crop_capture,'load_model',return_value=object()) as load,\
                patch.object(p.r,'direct_queries',side_effect=lambda *a:(forward.append(1) or query())),patch.object(p,'infer_image',side_effect=interrupted),\
                patch.object(p.g.j.optimized.n,'fast_nms',side_effect=p.r.old.classwise_nms),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(p.worker(out/'job.json'),1);self.assertFalse((out/'seed_0.json').exists());saved=out/'predictions/s0/0/prediction.json';before=p.sha256(saved)
                image_backup=root.parent/'HyCADO_Yedekleri/22_Konum_Kontrolu/HyCADO_Temiz_Baslangic/out/predictions/s0/0/prediction.json'
                self.assertEqual(p.sha256(image_backup),before)
                fail[0]=False;forward.clear();self.assertEqual(p.worker(out/'job.json'),0);self.assertEqual(len(forward),2);self.assertEqual(p.sha256(saved),before)
                doc=p.read_json(out/'seed_0.json');self.assertEqual(doc['reused_images_this_invocation'],1);self.assertEqual(doc['new_images_this_invocation'],1)
                load.reset_mock();forward.clear();self.assertEqual(p.worker(out/'job.json'),0);load.assert_not_called();self.assertEqual(forward,[])
                p.validate_seed(p.read_json(out/'seed_0.json'),cfg,plan,0)

    def fixture(self,root):
        from hycado_start import dev_metrics as m
        prepared=root/'data/prepared';records=self.records(prepared);base=root/'12';base.mkdir();folder19=root/'19';folder19.mkdir();out=root/'out';refs={};index=[];prior={};images=[]
        for i,rec in enumerate(records):
            anno=prepared/'visdrone_annotations/val'/f'{i}.txt';anno.parent.mkdir(parents=True,exist_ok=True);anno.write_text(f'10,10,10,10,1,{i+1},0,0\n');rec['annotation_sha256']=p.sha256(anno)
            baseline=[[10.,10.,20.,20.,.9,i]] if i==0 else [];hybrid=[[10.,10.,20.,20.,.9,i]];prior[rec['image']]=baseline
            path=base/f'{i}.json';p.r.val.signed_json(path,{'protocol_sha256':p.r.P12,'source_image':rec['image'],'image_sha256':rec['image_sha256'],
                'width':100,'height':100,'baseline':baseline,'hybrid_a025':hybrid,'crop_rects':[[0,0,20,20],[10,10,40,40]]})
            refs[rec['image']]=p.sha256(path);index.append({'source_image':rec['image'],'path':path.name,'sha256':p.sha256(path),'crops':2})
            images.append({'gt':m.prepare_groundtruth(m.raw_annotation(anno.read_text()),None),'baseline':m.xyxy_to_det(baseline),'hybrid_a025':m.xyxy_to_det(hybrid)})
        p.write_json(base/'prediction_index.json',{'records':index});plan=p.make_plan(self.inputs(records,refs),expected=(2,4));cfg=self.config(records,plan)
        for seed in p.SEEDS:
            for i,(rec,row) in enumerate(zip(records,plan['records'])):
                folder=out/'predictions'/f's{seed}'/str(i);p.save_prediction(folder,rec,cfg,row,seed,prior[rec['image']],np.stack([query(i)]*2))
                doc=p.checked_prediction(folder,rec,cfg,row,seed,prior[rec['image']]);images[i][p.variant(seed)]=m.xyxy_to_det(doc['detections'])
            doc=self.seed_doc(cfg,plan,seed)
            for item in doc['records']:
                path=out/item['path'];item['sha256']=p.sha256(path);item['raw_archive_sha256']=p.sha256(path.parent/'queries.npz')
            doc['content_sha256']=p.fingerprint({k:v for k,v in doc.items() if k!='content_sha256'});p.write_json(out/f'seed_{seed}.json',doc)
        names=['baseline','hybrid_a025']+[p.variant(s) for s in p.SEEDS]
        with contextlib.redirect_stdout(io.StringIO()):expected={v:m.evaluate(images,v) for v in names}
        for name in names:
            expected[name]['counts_at_fixed_max500']={}
            for t,key in [(.5,'0.5'),(.75,'0.75')]:
                flags=[]
                for image in images:
                    det=image[name][:500];gt=image['gt']
                    for c in range(1,11):flags.extend(m.match_flags(det[det[:,5]==c,:5],gt[gt[:,5]==c,:5],t).tolist())
                a=np.array(flags);expected[name]['counts_at_fixed_max500'][key]={'TP':int((a==1).sum()),'FP':int((a==0).sum()),'ignored_detections':int((a==-1).sum())}
        p.write_json(folder19/'metrics.json',{'variants':expected});return out,cfg,plan,base,folder19,expected

    def test_metrics_five_conditions_match_independent_reference_and_all_seeds_guard_GT(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(p.g.j.optimized.n,'fast_nms',side_effect=p.r.old.classwise_nms):
            root=Path(tmp);out,cfg,plan,base,folder19,expected=self.fixture(root)
            with patch.object(p,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):actual=p.measure(out,cfg,plan,base,folder19)
            for name in expected:
                p.g.scoring.compare_metrics(actual['variants'][name],expected[name]);self.assertEqual(actual['variants'][name]['counts_at_fixed_max500'],expected[name]['counts_at_fixed_max500'])
            (out/'predictions/s2/1/queries.npz').write_bytes(b'bad last seed')
            with patch.object(p,'ROOT',root),patch.object(p,'raw_annotation',side_effect=AssertionError('must validate every seed before GT')),self.assertRaises(ValueError):p.measure(out,cfg,plan,base,folder19)

    def test_seed_integrity_and_summary_do_not_conflate_location_and_training_seeds(self):
        records=[{'image':f'images/val/{i}.png','width':100,'height':100} for i in range(2)];plan=p.make_plan(self.inputs(records),expected=(2,4));cfg=self.config(records,plan);doc=self.seed_doc(cfg,plan,0)
        p.validate_seed(doc,cfg,plan,0)
        for kind in ['hash','coverage','order','crops']:
            bad=copy.deepcopy(doc)
            if kind=='hash':bad['content_sha256']='bad'
            elif kind=='coverage':bad['records'].pop()
            elif kind=='order':bad['records'].reverse()
            else:bad['records'][0]['crops']=100
            if kind!='hash':bad['content_sha256']=p.fingerprint({k:v for k,v in bad.items() if k!='content_sha256'})
            with self.assertRaises(ValueError):p.validate_seed(bad,cfg,plan,0)
        keys=['AP','AP50','AP75','AR500'];metrics={'variants':{'hybrid_a025':{k:10. for k in keys},**{p.variant(s):{k:2.+s for k in keys} for s in p.SEEDS}}}
        result=p.summarize(metrics,plan);self.assertEqual(result['relocated_mean']['AP'],3.);self.assertEqual(result['relocated_sample_SD']['AP'],1.)
        self.assertEqual(result['delta_hybrid_minus_relocated_mean_pp']['AP'],7.);self.assertFalse(result['FPS_claim']);self.assertFalse(result['training_started']);self.assertFalse(result['seed_selection_by_AP'])

    def test_backups_include_new_raw_images_immediately_but_report_excludes_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';code=root/'hycado_start/position_control.py';code.parent.mkdir(parents=True);code.write_bytes(b'code')
            out=root/'experiments'/p.OUT;out.mkdir(parents=True);p.write_json(out/'result.json',{'status':'ERROR'});folder=out/'predictions/s0/0';folder.mkdir(parents=True)
            p.write_json(folder/'prediction.json',{'record':'complete'});(folder/'queries.npz').write_bytes(b'raw');(out/'model.pt').write_bytes(b'not copied')
            staging=out/'staging/unfinished';staging.mkdir(parents=True);(staging/'queries.npz').write_bytes(b'partial')
            report=p.bundle(out);p.backup(root,out);target=root.parent/'HyCADO_Yedekleri/22_Konum_Kontrolu/HyCADO_Temiz_Baslangic'
            self.assertEqual((target/f'experiments/{p.OUT}/predictions/s0/0/queries.npz').read_bytes(),b'raw');self.assertFalse(list(target.rglob('model.pt')))
            self.assertFalse(any('staging' in x.parts for x in target.rglob('*')))
            with zipfile.ZipFile(report) as z:self.assertIsNone(z.testzip());self.assertFalse(any(n.endswith(('.npz','.pt','.png')) for n in z.namelist()))

    def test_parent_native_error_and_resume_preserves_complete_seed_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';code=root/'hycado_start/position_control.py';code.parent.mkdir(parents=True);code.write_bytes(b'code')
            records=[{'image':f'images/val/{i:04}.png','width':100,'height':100} for i in range(548)]
            inputs=self.inputs(records)
            for i,x in enumerate(inputs):x['crop_rects']=[[0,0,20,20]]*(19 if i<193 else 18)
            plan=p.make_plan(inputs);c12={'records':records,'stage1_weights_sha256':'a','crop_weights_sha256':'b','fusion':{},'crop_inference':{},'runtime_versions':{}}
            names=['hybrid_a025']+[p.variant(s) for s in p.SEEDS];metrics={'variants':{v:{k:3. for k in ['AP','AP50','AP75','AR500']} for v in names}}
            out=root/'experiments'/p.OUT;calls=[];fail=[True]
            def child(command,cwd):
                job=p.read_json(command[-1]);calls.append(job)
                if fail[0] and len(calls)==2:return -1073741819
                cfg=p.read_json(out/'protocol.json');p.write_json(out/f'seed_{job["position_seed"]}.json',self.seed_doc(cfg,plan,job['position_seed']));return 0
            with patch.object(p,'ROOT',root),patch.object(p,'source',return_value=(c12,root/'12',root/'19',root/'w2',root/'official',{})),\
                patch.object(p,'build_plan',return_value=(plan,{})),patch.object(p,'system_info',return_value={'cuda_available':True,'gpu':'fake','nvidia_smi':'fake,0,0,driver'}),\
                patch.object(p.r,'child_process',side_effect=child),patch.object(p,'measure',return_value=metrics),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertFalse(p.run());self.assertEqual(p.read_json(out/'result.json')['status'],'ERROR');before=p.sha256(out/'seed_0.json')
                staging=out/'staging/junk';staging.mkdir(parents=True);(staging/'queries.npz').write_bytes(b'partial');fail[0]=False;calls.clear()
                self.assertTrue(p.run());self.assertEqual(len(calls),2);self.assertEqual(p.sha256(out/'seed_0.json'),before);self.assertFalse((out/'staging').exists())
                result=p.read_json(out/'result.json');self.assertEqual(result['reused_completed_seeds'],1);self.assertEqual(result['prediction_records'],1644)
                self.assertEqual(result['new_crop_forwards_in_completed_predictions'],30171)


if __name__=='__main__':unittest.main()
