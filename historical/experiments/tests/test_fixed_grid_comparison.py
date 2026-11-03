import contextlib,copy,io,sys,tempfile,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from hycado_start import fixed_grid_comparison as g


def query(cls=0):
    a=np.zeros((300,14),np.float32);a[0,:4]=[.15,.15,.1,.1];a[0,4+cls]=.8
    return a


class GridTests(unittest.TestCase):
    def config(self,records):
        return {'records':records,'calibration':{'selected_side_px':80},'hybrid_crop_counts':{x['image']:2 for x in records},
            'code_sha256':g.sha256(g.__file__),'warmup_images':1,'gpu':'fake','driver':'driver'}

    def records(self,root,n=2):
        records=[]
        for i in range(n):
            path=root/'images/val'/f'{i}.png';path.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(100,100),(i,42,90)).save(path)
            records.append({'image':f'images/val/{i}.png','width':100,'height':100,'image_sha256':g.sha256(path)})
        return records

    def phase(self,cfg,p,v):
        rows=[]
        for rec in g.r.pass_order(cfg['records'],p):
            rows.append({'source_image':rec['image'],'total_ms':20. if v==g.GRID else 40.,'peak_allocated_mib':100.,'peak_reserved_mib':200.,
                'crop_count':len(g.grid_rects(rec['width'],rec['height'],cfg['calibration']['selected_side_px'])) if v==g.GRID else cfg['hybrid_crop_counts'][rec['image']],
                'exact_reference_output':True})
        doc={'status':'COMPLETED','pass_id':p,'variant':v,'protocol_sha256':g.fingerprint(cfg),'rows':rows};doc['content_sha256']=g.fingerprint(doc)
        return doc

    def test_grid_covers_every_pixel_without_duplicate_tiles_including_small_and_edge_images(self):
        for width,height,side in [(1,1,80),(79,203,80),(100,100,80),(321,197,64),(640,480,256),(1920,1080,448)]:
            rects=g.grid_rects(width,height,side);self.assertEqual(len(rects),len({tuple(r) for r in rects}))
            coverage=np.zeros((height,width),np.uint8)
            for x,y,x2,y2 in rects:
                self.assertTrue(0<=x<x2<=width and 0<=y<y2<=height)
                self.assertEqual((x2-x,y2-y),(min(width,side),min(height,side)));coverage[y:y2,x:x2]=1
            self.assertTrue(coverage.all());self.assertEqual(rects[0][:2],[0,0]);self.assertEqual(rects[-1][2:],[width,height])
        for bad in [(0,80),(5,0),(5.5,80),(5,True)]:
            with self.assertRaises(ValueError):g.starts(*bad)

    def test_development_selection_matches_budget_and_rejects_VAL_not_using_annotations(self):
        records=[{'source_image':f'images/train/{i}.jpg','width':800,'height':600,'crop_count':12} for i in range(2)]
        cfg=g.calibrate(records,expected=(2,24));stats=cfg['candidate_statistics'];best=min(stats,key=lambda x:(x['relative_count_error'],-x['side_px']))
        self.assertEqual(cfg['selected_side_px'],best['side_px']);self.assertFalse(cfg['VAL_geometry_or_metrics_used_for_selection']);self.assertFalse(cfg['GT_rows_parsed'])
        self.assertEqual(cfg,g.calibrate(records[::-1],expected=(2,24)))
        for kind in ['VAL','count','duplicate']:
            bad=copy.deepcopy(records)
            if kind=='VAL':bad[0]['source_image']='images/val/0.jpg'
            elif kind=='count':bad[0]['crop_count']=100
            else:bad[0]['source_image']=bad[1]['source_image']
            with self.assertRaises(ValueError):g.calibrate(bad,expected=(2,24))
        # Pure calibration sees no files, annotations, model, VAL dimensions or AP.
        self.assertTrue(all('AP' not in x for x in stats))

    def test_fresh_grid_pipeline_mapping_scaling_and_timing_boundary(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(g.j.optimized.n,'fast_nms',side_effect=g.r.old.classwise_nms):
            root=Path(tmp);rec=self.records(root,1)[0];calls=[]
            def predict1(path):calls.append('stage1');return 100,100,[[60.,60.,70.,70.,.9,1]],0
            def predict2(rgb):calls.append(('crop',rgb.shape));self.assertEqual(rgb.shape,(80,80,3));return query()
            times,baseline,result,rects,raw=g.run_grid(rec,root/rec['image'],predict1,predict2,80,lambda:calls.append('sync'),before_crops=lambda:calls.append('backend'))
            self.assertEqual(len([x for x in calls if isinstance(x,tuple)]),4);self.assertEqual(raw.shape,(4,300,14));self.assertGreater(times['total_ms'],0)
            self.assertEqual(calls[0],'sync');self.assertEqual(calls[1],'stage1');self.assertEqual(times['crop_count'],4)
            self.assertEqual(result,g.fuse(raw,rects,baseline,100,100,g.r.old.classwise_nms))
            self.assertEqual(result[0],baseline[0]);self.assertAlmostEqual(result[1][4],float(query()[0,4])*.25)
            np.testing.assert_allclose(result[1][:4],[8,8,16,16],atol=1e-5)

    def test_atomic_raw_cache_replay_corruption_and_independent_NMS_guard(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(g.j.optimized.n,'fast_nms',side_effect=g.r.old.classwise_nms):
            root=Path(tmp);rec=self.records(root,1)[0];cfg=self.config([rec]);rects=g.grid_rects(100,100,80);raw=np.stack([query()]*4)
            rows=g.fuse(raw,rects,[],100,100,g.r.old.classwise_nms);folder=root/'predictions/0'
            with self.assertRaises(ValueError):g.save_prediction(folder,rec,cfg,[],[],rects,raw)
            self.assertFalse(folder.exists());g.save_prediction(folder,rec,cfg,[],rows,rects,raw)
            self.assertEqual(g.checked_prediction(folder,rec,cfg)[g.GRID],rows)
            changed=copy.deepcopy(cfg);changed['calibration']['selected_side_px']=64
            with self.assertRaises(ValueError):g.checked_prediction(folder,rec,changed)
            (folder/'queries.npz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):g.checked_prediction(folder,rec,cfg)
            self.assertFalse(list((root/'staging').iterdir()))

    def test_phase_rejects_invalid_data_and_balanced_runtime_summary(self):
        cfg=self.config([{'image':str(i),'width':100,'height':100} for i in range(2)]);good=self.phase(cfg,0,g.GRID)
        g.validate_phase(good,cfg,0,g.GRID)
        for kind in ['hash','coverage','order','crop','exact','time','memory']:
            doc=copy.deepcopy(good)
            if kind=='hash':doc['content_sha256']='bad'
            elif kind=='coverage':doc['rows'].pop()
            elif kind=='order':doc['rows'].reverse()
            elif kind=='crop':doc['rows'][0]['crop_count']=1
            elif kind=='exact':doc['rows'][0]['exact_reference_output']=False
            elif kind=='time':doc['rows'][0]['total_ms']=-1
            elif kind=='memory':doc['rows'][0]['peak_reserved_mib']=1
            if kind!='hash':doc['content_sha256']=g.fingerprint({k:v for k,v in doc.items() if k!='content_sha256'})
            with self.assertRaises(ValueError):g.validate_phase(doc,cfg,0,g.GRID)
        phases=[self.phase(cfg,p,v) for p,v in g.PHASES];metrics={'variants':{v:{'AP':3.,'AP50':4.} for v in [g.GRID,g.HYBRID]}}
        result=g.summarize(phases,metrics,cfg)
        self.assertEqual(result['variants'][g.GRID]['throughput_images_per_second'],50.)
        self.assertEqual(result['variants'][g.HYBRID]['throughput_images_per_second'],25.)
        self.assertTrue(all(x['hybrid_to_grid_latency_ratio']==2 for x in result['paired_passes']))
        for v in [g.GRID,g.HYBRID]:self.assertEqual(sorted(i%2 for i,(_,name) in enumerate(g.PHASES) if name==v),[0,1])

    def test_worker_fresh_GPU_replay_after_interruption_and_exact_frozen_hybrid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'out';output.mkdir();base=root/'12';base.mkdir();records=self.records(root/'data/prepared');cfg=self.config(records)
            g.write_json(output/'protocol.json',cfg);baseline=[[60.,60.,70.,70.,.9,1]];rects=[[0,0,80,80],[20,20,100,100]];index=[];forward=[]
            for rec in records:
                path=base/(Path(rec['image']).stem+'.json');g.r.val.signed_json(path,{'protocol_sha256':g.r.P12,'source_image':rec['image'],
                    'image_sha256':rec['image_sha256'],'width':100,'height':100,'baseline':baseline,g.HYBRID:baseline,'crop_rects':rects})
                index.append({'source_image':rec['image'],'path':path.name,'sha256':g.sha256(path)})
            g.write_json(base/'prediction_index.json',{'records':index});cuda=SimpleNamespace(is_available=lambda:True,synchronize=lambda:None,
                reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:100*1024**2,max_memory_reserved=lambda:200*1024**2)
            def predict1(path):return 100,100,baseline,0
            def hybrid(*a,**kw):forward.append('hybrid');return {'total_ms':40.,'crop_count':2},baseline,baseline,rects
            actualsave=g.save_prediction;fail=[True];saved=[0]
            def save(*args):
                saved[0]+=1
                if fail[0] and saved[0]==2:raise InterruptedError('simulated reboot before second commit')
                return actualsave(*args)
            with patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=cuda)}),patch.object(g,'ROOT',root),\
                patch.object(g,'source',return_value=({'stage1_inference':{},'crop_policy':{}},base,root/'19',root/'w1',root/'w2',root/'official',{})),\
                patch.object(g,'system_info',return_value={'gpu':'fake','nvidia_smi':'fake,0,0,driver'}),\
                patch.object(g.r.proposals,'load_ema'),patch.object(g.r.proposals,'make_predictor',return_value=predict1),patch.object(g.r.crop_capture,'load_model'),\
                patch.object(g.r,'backend_state',return_value={}),patch.object(g.r,'restore_backend'),patch.object(g.j.optimized,'run_image',side_effect=hybrid),\
                patch.object(g.r,'direct_queries',side_effect=lambda *a:(forward.append('grid') or query())),\
                patch.object(g.j.optimized.n,'fast_nms',side_effect=g.r.old.classwise_nms),patch.object(g,'save_prediction',side_effect=save),\
                contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                job={'output':str(output),'pass_id':0,'variant':g.GRID,'protocol_sha256':g.fingerprint(cfg)};g.write_json(output/'job.json',job)
                self.assertEqual(g.worker(output/'job.json'),1);self.assertFalse((output/f'pass_1_{g.GRID}.json').exists())
                complete=list((output/'predictions').glob('*/prediction.json'));self.assertEqual(len(complete),1);oldhash=g.sha256(complete[0]);forward.clear();fail[0]=False
                self.assertEqual(g.worker(output/'job.json'),0);self.assertEqual(g.sha256(complete[0]),oldhash)
                self.assertEqual(forward.count('grid'),12)  # 1 warmup + both measured images, four forwards each; never cached speed
                for p,v in [(0,g.HYBRID),(1,g.GRID),(1,g.HYBRID)]:
                    job.update(pass_id=p,variant=v);g.write_json(output/'job.json',job);self.assertEqual(g.worker(output/'job.json'),0)
                    g.validate_phase(g.read_json(output/f'pass_{p+1}_{v}.json'),cfg,p,v)

    def metric_fixture(self,root):
        from hycado_start import dev_metrics as m
        records=self.records(root/'data/prepared');output=root/'out';base=root/'12';folder19=root/'19';base.mkdir();folder19.mkdir()
        for i,rec in enumerate(records):
            anno=root/'data/prepared/visdrone_annotations/val'/f'{i}.txt';anno.parent.mkdir(parents=True,exist_ok=True);anno.write_text(f'10,10,10,10,1,{i+1},0,0\n')
            rec['annotation_sha256']=g.sha256(anno)
        cfg=self.config(records);cfg['calibration']['selected_side_px']=100
        index=[];images=[]
        for i,rec in enumerate(records):
            anno=root/'data/prepared/visdrone_annotations/val'/f'{i}.txt';anno.parent.mkdir(parents=True,exist_ok=True);anno.write_text(f'10,10,10,10,1,{i+1},0,0\n')
            baseline=[[10.,10.,20.,20.,.9,i]] if i==0 else [];hybrid=[[10.,10.,20.,20.,.9,i]]
            path=base/f'{i}.json';g.r.val.signed_json(path,{'protocol_sha256':g.r.P12,'source_image':rec['image'],'image_sha256':rec['image_sha256'],
                'width':100,'height':100,'baseline':baseline,g.HYBRID:hybrid})
            index.append({'source_image':rec['image'],'path':path.name,'sha256':g.sha256(path)})
            raw=query(i)[None];rects=[[0,0,100,100]];grid=g.fuse(raw,rects,baseline,100,100,g.r.old.classwise_nms)
            g.save_prediction(output/'predictions'/str(i),rec,cfg,baseline,grid,rects,raw)
            images.append({'gt':m.prepare_groundtruth(m.raw_annotation(anno.read_text()),None),'baseline':m.xyxy_to_det(baseline),g.HYBRID:m.xyxy_to_det(hybrid),g.GRID:m.xyxy_to_det(grid)})
        g.write_json(base/'prediction_index.json',{'records':index})
        with contextlib.redirect_stdout(io.StringIO()):expected={v:m.evaluate(images,v) for v in ['baseline',g.HYBRID,g.GRID]}
        for v,tp in [('baseline',1),(g.HYBRID,2),(g.GRID,2)]:
            expected[v]['counts_at_fixed_max500']={t:{'TP':tp,'FP':0,'ignored_detections':0} for t in ['0.5','0.75']}
        g.write_json(folder19/'metrics.json',{'variants':{**expected,'rtdetr_full960':{}}})
        return records,output,base,folder19,cfg,expected

    def test_same_metric_matches_independent_reference_and_checks_all_predictions_before_GT(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(g.j.optimized.n,'fast_nms',side_effect=g.r.old.classwise_nms):
            root=Path(tmp);records,out,base,folder19,cfg,expected=self.metric_fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):actual=g.measure(records,root/'data/prepared',base,out,cfg,folder19)
            for v in ['baseline',g.HYBRID,g.GRID]:
                g.scoring.compare_metrics(actual['variants'][v],expected[v]);self.assertEqual(actual['variants'][v]['counts_at_fixed_max500'],expected[v]['counts_at_fixed_max500'])
            self.assertAlmostEqual(actual['variants'][g.GRID]['AP'],100.);self.assertAlmostEqual(actual['variants']['baseline']['AP'],50.)
            (out/'predictions/1/queries.npz').write_bytes(b'corrupted before GT')
            with patch.object(g,'raw_annotation',side_effect=AssertionError('GT must not be parsed')),self.assertRaises(ValueError):g.measure(records,root/'data/prepared',base,out,cfg,folder19)

    def test_report_is_small_backup_includes_new_raw_but_not_models_or_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';code=root/'hycado_start/fixed_grid_comparison.py';code.parent.mkdir(parents=True);code.write_bytes(b'code')
            out=root/'experiments'/g.OUT;out.mkdir(parents=True);g.write_json(out/'result.json',{'status':'ERROR'})
            folder=out/'predictions/0';folder.mkdir(parents=True);g.write_json(folder/'prediction.json',{'record':'complete'});(folder/'queries.npz').write_bytes(b'raw')
            (out/'model.pt').write_bytes(b'not a backup target');staging=out/'staging/unfinished';staging.mkdir(parents=True);(staging/'queries.npz').write_bytes(b'partial')
            report=g.bundle(out);g.backup(root,out)
            target=root.parent/'HyCADO_Yedekleri/21_Sabit_Dilimleme/HyCADO_Temiz_Baslangic'
            self.assertEqual((target/f'experiments/{g.OUT}/predictions/0/queries.npz').read_bytes(),b'raw')
            self.assertFalse(list(target.rglob('model.pt')));self.assertFalse(any('staging' in p.parts for p in target.rglob('*')))
            with zipfile.ZipFile(report) as z:self.assertIsNone(z.testzip());self.assertFalse(any(n.endswith(('.npz','.pt')) for n in z.namelist()))

    def test_parent_native_exit_error_resume_and_staging_cleanup_preserve_complete_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';code=root/'hycado_start/fixed_grid_comparison.py';code.parent.mkdir(parents=True);code.write_bytes(b'code')
            base=root/'12';base.mkdir();records=[{'image':str(i),'width':100,'height':100} for i in range(548)]
            g.write_json(base/'prediction_index.json',{'records':[{'source_image':x['image'],'crops':2} for x in records]})
            c12={'records':records,'runtime_versions':{},'stage1_weights_sha256':'a','crop_weights_sha256':'b','fusion':{},'crop_inference':{}}
            metrics={'variants':{v:{'AP':3.,'AP50':4.} for v in [g.GRID,g.HYBRID]}};output=root/'experiments'/g.OUT;calls=[];fail=[True]
            def child(command,cwd):
                job=g.read_json(command[-1]);calls.append(job)
                if fail[0] and len(calls)==2:return -1073741819
                cfg=g.read_json(output/'protocol.json');p,v=job['pass_id'],job['variant'];g.write_json(output/f'pass_{p+1}_{v}.json',self.phase(cfg,p,v));return 0
            with patch.object(g,'ROOT',root),patch.object(g,'source',return_value=(c12,base,root/'19',root/'w1',root/'w2',root/'official',{})),\
                patch.object(g,'development_geometry',return_value=({'selected_side_px':80,'selected_relative_count_error':.03},{})),\
                patch.object(g,'system_info',return_value={'cuda_available':True,'gpu':'fake','nvidia_smi':'fake,0,0,driver'}),\
                patch.object(g.r,'child_process',side_effect=child),patch.object(g,'measure',return_value=metrics),\
                contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertFalse(g.run());self.assertEqual(g.read_json(output/'result.json')['status'],'ERROR')
                before={x.name:g.sha256(x) for x in output.glob('pass_*.json')};self.assertEqual(len(before),1)
                staging=output/'staging/junk';staging.mkdir(parents=True);(staging/'partial.npz').write_bytes(b'partial')
                fail[0]=False;calls.clear();self.assertTrue(g.run());self.assertEqual(len(calls),3)
                self.assertEqual(before,{name:g.sha256(output/name) for name in before});self.assertFalse((output/'staging').exists())
                result=g.read_json(output/'result.json');self.assertEqual(result['reused_completed_phases'],1);self.assertEqual(result['measured_image_calls'],2192)


if __name__=='__main__':unittest.main()
