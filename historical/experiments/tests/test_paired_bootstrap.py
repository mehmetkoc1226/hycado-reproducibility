import contextlib,copy,io,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from hycado_start import bootstrap_statistics as s,paired_bootstrap as b,dev_metrics as m


class BootstrapTests(unittest.TestCase):
    def synthetic_parts(self,n=4):
        rng=np.random.default_rng(2);parts=[]
        for _ in range(n):
            part={'gt_denominators':np.full(10,50,dtype=np.int64),'presence':np.ones(10,dtype=np.int64)}
            for name in s.NAMES:
                part[name+'_scores']=np.sort(rng.integers(1,6,40))[::-1].astype(float)/10
                part[name+'_classes']=rng.integers(1,11,40,dtype=np.int8)
                part[name+'_flags']=rng.choice(np.array([-1,0,1],dtype=np.int8),(40,10))
            parts.append(part)
        return parts

    def test_proxy_groups_union_cross_prefix_duplicate_without_dropping_records(self):
        records=[{'image':f'images/val/{stem}.jpg','image_sha256':h} for stem,h in
            [('0000001_00000_d_0000001','a'),('0000001_00001_d_0000002','b'),('0000002_00000_d_0000003','c'),('0000003_00000_d_0000004','c'),('other_image','d')]]
        groups=s.make_groups(records);self.assertEqual(groups['initial_filename_groups'],4);self.assertEqual(groups['groups_after_hash_union'],3)
        self.assertEqual(groups['image_to_group'][0],groups['image_to_group'][1]);self.assertEqual(groups['image_to_group'][2],groups['image_to_group'][3])
        self.assertEqual(len(groups['image_to_group']),5);self.assertFalse(groups['verified_video_or_scene_identity'])

    def test_frequency_draws_are_paired_deterministic_and_preserve_cluster_members(self):
        groups={'labels':['a','b','c'],'image_to_group':[0,0,0,1,2]}
        for i in range(20):
            f=s.frequencies(groups,'cluster',i);self.assertTrue(np.array_equal(f,s.frequencies(groups,'cluster',i)))
            self.assertEqual(len(set(f[:3])),1);self.assertEqual(f[0]+f[3]+f[4],3)
            self.assertEqual(s.frequencies(groups,'image',i).sum(),5)
        self.assertFalse(np.array_equal(s.frequencies(groups,'cluster',3),s.frequencies(groups,'image',3)))

    def test_tied_TP_FP_copies_preserve_literal_order_instead_of_naive_weights(self):
        part={'gt_denominators':np.array([1]+[0]*9),'presence':np.array([1]+[0]*9)}
        for name in s.NAMES:
            part[name+'_scores']=np.array([.5,.5]);part[name+'_classes']=np.ones(2,np.int8)
            part[name+'_flags']=np.tile(np.array([[1],[0]],np.int8),(1,10))
        other=copy.deepcopy(part);other['baseline_flags'][:]=0
        engine=s.compile_engine([part,other]);result=s.evaluate(engine,np.array([2,0]))
        self.assertAlmostEqual(result[0,0],100*5/6)
        self.assertEqual(s.repeated_indices(engine['classes']['baseline'][0],np.array([2,0])).tolist(),[0,1,0,1])
        literal=b.p.g.scoring.aggregate([part,part],'baseline');self.assertAlmostEqual(result[0,0],literal['AP'])

    def test_optimized_all_metrics_match_literal_repeat_with_ties_ignores_and_missing_classes(self):
        parts=self.synthetic_parts();parts[1]['gt_denominators'][5:]=0;parts[1]['presence'][5:]=0
        engine=s.compile_engine(parts)
        for freq in [np.ones(4,dtype=np.int64),np.array([0,3,1,2]),np.array([0,2,0,0])]:
            fast=s.evaluate(engine,freq);expanded=[part for part,n in zip(parts,freq) for _ in range(n)]
            literal=np.array([[b.p.g.scoring.aggregate(expanded,v)[k] for k in s.KEYS] for v in s.NAMES])
            np.testing.assert_allclose(fast,literal,atol=1e-10,rtol=0)
        self.assertTrue(np.isnan(s.evaluate(engine,np.zeros(4,dtype=np.int64))).all())

    def test_difference_intervals_use_paired_draws_and_mean_fixed_location_seeds(self):
        point=np.arange(len(s.NAMES)*4,dtype=float).reshape(len(s.NAMES),4)
        draws=np.stack([point+i for i in range(10)]);draws[-1]=np.nan
        result=s.intervals(point,draws);self.assertEqual(result['valid_replicates'],9);self.assertEqual(result['undefined_no_target_replicates'],1)
        diff=result['differences_pp']['hybrid_minus_baseline']['AP'];self.assertEqual(diff['point'],12.);self.assertEqual(diff['lower95'],12.);self.assertEqual(diff['upper95'],12.)
        self.assertEqual(result['differences_pp']['hybrid_minus_relocated_mean']['AP']['point'],-8.)
        self.assertIn('fixed trained weights',result['conditional_scope'])

    def fixture(self,root):
        records=[];rows_by_image=[];ctx={'base':root/'12','f19':root/'19','f21':root/'21','f22':root/'22',
            'c12':{'step':12},'c19':{'step':19},'c21':{'step':21},'c22':{'step':22},'m12':{},'m19':{},'m21':{},'m22':{x:{} for x in range(3)},'plan':{}}
        for i in range(3):
            image=f'images/val/{i+1:07d}_00000_d_{i:07d}.jpg';annotation=root/'data/prepared/visdrone_annotations/val'/(Path(image).stem+'.txt');annotation.parent.mkdir(parents=True,exist_ok=True)
            annotation.write_text(f'10,10,10,10,1,{i+1},0,0\n80,80,10,10,0,0,0,0\n')
            rec={'image':image,'image_sha256':'duplicate' if i<2 else 'unique','annotation_sha256':b.sha256(annotation),'width':100,'height':100};records.append(rec)
            baseline=[[10.,10.,20.,20.,.9,i]] if i==0 else [];hybrid=[[10.,10.,20.,20.,.9,i]]
            rows={'baseline':baseline,'hybrid_a025':hybrid,'rtdetr_full960':[[10.,10.,20.,20.,.8,i]],
                'fixed_grid_a025':hybrid+[[60.,60.,70.,70.,.1,i]],'relocated_s0_a025':baseline,
                'relocated_s1_a025':baseline+[[30.,30.,40.,40.,.1,i]],'relocated_s2_a025':hybrid}
            rows_by_image.append(rows)
            for key,label in [('12',None),('19','rtdetr_full960'),('21','fixed_grid_a025')]:
                path=ctx['base' if key=='12' else 'f'+key]/f'{i}.json'
                body={'protocol_sha256':b.fingerprint(ctx['c'+key]),'source_image':image,'image_sha256':rec['image_sha256'],'width':100,'height':100}
                if label:body[label]=rows[label]
                else:body.update(baseline=baseline,hybrid_a025=hybrid)
                b.r.val.signed_json(path,body);ctx['m'+key][image]={'path':path.name,'sha256':b.sha256(path)}
            plan={'source_image':image,'reference12_sha256':ctx['m12'][image]['sha256'],**{b.p.variant(s):[[0,0,100,100]] for s in range(3)}};ctx['plan'][image]=plan
            for seed in range(3):
                path=ctx['f22']/f's{seed}_{i}.json';name=b.p.variant(seed);rawhash=f'raw_{seed}_{i}'
                b.r.val.signed_json(path,{'protocol_sha256':b.fingerprint(ctx['c22']),'source_image':image,'image_sha256':rec['image_sha256'],'width':100,'height':100,
                    'position_seed':seed,'variant':name,'reference12_sha256':plan['reference12_sha256'],'plan_record_sha256':b.fingerprint(plan),
                    'crop_rects':plan[name],'raw_archive_sha256':rawhash,'detections':rows[name]})
                ctx['m22'][seed][image]={'path':path.name,'sha256':b.sha256(path),'raw_archive_sha256':rawhash}
        images=[]
        for rec,rows in zip(records,rows_by_image):
            path=root/'data/prepared/visdrone_annotations/val'/(Path(rec['image']).stem+'.txt');raw=m.raw_annotation(path.read_text());integral=m.ignore_integral(raw,100,100)
            gt=m.prepare_groundtruth(raw,integral);image={'gt':gt}
            for v in s.NAMES:image[v]=m.outside_ignore(m.xyxy_to_det(rows[v]),integral)
            images.append(image)
        with contextlib.redirect_stdout(io.StringIO()):expected={v:m.evaluate(images,v) for v in s.NAMES}
        for v in s.NAMES:
            expected[v]['counts_at_fixed_max500']={}
            for t,label in [(.5,'0.5'),(.75,'0.75')]:
                flags=[]
                for image in images:
                    det=image[v][:500];gt=image['gt']
                    for c in range(1,11):flags.extend(m.match_flags(det[det[:,5]==c,:5],gt[gt[:,5]==c,:5],t).tolist())
                f=np.array(flags);expected[v]['counts_at_fixed_max500'][label]={'TP':int((f==1).sum()),'FP':int((f==0).sum()),'ignored_detections':int((f==-1).sum())}
        ctx['expected']=expected;cfg={'records':records,'test_protocol':'fixture'}
        for folder in [root/'experiments'/b.p.g.j.OUT,root/'experiments'/b.p.g.OUT]:
            b.write_json(folder/'summary.json',{'variants':{'hybrid_a025':{k:1. for k in ['mean_ms','median_ms','p95_ms','throughput_images_per_second','peak_allocated_mib','peak_reserved_mib']}}})
        return cfg,ctx

    def test_prepare_cache_is_CPU_only_matches_independent_metric_and_resumes_without_GT_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';cfg,ctx=self.fixture(root);out=root/'out';out.mkdir()
            with patch.object(b.p.r,'direct_queries',side_effect=AssertionError('no inference')),patch.object(b.p.r.crop_capture,'load_model',side_effect=AssertionError('no model')),contextlib.redirect_stdout(io.StringIO()):
                parts,index,_=b.prepare_parts(root,out,cfg,ctx);engine=s.compile_engine(parts);groups=s.make_groups(cfg['records']);point,metrics,checks=b.check_engine(parts,engine,groups,ctx['expected'])
            self.assertEqual(len(parts),3);self.assertEqual(len(checks),3)
            for v in s.NAMES:b.p.g.scoring.compare_metrics(metrics[v],ctx['expected'][v])
            before={x['source_image']:b.sha256(out/x['path']/'matches.npz') for x in index}
            with patch.object(b,'raw_annotation',side_effect=AssertionError('must reuse')),contextlib.redirect_stdout(io.StringIO()):parts,index,_=b.prepare_parts(root,out,cfg,ctx)
            self.assertEqual(before,{x['source_image']:b.sha256(out/x['path']/'matches.npz') for x in index})
            (out/index[0]['path']/'matches.npz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):b.prepare_parts(root,out,cfg,ctx)

    def test_block_validation_catches_changed_draws_values_protocol_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'cluster_0000';groups={'labels':['a','b'],'image_to_group':[0,0,1]};cfg={'fixed':'protocol'}
            counts=np.array([s.frequencies(groups,'cluster',i,b.DRAW_SEED) for i in range(3)]);values=np.full((3,7,4),10.)
            b.save_block(folder,values,counts,cfg,'cluster',0);a,c=b.read_block(folder,cfg,'cluster',0,groups,3);np.testing.assert_array_equal(a,values)
            with self.assertRaises(ValueError):b.read_block(folder,{'changed':'protocol'},'cluster',0,groups,3)
            bad=counts.copy();bad[0,0]+=1;b.save_block(folder,values,bad,cfg,'cluster',0)
            with self.assertRaises(ValueError):b.read_block(folder,cfg,'cluster',0,groups,3)
            b.save_block(folder,np.full((3,7,4),101.),counts,cfg,'cluster',0)
            with self.assertRaises(ValueError):b.read_block(folder,cfg,'cluster',0,groups,3)

    def test_full_parent_resume_keeps_blocks_and_source_files_and_exports_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';c12,ctx=self.fixture(root);c12.update(runtime_versions={});out=root/'experiments'/b.OUT
            immutable={str(root/'data/prepared/visdrone_annotations/val'/f'{Path(x["image"]).stem}.txt'):x['annotation_sha256'] for x in c12['records']}
            actual=b.save_block;interrupted=[True]
            def save(*args):
                if interrupted[0] and args[-1]==3:raise InterruptedError('restart at second bootstrap block')
                return actual(*args)
            with patch.object(b,'ROOT',root),patch.object(b,'source',return_value=(c12,ctx,immutable)),patch.object(b,'B',7),patch.object(b,'BLOCK',3),\
                patch.object(b,'save_block',side_effect=save),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertFalse(b.run());self.assertEqual(b.read_json(out/'result.json')['status'],'ERROR');before=b.sha256(out/'bootstrap/cluster_0000.npz');interrupted[0]=False
                with patch.object(b,'raw_annotation',side_effect=AssertionError('completed match cache reused')):self.assertTrue(b.run())
                self.assertEqual(before,b.sha256(out/'bootstrap/cluster_0000.npz'));result=b.read_json(out/'result.json')
                self.assertEqual(result['status'],'COMPLETED');self.assertFalse(result['GPU_inference']);self.assertEqual(result['proxy_clusters'],2)
                summary=b.read_json(out/'summary.json');self.assertEqual(summary['intervals']['cluster']['replicates_requested'],7)
                self.assertTrue((out/'results_summary.md').exists());self.assertTrue((out/'historical_runtime_sessions.csv').exists())
                for path,digest in immutable.items():self.assertEqual(b.sha256(path),digest)
                with zipfile.ZipFile(out/'23_esli_bootstrap_rapor.zip') as z:self.assertIsNone(z.testzip());self.assertTrue(any(x.endswith('matches.npz') for x in z.namelist()));self.assertFalse(any(x.endswith('.pt') for x in z.namelist()))


if __name__=='__main__':unittest.main()
