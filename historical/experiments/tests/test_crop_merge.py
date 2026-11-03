"""Geometry invariants, provenance, and CPU-only resume/report integration."""
import copy
import importlib.metadata
import json
import random
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from hycado_start import crop_merge as cm
from hycado_start import merge_geometry as mg
from hycado_start.common import sha256,write_json
from hycado_start.stage1 import fingerprint
from hycado_start.proposal_geometry import canonical_detections,select_regions,coverage_counts,union_area
from hycado_start.proposals import prediction_path


class MergeGeometryTests(unittest.TestCase):
    def test_overlap_merge_and_all_original_members_enclosed(self):
        result=mg.merge_to_budgets([(0,0,10,10),(5,0,15,10)],mg.POLICIES[1],[1])[1]
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['rect'],(0,0,15,10))
        self.assertEqual(result[0]['members'],(0,1))

    def test_global_zoom_bound_prevents_repeated_local_chain_growth(self):
        regions=[(0,0,10,10),(8,0,18,10),(16,0,26,10)]
        result=mg.merge_to_budgets(regions,mg.POLICIES[1],[1])[1]
        self.assertEqual(len(result),2)
        self.assertTrue(all(c['rect'][2]-c['rect'][0]<=20 for c in result))

    def test_unmet_budget_keeps_all_regions(self):
        regions=[(0,0,10,10),(100,100,110,110),(200,200,210,210)]
        result=mg.merge_to_budgets(regions,mg.POLICIES[2],[1])[1]
        self.assertEqual(len(result),3)
        self.assertEqual(sorted(m for c in result for m in c['members']),[0,1,2])

    def test_nearby_gap_is_allowed_only_by_declared_policy(self):
        regions=[(0,0,10,10),(12,0,22,10)]
        strict=mg.merge_to_budgets(regions,mg.POLICIES[0],[1])[1]
        loose=mg.merge_to_budgets(regions,mg.POLICIES[2],[1])[1]
        self.assertEqual(len(strict),2);self.assertEqual(len(loose),1)

    def test_l_shaped_hull_waste_limit(self):
        a,b=mg.initial_clusters([(0,0,100,10),(90,0,100,100)])
        policy={'max_axis_growth':20,'max_gap_ratio':1,'max_pair_hull_ratio':1.5}
        self.assertIsNone(mg.pair_priority(a,b,policy))

    def test_stable_under_input_permutation(self):
        regions=[(0,0,10,10),(5,0,15,10),(10,0,20,10),(30,0,40,10)]
        a=mg.merge_to_budgets(regions,mg.POLICIES[2],[1,2,3])
        b=mg.merge_to_budgets(list(reversed(regions)),mg.POLICIES[2],[1,2,3])
        self.assertEqual(a,b)

    def test_empty_and_invalid_policy(self):
        self.assertEqual(mg.merge_to_budgets([],mg.POLICIES[0],[5]),{5:[]})
        with self.assertRaises(ValueError):mg.merge_to_budgets([],mg.POLICIES[0],[0])
        with self.assertRaises(ValueError):mg.merge_to_budgets([(1,1,1,5)],mg.POLICIES[0],[1])

    def test_random_memberships_bounds_and_coverage_with_independent_checks(self):
        rng=random.Random(73)
        for _ in range(30):
            regions=list({(x,y,x+w,y+h) for x,y,w,h in
                         [(rng.randrange(60),rng.randrange(60),rng.randrange(4,20),rng.randrange(4,20)) for _ in range(18)]})
            originals=sorted(regions)
            for policy in mg.POLICIES:
                for target,result in mg.merge_to_budgets(regions,policy,[3,8,12]).items():
                    self.assertEqual(sorted(m for c in result for m in c['members']),list(range(len(originals))))
                    for c in result:
                        r=c['rect']
                        for m in c['members']:
                            old=originals[m]
                            self.assertTrue(r[0]<=old[0] and r[1]<=old[1] and r[2]>=old[2] and r[3]>=old[3])
                            self.assertLessEqual((r[2]-r[0])/(old[2]-old[0]),policy['max_axis_growth']+1e-12)
                            self.assertLessEqual((r[3]-r[1])/(old[3]-old[1]),policy['max_axis_growth']+1e-12)

    def test_projected_square_fill_geometry(self):
        eligible,side=mg.projected_views([[5,5,15,15,0]],[(0,0,20,40)],960)
        self.assertTrue(eligible[0]);self.assertAlmostEqual(side[0],240)
        _,bigger=mg.projected_views([[5,5,15,15,0]],[(0,0,40,80)],960)
        self.assertAlmostEqual(bigger[0],120)

    def test_coverage_can_stay_while_small_object_pixel_view_is_lost(self):
        gt=[[1,1,1.1,1.1,0]]
        base_eligible,base_side=mg.projected_views(gt,[(0,0,10,10)],960)
        counts=mg.view_counts(gt,[(0,0,20,10)],base_eligible,base_side,960)
        self.assertEqual(counts['small']['ioa90'],1)
        self.assertEqual(counts['small']['visible_min8px'],0)
        self.assertEqual(counts['small_lost_min8px'],1)
        self.assertAlmostEqual(counts['small_view_ratios_vs_baseline'][0],.5)

    def test_two_partial_views_do_not_pass_ioa90(self):
        eligible,side=mg.projected_views([[0,0,10,10,0]],[(0,0,5,10),(5,0,10,10)])
        self.assertFalse(eligible[0]);self.assertEqual(side[0],0)


def fixture(root):
    from PIL import Image
    stage1=root/'stage1';source=root/'source';output=root/'output'
    output.mkdir();records=[];index={}
    p6={'test':'synthetic06'}
    for number,split in enumerate(['train','val','val']):
        stem=f'example_{number}'
        image=f'images/{split}/{stem}.jpg'
        file=stage1/'data'/image;file.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(200,100),(40+number*30,60,80)).save(file)
        label=stage1/'data/labels'/split/(stem+'.txt');label.parent.mkdir(parents=True,exist_ok=True)
        label.write_text('4 .10 .20 .05 .10\n0 .425 .55 .015 .03\n')
        record={'source_image':f'images/train/{stem}.jpg','image':image,'source_split':'train',
                'split':split,'image_sha256':sha256(file),'label_sha256':sha256(label),'gt_count':2}
        records.append(record)
        rows,_=canonical_detections([[10,10,30,30,.9,4],[20,10,40,30,.8,4],[80,50,90,60,.5,0]],200,100)
        payload={'protocol_sha256':fingerprint(p6),'source_image':record['source_image'],'image_sha256':record['image_sha256'],
                 'split':split,'width':200,'height':100,'detections':rows,'dropped_zero_area':0}
        payload['content_sha256']=fingerprint(payload)
        cache=prediction_path(source,record);write_json(cache,payload)
        index[record['source_image']]={'sha256':sha256(cache)}
    configuration={'context_scales':[2.,3.],'threshold':.05,'top_k':60,'policies':copy.deepcopy(mg.POLICIES),
                   'budgets':[5,10,20],'projected_input_side':960}
    references=[]
    from hycado_start.proposal_geometry import load_targets
    for scale in [2.,3.]:
        counts={'all':{},'small_native_area_lt_1024':{},'classes':{str(i):{} for i in range(10)}}
        nc=[];unions=[]
        for record in records[1:]:
            payload=json.loads(prediction_path(source,record).read_text())
            gt=load_targets((stage1/'data/labels/val'/(Path(record['image']).stem+'.txt')).read_text(),200,100)
            regions=select_regions(payload['detections'],200,100,.05,60,scale)
            c=coverage_counts(gt,regions)
            for key in ['all','small_native_area_lt_1024']:cm.add_counts(counts[key],c[key])
            for cls in counts['classes']:cm.add_counts(counts['classes'][cls],c['classes'][cls])
            nc.append(len(regions));unions.append(union_area(regions)/20000)
        references.append({'threshold':.05,'top_k':60,'context_scale':scale,'counts':counts,
                           'mean_crops':sum(nc)/2,'mean_union_fraction':sum(unions)/2})
    sources={'p6':p6,'index':index,'coverage':{'configurations':references},'manifest':{'records':records}}
    return stage1,source,output,records,sources,configuration


class MergePipelineTests(unittest.TestCase):
    def test_source_metadata_gate_without_loading_weights_or_image_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage1=root/'stage1';source=root/'source'
            records=[]
            for i in range(6471):
                split='train' if i<5824 else 'val'
                records.append({'image':f'images/{split}/{i:07d}.jpg','source_image':f'images/train/{i:07d}.jpg',
                                'split':split,'source_split':'train','group':str(i),'image_sha256':str(i)})
            manifest={'records':records,'official_val_used_for_training_or_selection':False}
            write_json(stage1/'data/manifest.json',manifest)
            p1={'protocol_id':'hycado_stage1_development_v1','epochs':80,'seed':42,
                'dataset_manifest_sha256':sha256(stage1/'data/manifest.json'),
                'code_hashes':{name:sha256(Path(cm.__file__).with_name(name)) for name in
                               ['common.py','stage1.py','data.py','stability.py','experiment_data.py']}}
            checkpoint={'epoch':80,'path':'weights/epoch_0080.pt','sha256':'synthetic-only'}
            p6={'protocol_id':'hycado_stage1_predictions_coverage_v1','weights_used':'ema',
                'source_stage1_protocol_sha256':fingerprint(p1),'checkpoint':checkpoint,
                'official_val_used':False,'dataset_manifest_sha256':p1['dataset_manifest_sha256'],
                'code_hashes':{name:sha256(Path(cm.__file__).with_name(name)) for name in ['proposals.py','proposal_geometry.py']},
                'runtime_versions':{name:importlib.metadata.version(name) for name in ['numpy','Pillow']}}
            write_json(stage1/'protocol.json',p1);write_json(source/'protocol.json',p6)
            write_json(stage1/'result.json',{'status':'COMPLETED','completed_epochs':80})
            write_json(source/'result.json',{'status':'COMPLETED','predictions':{'images':6471},
                                          'coverage':{'development_images':647},'official_val_used':False})
            write_json(stage1/'checkpoint_index.json',{'protocol_sha256':fingerprint(p1),'latest':checkpoint})
            index={'protocol_sha256':fingerprint(p6),'records':[{'source_image':r['source_image'],'split':r['split'],
                     'path':prediction_path(source,r).relative_to(source).as_posix(),'sha256':'synthetic-only'} for r in records]}
            write_json(source/'prediction_index.json',index)
            write_json(source/'coverage_summary.json',{'protocol_sha256':fingerprint(p6),'official_val_used':False})
            passed=cm.validate_sources(stage1,source)
            self.assertEqual(len(passed['index']),6471)
            self.assertFalse((stage1/'weights').exists())
            index['records'][-1]['split']='train';write_json(source/'prediction_index.json',index)
            with self.assertRaisesRegex(ValueError,'rolleri'):cm.validate_sources(stage1,source)

    def test_import_does_not_import_torch_or_ultralytics(self):
        code="import sys; import hycado_start.crop_merge; assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_full_run_route_and_failure_report_preserve_prior_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage1,source,out,records,sources,cfg=fixture(Path(tmp))
            cfg.update(preview_count=2,source_checkpoint={'epoch':80,'sha256':'synthetic'})
            with patch.object(cm,'validate_sources',return_value=sources),patch.object(cm,'configuration_for',return_value=cfg),\
                    patch.object(cm,'system_info',return_value={'synthetic_cpu_harness':True}) as environment:
                self.assertTrue(cm.run(stage1,source,out))
                environment.assert_called_once_with(include_torch=False)
                self.assertEqual(json.loads((out/'result.json').read_text())['status'],'COMPLETED')
                completed=sorted((out/'measurements/val').glob('*.json'))
                hashes=[sha256(p) for p in completed]
                with patch.object(cm,'process_measurements',side_effect=RuntimeError('synthetic later failure')):
                    self.assertFalse(cm.run(stage1,source,out))
                self.assertEqual(hashes,[sha256(p) for p in completed])
                with zipfile.ZipFile(out/'07_crop_birlestirme_rapor.zip') as z:
                    self.assertNotIn('merge_summary.json',z.namelist())
                    self.assertEqual(json.loads(z.read('result.json'))['status'],'ERROR')

    def test_measure_summary_preview_bundle_and_cache_only_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage1,source,out,records,sources,cfg=fixture(Path(tmp))
            digest=fingerprint(cfg)
            paths,work=cm.process_measurements(records,stage1/'data',source,out,sources,cfg,digest)
            self.assertEqual(work['images'],2);self.assertEqual(work['new_this_invocation'],2)
            with patch.object(cm,'measure_one',side_effect=AssertionError('Must reuse cached measurement')):
                _,again=cm.process_measurements(records,stage1/'data',source,out,sources,cfg,digest)
            self.assertEqual(again['reused_this_invocation'],2)
            summary=cm.summarize(paths,cfg,digest,sources['coverage'])
            self.assertTrue(summary['unmerged_controls_match_06']);self.assertEqual(len(summary['configurations']),20)
            summary['previews']=cm.make_previews(paths,stage1/'data',out,2)
            cm.write_outputs(out,summary)
            write_json(out/'result.json',{'status':'COMPLETED'})
            with zipfile.ZipFile(cm.make_bundle(out)) as z:
                self.assertEqual(len([n for n in z.namelist() if n.startswith('previews/')]),2)
                self.assertFalse(any(n.startswith('measurements/') for n in z.namelist()))
            self.assertEqual(summary['configurations'][0]['targets'],4)

    def test_interruption_and_resume_reuse_first_committed_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage1,source,out,records,sources,cfg=fixture(Path(tmp));digest=fingerprint(cfg)
            real=cm.write_json
            def interrupt(path,value):
                if Path(path).name=='example_2.json':raise InterruptedError('synthetic stop before second image commit')
                return real(path,value)
            with patch.object(cm,'write_json',side_effect=interrupt):
                with self.assertRaises(InterruptedError):cm.process_measurements(records,stage1/'data',source,out,sources,cfg,digest)
            _,work=cm.process_measurements(records,stage1/'data',source,out,sources,cfg,digest)
            self.assertEqual(work['new_this_invocation'],1);self.assertEqual(work['reused_this_invocation'],1)

    def test_source_prediction_and_label_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage1,source,out,records,sources,cfg=fixture(Path(tmp))
            path=prediction_path(source,records[1]);original=path.read_bytes();path.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError,'tahmin'):cm.input_for(records[1],stage1/'data',source,sources)
            path.write_bytes(original)
            label=stage1/'data/labels/val/example_1.txt';label.write_text('')
            with self.assertRaisesRegex(ValueError,'etiketi'):cm.input_for(records[1],stage1/'data',source,sources)

    def test_changed_measurement_and_wrong_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage1,source,out,records,sources,cfg=fixture(Path(tmp));digest=fingerprint(cfg)
            paths,_=cm.process_measurements(records,stage1/'data',source,out,sources,cfg,digest)
            record=records[1];payload=json.loads(paths[0].read_text())
            args=(record,'different',payload['source_prediction_sha256'],payload['source_label_sha256'])
            with self.assertRaisesRegex(ValueError,'kimligi'):cm.validate_measurement(payload,*args)
            payload['variants'][0]['crop_count']=999
            with self.assertRaisesRegex(ValueError,'hash'):cm.validate_measurement(payload,*args)

    def test_training_role_cannot_be_measured_and_wrong_baseline_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage1,source,out,records,sources,cfg=fixture(Path(tmp))
            with self.assertRaises(ValueError):cm.input_for(records[0],stage1/'data',source,sources)
            paths,_=cm.process_measurements(records,stage1/'data',source,out,sources,cfg,fingerprint(cfg))
            bad=copy.deepcopy(sources['coverage']);bad['configurations'][0]['counts']['all']['ioa90']+=1
            with self.assertRaisesRegex(ValueError,'yeniden uretmedi'):cm.summarize(paths,cfg,fingerprint(cfg),bad)

    def test_error_bundle_does_not_reuse_previous_success_table_or_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);write_json(out/'result.json',{'status':'ERROR'})
            write_json(out/'merge_summary.json',{'old':True})
            (out/'previews').mkdir();(out/'previews/sample_01.jpg').write_bytes(b'old')
            with zipfile.ZipFile(cm.make_bundle(out)) as z:self.assertEqual(z.namelist(),['result.json'])


if __name__=='__main__':unittest.main()
