"""Positive/ignore semantics and lossless, restartable crop materialization."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from hycado_start import crop_data as cd
from hycado_start.crop_labels import ANNOTATION_POLICY,canonical_source_targets,crop_annotations
from hycado_start.common import sha256,write_json
from hycado_start.data import parse_annotation
from hycado_start.stability import inspect_labels
from hycado_start.stage1 import fingerprint
from hycado_start.proposal_geometry import canonical_detections
from hycado_start.proposals import prediction_path


def clean_text(raw,width=100,height=100):
    p=parse_annotation(raw,width,height)
    text=''.join(str(r[0])+' '+' '.join(f'{v:.10f}' for v in r[1:])+'\n' for r in p['labels'])
    return inspect_labels(text,'synthetic')[0]


def canonical(raw):return canonical_source_targets(raw,100,100,clean_text(raw))


class CropLabelTests(unittest.TestCase):
    def test_exact_half_remains_positive_below_half_is_ignore(self):
        s=canonical('10,10,20,20,1,4,0,0\n')
        half=crop_annotations(s['targets'],[],(0,0,20,40))
        self.assertEqual(len(half['positive_targets']),1)
        self.assertEqual(half['positive_targets'][0]['visible_source_fraction'],.5)
        self.assertTrue(half['positive_targets'][0]['new_crop_truncation'])
        less=crop_annotations(s['targets'],[],(0,0,19,40))
        self.assertEqual(less['positive_targets'],[])
        self.assertEqual(less['ignore_regions'][0]['reason'],'partial_target_below_50pct')

    def test_local_box_translation_and_normalization(self):
        s=canonical('20,30,20,10,1,1,0,0\n')
        d=crop_annotations(s['targets'],[],(10,20,60,70))
        row=d['positive_targets'][0]
        self.assertEqual(row['box_xyxy'],[10.,10.,30.,20.])
        self.assertEqual(row['box_cxcywh_normalized'],[.4,.3,.4,.2])
        self.assertEqual(row['class_id'],0)

    def test_source_ignored_score_and_others_preserved(self):
        raw='0,0,10,10,1,0,0,0\n20,0,10,10,0,4,0,0\n40,0,10,10,1,11,0,0\n'
        s=canonical(raw);d=crop_annotations([],s['ignore_regions'],(0,0,60,20))
        self.assertEqual(d['positive_targets'],[])
        self.assertEqual([r['reason'] for r in d['ignore_regions']],['source_ignore','source_ignore','source_others'])
        self.assertEqual(d['supervision_state'],'ignore_with_background')

    def test_positive_over_ignore_not_erased(self):
        s=canonical('10,10,20,20,1,4,0,0\n0,0,100,100,0,0,0,0\n')
        d=crop_annotations(s['targets'],s['ignore_regions'],(0,0,100,100))
        self.assertEqual(len(d['positive_targets']),1)
        self.assertEqual(d['ignore_union_fraction'],1)
        self.assertEqual(d['supervision_state'],'positive')

    def test_empty_and_fully_ignored_are_distinguished(self):
        self.assertEqual(crop_annotations([],[],(0,0,10,10))['supervision_state'],'clean_background')
        s=canonical('0,0,100,100,0,0,0,0\n')
        self.assertEqual(crop_annotations([],s['ignore_regions'],(0,0,10,10))['supervision_state'],'fully_ignored_no_positive')

    def test_duplicates_and_invalid_geometry_match_stage1_audit(self):
        s=canonical('10,10,20,20,1,4,0,0\n10,10,20,20,1,4,0,0\n1,2,3,0,1,4,0,0\n')
        self.assertEqual(len(s['targets']),1)
        self.assertEqual(s['duplicate_target_rows_removed'],1)
        self.assertEqual(s['invalid_geometry_rows_omitted'],1)
        self.assertEqual(s['targets'][0]['source_record_index'],1)

    def test_touching_edge_does_not_create_positive_or_ignore_sliver(self):
        s=canonical('10,10,20,20,1,4,0,0\n')
        d=crop_annotations(s['targets'],[],(30,0,50,50))
        self.assertEqual(d['positive_targets'],[]);self.assertEqual(d['ignore_regions'],[])

    def test_one_pixel_target_is_not_silently_dropped(self):
        s=canonical('10,10,1,1,1,10,0,0\n')
        d=crop_annotations(s['targets'],[],(0,0,20,20))
        self.assertEqual(d['positive_targets'][0]['class_id'],9)
        self.assertEqual(d['positive_targets'][0]['box_xyxy'],[10.,10.,11.,11.])

    def test_raw_annotations_must_reproduce_clean_labels(self):
        with self.assertRaisesRegex(ValueError,'eslesmiyor'):
            canonical_source_targets('10,10,20,20,1,4,0,0',100,100,'')

    def test_ignore_union_does_not_double_count_overlapping_boxes(self):
        s=canonical('0,0,10,10,0,0,0,0\n5,0,10,10,0,0,0,0\n')
        d=crop_annotations([],s['ignore_regions'],(0,0,20,20))
        self.assertAlmostEqual(d['ignore_union_fraction'],150/400)


def fixture(root):
    from PIL import Image,ImageDraw
    stage1=root/'stage1';s6=root/'s6';s7=root/'s7';out=root/'out';out.mkdir()
    p6={'synthetic':'source06'};p7={'synthetic':'source07'};records=[];index={}
    decision={'selected_configuration_id':'c3_near_z3_b10',
              'proposal_generation':{'threshold':.05,'top_k':60,'context_scale':3.},
              'merge':{'policy':{'name':'near_z3','max_gap_ratio':.5,'max_axis_growth':3.,'max_pair_hull_ratio':2.},
                       'requested_crop_count':10}}
    raw='10,10,20,20,1,4,0,0\n28,10,20,20,1,1,0,0\n48,42,10,10,1,6,0,0\n0,0,8,8,0,0,0,0\n60,60,20,20,1,11,0,0\n'
    for i,split in enumerate(['train','train','val','val']):
        stem=f'example_{i}';image=f'images/{split}/{stem}.jpg'
        f=stage1/'data'/image;f.parent.mkdir(parents=True,exist_ok=True)
        im=Image.new('RGB',(100,100),(30,50,70));draw=ImageDraw.Draw(im)
        for x in range(0,100,5):draw.line((x,0,x,99),fill=(x,180,40))
        im.save(f)
        label=stage1/'data/labels'/split/(stem+'.txt');label.parent.mkdir(parents=True,exist_ok=True);label.write_text(clean_text(raw))
        annotation=stage1/'data/annotations'/split/(stem+'.txt');annotation.parent.mkdir(parents=True,exist_ok=True);annotation.write_text(raw)
        record={'image':image,'source_image':f'images/train/{stem}.jpg','split':split,'source_split':'train','group':str(i),
                'image_sha256':sha256(f),'label_sha256':sha256(label),'annotation_sha256':sha256(annotation),'gt_count':3}
        detections=[] if i==1 else [[10,10,30,30,.95,3],[62,62,78,78,.7,5],[85,5,95,15,.4,1]]
        detections,_=canonical_detections(detections,100,100)
        payload={'protocol_sha256':fingerprint(p6),'source_image':record['source_image'],'image_sha256':record['image_sha256'],
                 'split':split,'width':100,'height':100,'detections':detections,'dropped_zero_area':0}
        payload['content_sha256']=fingerprint(payload);cache=prediction_path(s6,record);write_json(cache,payload)
        index[record['source_image']]={'sha256':sha256(cache)}
        if split=='val':
            _,clusters=cd.select_clusters(payload,decision)
            m={'protocol_sha256':fingerprint(p7),'source_image':record['source_image'],
               'source_prediction_sha256':sha256(cache),'source_label_sha256':record['label_sha256'],
               'variants':[{'id':decision['selected_configuration_id'],'clusters':clusters}]}
            m['content_sha256']=fingerprint(m);write_json(cd.crop_merge.measurement_path(s7,record),m)
        records.append(record)
    sources={'p6':p6,'p7':p7,'decision':decision,'index':index,'manifest':{'records':records},
             'selected':{'mean_crops':3.,'images':2}}
    cfg={'decision':decision,'annotation_policy':copy.deepcopy(ANNOTATION_POLICY),'png_compress_level':1,'preview_count_per_split':1}
    return stage1,s6,s7,out,records,sources,cfg


class MaterializationTests(unittest.TestCase):
    def test_import_does_not_load_torch_or_ultralytics(self):
        p=subprocess.run([sys.executable,'-c',"import sys; import hycado_start.crop_data; assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_lossless_pixels_annotations_manifests_and_no_oracle_fallback(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            stage,s6,s7,out,records,sources,cfg=fixture(Path(tmp));digest=fingerprint(cfg)
            files,work=cd.process_sources(records,stage/'data',s6,s7,out,sources,cfg,digest)
            summary=cd.finalize_dataset(files,out,sources,cfg,digest)
            self.assertEqual(summary['splits']['training']['crops'],3)
            self.assertEqual(summary['splits']['development']['crops'],6)
            self.assertEqual(summary['splits']['training']['zero_crop_source_images'],1)
            self.assertFalse((out/'dataset.yaml').exists())
            for path in files:
                parent=json.loads(path.read_text())
                with Image.open(stage/'data'/parent['source_image_path']) as im:
                    original=im.convert('RGB')
                    for item in parent['crops']:
                        with Image.open(path.parent/item['image']) as crop:
                            np.testing.assert_array_equal(np.asarray(crop),np.asarray(original.crop(item['source_crop_xyxy'])))
                        a=json.loads((path.parent/item['annotation']).read_text())
                        self.assertTrue(a['requires_ignore_aware_loader'])
            with patch.object(cd,'build_parent',side_effect=AssertionError('must reuse committed parent')):
                _,again=cd.process_sources(records,stage/'data',s6,s7,out,sources,cfg,digest)
            self.assertEqual(again['reused_this_invocation'],4)
            previews=cd.make_previews(files,out,1);self.assertEqual(len(previews),2)
            summary['previews']=previews;cd.write_html(out,summary)
            write_json(out/'result.json',{'status':'COMPLETED'})
            with zipfile.ZipFile(cd.make_bundle(out)) as z:
                self.assertFalse(any(n.startswith('items/') or n.endswith('.png') for n in z.namelist()))
                self.assertIn('dataset_contract.json',z.namelist())

    def test_image_level_atomic_failure_removes_uncommitted_temporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage,s6,s7,out,records,sources,cfg=fixture(Path(tmp));digest=fingerprint(cfg)
            real=cd.write_json
            def fail(path,value):
                if Path(path).name=='crop_0001.json':raise InterruptedError('synthetic write failure')
                return real(path,value)
            with patch.object(cd,'write_json',side_effect=fail):
                with self.assertRaises(InterruptedError):cd.process_sources(records,stage/'data',s6,s7,out,sources,cfg,digest)
            self.assertFalse(cd.parent_dir(out,records[0]).exists())
            self.assertEqual(list((out/'building').iterdir()),[])
            files,_=cd.process_sources(records,stage/'data',s6,s7,out,sources,cfg,digest)
            self.assertEqual(len(files),4)

    def test_resume_after_first_source_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage,s6,s7,out,records,sources,cfg=fixture(Path(tmp));digest=fingerprint(cfg);real=cd.build_parent
            def stop(output,record,*args,**kwargs):
                if record['image']==records[1]['image']:raise InterruptedError('synthetic source interruption')
                return real(output,record,*args,**kwargs)
            with patch.object(cd,'build_parent',side_effect=stop):
                with self.assertRaises(InterruptedError):cd.process_sources(records,stage/'data',s6,s7,out,sources,cfg,digest)
            _,work=cd.process_sources(records,stage/'data',s6,s7,out,sources,cfg,digest)
            self.assertEqual(work['reused_this_invocation'],1);self.assertEqual(work['new_this_invocation'],3)

    def test_source_png_annotation_and_role_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage,s6,s7,out,records,sources,cfg=fixture(Path(tmp));digest=fingerprint(cfg)
            files,_=cd.process_sources(records,stage/'data',s6,s7,out,sources,cfg,digest)
            folder=files[0].parent;body=json.loads(files[0].read_text());hashes=body['source_hashes']
            image=folder/body['crops'][0]['image'];old=image.read_bytes();image.write_bytes(old+b'corruption')
            with self.assertRaises(ValueError):cd.validate_parent(folder,records[0],digest,hashes)
            image.write_bytes(old)
            with self.assertRaises(ValueError):cd.validate_parent(folder,records[0],'wrong',hashes)
            raw=stage/'data/annotations/train/example_0.txt';raw.write_text('')
            with self.assertRaises(ValueError):cd.read_source(records[0],stage/'data',s6,s7,sources)

    def test_development_geometry_change_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage,s6,s7,out,records,sources,cfg=fixture(Path(tmp))
            record=records[2];path=cd.crop_merge.measurement_path(s7,record);m=json.loads(path.read_text())
            m['variants'][0]['clusters'][0]['rect'][2]+=1 if isinstance(m['variants'][0]['clusters'][0]['rect'],list) else 0
            m.pop('content_sha256');m['content_sha256']=fingerprint(m);write_json(path,m)
            with self.assertRaisesRegex(ValueError,'geometrisi'):cd.read_source(record,stage/'data',s6,s7,sources)

    def test_standard_loader_contract_and_error_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);write_json(out/'result.json',{'status':'ERROR'})
            write_json(out/'dataset_summary.json',{'old_success':True})
            with zipfile.ZipFile(cd.make_bundle(out)) as z:self.assertEqual(z.namelist(),['result.json'])

    def test_full_run_success_uses_embedded_decision_then_reports_error_without_stale_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage,s6,s7,out,records,sources,cfg=fixture(Path(tmp))
            with patch.object(cd,'validate_inputs',return_value=sources),patch.object(cd,'configuration_for',return_value=cfg),\
                    patch.object(cd,'system_info',return_value={'synthetic_cpu_fixture':True}) as env:
                self.assertTrue(cd.run(stage,s6,s7,out,Path(tmp)/'embedded.json'))
                env.assert_called_with(include_torch=False)
                self.assertEqual(json.loads((out/'result.json').read_text())['crop_counts'],{'training':3,'development':6})
                with patch.object(cd,'process_sources',side_effect=RuntimeError('synthetic later failure')):
                    self.assertFalse(cd.run(stage,s6,s7,out,Path(tmp)/'embedded.json'))
                with zipfile.ZipFile(out/'08_crop_veri_rapor.zip') as z:self.assertNotIn('dataset_summary.json',z.namelist())


if __name__=='__main__':unittest.main()
