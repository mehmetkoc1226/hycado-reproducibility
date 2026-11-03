import contextlib
import copy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
import numpy as np
from hycado_start import size_diagnostics as s


def raw(rows):return np.array(rows,dtype=np.float64).reshape(-1,8)
def payload(baseline,hybrid,width=500,height=500):return {'width':width,'height':height,'baseline':baseline,'hybrid_a025':hybrid}
def reference(parts):
    return {'variants':{v:{'counts_at_fixed_max500':{t:{k:sum(x['thresholds'][t]['detections'][v][k] for x in parts)
             for k in ['TP','FP','ignored_detections']} for t in ['0.5','0.75']}} for v in ['baseline','hybrid_a025']}}


class SizeTests(unittest.TestCase):
    def test_size_and_object_count_boundaries_are_disjoint(self):
        for dimensions,group in [((1,255),'lt16_squared'),((16,16),'16_to_32_squared'),((32,32),'32_to_96_squared'),
                                 ((96,96),'ge96_squared'),((1,1023),'16_to_32_squared'),((0,10),'nonpositive_box'),((-1,10),'nonpositive_box')]:
            self.assertEqual(s.size_group(*dimensions),group)
        with self.assertRaises(ValueError):s.size_group(float('nan'),10)
        self.assertEqual([s.object_count_group(x) for x in [0,1,30,31,100,101]],
                         ['zero_GT','1_to_30_GT','1_to_30_GT','31_to_100_GT','31_to_100_GT','101_plus_GT'])

    def test_gained_tiny_lost_large_and_all_class_size_partitions(self):
        gt=raw([[0,0,10,10,1,1,0,0],[100,100,100,100,1,2,0,0]])
        data=s.validate_analysis(s.analyze(gt,payload([[100,100,200,200,.8,1]],[[0,0,10,10,.7,0]])))
        for t in data['thresholds'].values():
            rows={(r['class_id'],r['area_group']):r for r in t['class_size_rows']}
            self.assertEqual(rows[0,'lt16_squared']['gained_GT'],1);self.assertEqual(rows[0,'ge96_squared']['lost_GT'],1)
            self.assertEqual(rows[1,'lt16_squared']['delta_recall_pp'],100)
            self.assertEqual(rows[2,'ge96_squared']['delta_recall_pp'],-100)
            self.assertEqual(t['overall']['delta_recall_pp'],0)

    def test_matching_before_grouping_and_global_cap_before_class_filter(self):
        gt=raw([[0,0,10,10,1,1,0,0],[0,0,20,20,1,1,0,0]])
        det=[[0,0,20,20,.9,0]]
        d=s.analyze(gt,payload(det,det))['thresholds']['0.5']
        rows={r['area_group']:r for r in d['class_size_rows'] if r['class_id']==0}
        self.assertEqual(rows['lt16_squared']['baseline_matched_GT'],0)
        self.assertEqual(rows['16_to_32_squared']['baseline_matched_GT'],1)
        # The true class-1 hit at rank 501 must not enter after filtering class-2 FP.
        fp=[[300,300,310,310,.99-i*.0001,1] for i in range(500)]+[[0,0,10,10,.1,0]]
        gt=raw([[0,0,10,10,1,1,0,0]])
        d=s.analyze(gt,payload(fp,fp))['thresholds']['0.5']
        self.assertEqual(d['detections']['baseline'],{'TP':0,'FP':500,'ignored_detections':0})
        self.assertEqual(d['overall']['baseline_matched_GT'],0)

    def test_ignore_category11_duplicates_and_empty_images_preserved(self):
        gt=raw([[0,0,10,10,1,1,0,0],[20,20,30,30,0,1,0,0],[70,70,10,10,1,11,0,0]])
        det=[[0,0,10,10,.9,0],[0,0,10,10,.8,0],[25,25,30,30,.7,0],[70,70,80,80,.6,0]]
        d=s.validate_analysis(s.analyze(gt,payload(det,det)))
        self.assertEqual(d['normal_GT'],1)
        self.assertEqual(d['thresholds']['0.5']['detections']['baseline'],{'TP':1,'FP':2,'ignored_detections':1})
        empty=s.validate_analysis(s.analyze(raw([]),payload([[0,0,10,10,.9,0]],[])))
        self.assertEqual(empty['object_count_group'],'zero_GT')
        self.assertIsNone(empty['thresholds']['0.5']['overall']['baseline_recall_percent'])
        summary=s.combine([d,empty],reference([d,empty]))
        groups={r['object_count_group']:r for r in summary['thresholds']['0.5']['object_count_groups']}
        self.assertEqual(groups['zero_GT']['images'],1);self.assertEqual(groups['zero_GT']['detection_counts']['baseline']['FP'],1)
        # Category-0 ignore map uses the unchanged original integral/filter path.
        ignored=s.analyze(raw([[1,1,50,50,0,0,0,0],[5,5,10,10,1,1,0,0]]),payload([[5,5,15,15,.9,0]],[]))
        self.assertEqual(ignored['normal_GT'],0)
        self.assertEqual(ignored['thresholds']['0.5']['detections']['baseline']['FP'],0)

    def test_aggregation_uses_GT_weighting_and_checks_original12_counts(self):
        one=s.analyze(raw([[0,0,10,10,1,1,0,0]]),payload([[0,0,10,10,.9,0]],[]))
        many=s.analyze(raw([[20*i,20,10,10,1,1,0,0] for i in range(99)]),payload([],[]))
        metrics=reference([one,many]);result=s.combine([one,many],metrics)
        overall=next(x for x in result['thresholds']['0.5']['size_rows_all_classes'] if x['area_group']=='all')
        self.assertEqual(overall['baseline_recall_percent'],1.)
        self.assertEqual(overall['delta_recall_pp'],-1.)
        metrics['variants']['baseline']['counts_at_fixed_max500']['0.5']['FP']=1
        with self.assertRaisesRegex(ValueError,'12 ile farkli'):s.combine([one,many],metrics)

    def test_cache_validation_rejects_bad_hash_binding_partition_and_fraction(self):
        d=s.analyze(raw([[0,0,10,10,1,1,0,0]]),payload([],[]))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.json';bindings={'fixed':'source'};s.save_cache(p,d,bindings);self.assertEqual(s.load_cache(p,bindings),d)
            with self.assertRaises(ValueError):s.load_cache(p,{'fixed':'changed'})
            doc=s.read_json(p);doc['analysis']['normal_GT']=2;s.write_json(p,doc)
            with self.assertRaises(ValueError):s.load_cache(p,bindings)
        for kind in ['missing','counts','fraction']:
            bad=copy.deepcopy(d);rows=bad['thresholds']['0.5']['class_size_rows']
            if kind=='missing':rows.pop()
            elif kind=='counts':rows[0]['normal_GT']=5
            else:bad['thresholds']['0.5']['overall']['normal_GT']=1.5
            with self.assertRaises(ValueError):s.validate_analysis(bad)

    def test_real_cache_contract_resume_source_integrity_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'source';out=root/'analysis';out.mkdir();records=[];index=[];parts=[]
            for i in range(3):
                anno=root/'data/prepared/visdrone_annotations/val'/f'{i}.txt';anno.parent.mkdir(parents=True,exist_ok=True)
                anno.write_text('0,0,10,10,1,1,0,0\n',encoding='utf-8')
                rec={'image':f'images/val/{i}.jpg','width':100,'height':100,'image_sha256':'synthetic','annotation_sha256':s.sha256(anno)}
                records.append(rec);p=payload([[0,0,10,10,.9,0]],[[0,0,10,10,.8,0]],100,100)
                p.update(protocol_sha256=s.o.r.P12,source_image=rec['image'],image_sha256=rec['image_sha256'])
                path=base/f'{i}.json';s.o.r.val.signed_json(path,p)
                index.append({'source_image':rec['image'],'path':path.name,'sha256':s.sha256(path),'annotation_sha256':rec['annotation_sha256']})
                parts.append(s.analyze(s.raw_annotation(anno.read_text()),p))
            s.write_json(base/'prediction_index.json',{'records':index});s.write_json(base/'metrics.json',reference(parts))
            before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()};config={'fixed':True}
            with contextlib.redirect_stdout(io.StringIO()):s.process(root,base,{'records':records},out,config)
            cachehash={p:s.sha256(p) for p in (out/'images').glob('*.json')}
            with contextlib.redirect_stdout(io.StringIO()):result=s.process(root,base,{'records':records},out,config)
            self.assertEqual(result['images'],3);self.assertEqual(cachehash,{p:s.sha256(p) for p in cachehash})
            self.assertTrue(all(p.read_bytes()==b for p,b in before.items()))
            with zipfile.ZipFile(s.bundle(out)) as z:
                self.assertIsNone(z.testzip());self.assertEqual(len([n for n in z.namelist() if n.startswith('images/')]),3)
                self.assertIn('size_recall.csv',z.namelist());self.assertFalse(any(n.endswith('.jpg') for n in z.namelist()))
            (root/'data/prepared/visdrone_annotations/val/0.txt').write_text('1,1,10,10,1,1,0,0\n')
            with self.assertRaises(ValueError):s.process(root,base,{'records':records},out,config)

    def test_import_does_not_load_torch_or_ultralytics(self):
        result=subprocess.run([sys.executable,'-c',"import sys;from hycado_start import size_diagnostics;assert 'torch' not in sys.modules;assert 'ultralytics' not in sys.modules"],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_small_backup_copies_new_files_only_and_repairs_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';code=root/'hycado_start/size_diagnostics.py';code.parent.mkdir(parents=True);code.write_bytes(b'code')
            weights=root/'weights.pt';weights.write_bytes(b'not copied')
            out=root/'experiments/test';out.mkdir(parents=True);s.write_json(out/'result.json',{'status':'ERROR'});s.bundle(out)
            target=Path(s.small_backup(root,out,True));self.assertFalse(list(target.rglob('weights.pt')))
            dest=next(target.rglob('size_diagnostics.py'));dest.write_bytes(b'corrupt');s.small_backup(root,out)
            self.assertEqual(dest.read_bytes(),b'code');self.assertTrue(list(target.rglob('16_boyut_rapor.zip')))


if __name__=='__main__':unittest.main()
