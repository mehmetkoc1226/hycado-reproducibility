"""CPU-only synthetic diagnostics; no metric tuning or model loading."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
import numpy as np
from hycado_start import dev_diagnostics as d
from hycado_start.common import write_json,sha256
from hycado_start.stage1 import fingerprint


class DiagnosticTests(unittest.TestCase):
    def test_matching_with_duplicate_and_ignore(self):
        gt=np.array([[0,0,10,10,0,1],[20,20,20,20,1,1]],dtype=float)
        det=np.array([[0,0,10,10,.9,1],[0,0,10,10,.8,1],[22,22,5,5,.7,1]],dtype=float)
        flags,ids=d.detailed_matches(det,gt,.5)
        np.testing.assert_array_equal(flags,[1,0,-1]);np.testing.assert_array_equal(ids,[0,-1,-1])
        self.assertEqual(d.false_positive_reasons(det,gt,flags,.5)[1],'duplicate_same_class_overlap')

    def test_gained_and_lost_gt_and_box_absence(self):
        gt=np.array([[0,0,10,10,0,1],[20,20,10,10,0,2]],dtype=float)
        base=np.array([[0,0,10,10,.9,1]],dtype=float)
        hybrid=np.array([[20,20,10,10,.9,2]],dtype=float)
        report,rows=d.analyze_image(gt,base,hybrid,.5)
        self.assertEqual(report['gained_GT'],1);self.assertEqual(report['lost_GT'],1)
        self.assertEqual(report['lost_GT_observations'],{'baseline_TP_box_absent_from_final_hybrid':1})
        self.assertEqual(report['baseline']['FN_normal_GT'],1)

    def test_rank_cutoff_displacement_is_distinct_from_absence(self):
        gt=np.array([[0,0,10,10,0,1]],dtype=float)
        base=np.array([[0,0,10,10,.2,1]],dtype=float)
        fp=np.array([[100+i*20,100,10,10,.9,2] for i in range(500)],dtype=float)
        hybrid=np.concatenate([fp,base])
        report,_=d.analyze_image(gt,base,hybrid,.5)
        self.assertEqual(report['lost_GT_observations'],{'baseline_TP_box_below_global500_cutoff':1})
        self.assertEqual(report['hycado']['FP'],500)

    def test_fp_reason_partition(self):
        gt=np.array([[0,0,10,10,0,1],[30,30,10,10,0,2]],dtype=float)
        det=np.array([[0,0,10,10,.9,1],[0,0,10,10,.8,1],[30,30,10,10,.7,1],
                      [5,5,10,10,.6,1],[100,100,10,10,.5,1]],dtype=float)
        flags,_=d.detailed_matches(det,gt,.5)
        reasons=d.false_positive_reasons(det,gt,flags,.5)
        self.assertEqual(reasons,['','duplicate_same_class_overlap','other_class_overlap',
                                 'same_class_localization_0.1_to_threshold','background_or_unexplained'])

    def test_histograms_do_not_select_a_threshold_and_keep_score_one(self):
        gt=np.array([[0,0,10,10,0,1]],dtype=float)
        det=np.array([[0,0,10,10,1.,1],[40,40,10,10,.001,1]],dtype=float)
        report,rows=d.analyze_image(gt,det,det,.5)
        totals=d.initialize_totals();d.accumulate(totals,report,rows);answer=d.finalize(totals)
        self.assertEqual(answer['baseline']['score_bins'][-1]['TP'],1)
        self.assertEqual(answer['baseline']['counts']['precision_at_fixed_max500'],.5)
        self.assertEqual(sum(x.get('TP',0)+x.get('FP',0)+x.get('ignored_detections',0)
                             for x in answer['baseline']['score_bins']),2)

    def test_randomized_matching_identity_and_transition_invariants(self):
        rng=np.random.default_rng(12)
        for _ in range(20):
            gt=np.column_stack([rng.uniform(0,50,(20,2)),rng.uniform(5,30,(20,2)),
                                rng.integers(0,2,20),rng.integers(1,4,20)])
            base=np.column_stack([rng.uniform(0,50,(40,2)),rng.uniform(5,30,(40,2)),
                                  np.linspace(.99,.01,40),rng.integers(1,4,40)])
            hybrid=base[::-1].copy();hybrid[:,4]=np.linspace(.99,.01,40)
            report,_=d.analyze_image(gt,base,hybrid,.5)
            self.assertEqual(report['hycado']['TP']-report['baseline']['TP'],report['gained_GT']-report['lost_GT'])

    def test_real_cache_contract_hashes_and_source_files_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';output=root/'diagnostic';output.mkdir()
            p={'protocol_sha256':'fixed','source_image':'images/train/a.jpg','width':100,'height':100,
               'baseline':[[0.,0.,10.,10.,.9,0]],'hycado':[[0.,0.,10.,10.,.9,0]],'all_crops_processed':1}
            p['content_sha256']=fingerprint(p);cache=source/'predictions/a.json';write_json(cache,p)
            ann=source/'reference_annotations/a.txt';ann.parent.mkdir(parents=True);ann.write_text('0,0,10,10,1,1,0,0\n')
            index={'records':[{'source_image':p['source_image'],'path':'predictions/a.json',
                               'sha256':sha256(cache),'annotation_sha256':sha256(ann)}]}
            before={path:path.read_bytes() for path in source.rglob('*') if path.is_file()}
            report=d.diagnostic_analysis(source,output,index,'fixed')
            self.assertEqual(report['per_IoU']['0.5']['detectors']['hycado']['counts']['TP'],1)
            self.assertTrue(all(path.read_bytes()==data for path,data in before.items()))
            with zipfile.ZipFile(d.bundle(output)) as z:
                self.assertIn('diagnostics.json',z.namelist());self.assertNotIn('predictions/a.json',z.namelist())
            ann.write_text('1,1,10,10,1,1,0,0\n')
            with self.assertRaises(ValueError):d.diagnostic_analysis(source,output,index,'fixed')

    def test_import_does_not_load_torch_cuda_or_ultralytics(self):
        script="import sys; from hycado_start import dev_diagnostics; assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"
        r=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)


if __name__=='__main__':unittest.main()
