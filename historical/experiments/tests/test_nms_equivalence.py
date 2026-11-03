import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hycado_start import nms_equivalence as n


class NMSTests(unittest.TestCase):
    def test_ties_duplicates_cross_class_and_empty(self):
        cases=[[],[[0,0,10,10,.5,0],[0,0,10,10,.5,0],[0,0,10,10,.5,1]],
               [[-1,-1,10,10,.4,0],[0,0,8,8,.3,0],[10,10,10,11,.7,0]]]
        for rows in cases:
            self.assertEqual(n.fast_nms(rows,100,100),n.old.classwise_nms(rows,100,100))

    def test_float64_iou_exact_and_either_side_of_threshold(self):
        for shift in [3.,3.-1e-10,3.+1e-10]:
            rows=[[0,0,17,1,.8,0],[shift,0,shift+17,1,.8,0]]
            self.assertEqual(n.fast_nms(rows,30,3),n.old.classwise_nms(rows,30,3))
        self.assertEqual(len(n.fast_nms([[0,0,17,1,.8,0],[3,0,20,1,.7,0]],30,3)),2)

    def test_random_dense_boxes_ties_and_global_caps(self):
        rng=np.random.default_rng(42)
        for _ in range(60):
            low=rng.integers(0,25,(150,2));hi=low+rng.integers(1,30,(150,2))
            rows=np.column_stack([low,hi,rng.choice([.0011,.1,.5,.9],150),rng.integers(0,10,150)]).tolist()
            for limit in [0,7,1000]:
                self.assertEqual(n.fast_nms(rows,60,60,limit=limit),n.old.classwise_nms(rows,60,60,limit=limit))

    def test_global_first1000_after_independent_class_nms(self):
        rows=[[2*i,0,2*i+1,1,.5,i%10] for i in range(1100)]
        self.assertEqual(n.fast_nms(rows,3000,10),n.old.classwise_nms(rows,3000,10))
        self.assertEqual(len(n.fast_nms(rows,3000,10)),1000)

    def test_real_npz_candidate_reconstruction_no_GT_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=np.zeros((1,300,14),dtype=np.float32);raw[0,0,:4]=[.5,.5,.3,.3];raw[0,0,4]=.8
            rects=[[10,10,60,60]];np.savez_compressed(root/'raw_crop_queries.npz',raw_queries=raw,crop_rects=np.array(rects,dtype=np.int32))
            record={'image':'images/val/a.png','image_sha256':'pixels','width':100,'height':100}
            expected=n.old.map_queries(raw[0],{'source_crop_xyxy':rects[0]})
            for row in expected:row[4]*=.25
            p={'protocol_sha256':n.runtime.P12,'source_image':record['image'],'image_sha256':'pixels','width':100,'height':100,
               'baseline':[],'hybrid_a025':expected,'raw_archive_sha256':n.sha256(root/'raw_crop_queries.npz'),'crop_count':1,'crop_rects':rects}
            n.val.signed_json(root/'prediction.json',p)
            item={'path':'prediction.json','sha256':n.sha256(root/'prediction.json'),'raw_archive_sha256':p['raw_archive_sha256'],'crops':1,'source_image':record['image']}
            rows,_=n.candidates(root,item,{'records':[record]})
            self.assertEqual(rows,expected);self.assertEqual(n.fast_nms(rows,100,100),expected)
            with (root/'raw_crop_queries.npz').open('ab') as f:f.write(b'changed')
            with self.assertRaises(ValueError):n.candidates(root,item,{'records':[record]})

    def test_interruption_does_not_commit_partial_trial_then_full_trial_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);index={'records':[{'source_image':str(i)} for i in range(548)]}
            config={'source_images':[str(i) for i in range(548)]};rows=[[0.,0.,10.,10.,.9,0.]]
            p={'width':100,'height':100,'hybrid_a025':rows};calls=[]
            def stop(*args):
                calls.append(1)
                if len(calls)==2:raise KeyboardInterrupt()
                return rows,p
            with patch.object(n,'candidates',side_effect=stop),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):n.measure_trial(output,index,{},output,config,0)
            self.assertFalse((output/'trial_1.json').exists())
            with patch.object(n,'candidates',return_value=(rows,p)),contextlib.redirect_stdout(io.StringIO()):
                result=n.measure_trial(output,index,{},output,config,0)
            n.validate_trial(result,config,0);self.assertEqual(len(result['rows']),548)
            result['rows'][0]['cpp_exact12']=False
            with self.assertRaises(ValueError):n.validate_trial(result,config,0)

    def test_summary_speed_gate_is_only_a_recommendation(self):
        trials=[{'trial':i,'rows':[{'old_ms':100.,'cpp_ms':10.}]} for i in range(3)]
        s=n.summary(trials);self.assertTrue(s['implementation_candidate_recommended']);self.assertFalse(s['automatic_installation'])
        trials[1]['rows'][0]['cpp_ms']=200.
        self.assertFalse(n.summary(trials)['implementation_candidate_recommended'])


if __name__=='__main__':unittest.main()
