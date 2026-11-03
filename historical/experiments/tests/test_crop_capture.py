"""Query preservation, unchanged NMS, provenance and interrupted capture tests."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from hycado_start import crop_capture as c
from hycado_start.common import write_json,sha256
from hycado_start.stage1 import fingerprint


def queries():
    a=np.zeros((300,14),dtype=np.float32)
    a[23,:4]=[.05,.05,.1,.1];a[23,4]=.8
    return a


class CaptureTests(unittest.TestCase):
    def test_query_ids_preserved_after_filtering(self):
        rows,ids=c.decode_with_ids(queries(),{'source_crop_xyxy':[0,0,100,100]})
        self.assertEqual(ids.tolist(),[23]);self.assertEqual(len(rows),1)

    def test_suppression_trace_records_cross_stage_and_cross_class(self):
        rows=[[0,0,10,10,.8,0],[0,0,10,10,.9,0],[0,0,10,10,.7,1]]
        origins=[[0,-1,0],[1,7,23],[1,7,24]]
        a,o,status,by,selected,summary=c.trace_nms(rows,origins,20,20)
        self.assertEqual(status.tolist(),[2,1,1]);self.assertEqual(by.tolist(),[1,-1,-1])
        self.assertEqual(selected.tolist(),[1,2]);self.assertEqual(summary['baseline_suppressed_by_crop'],1)

    def test_global_limit_trace(self):
        rows=[[i*2,0,i*2+1,1,.9,0] for i in range(1001)]
        origins=[[0,-1,i] for i in range(1001)]
        result=c.trace_nms(rows,origins,3000,10)
        self.assertEqual(np.count_nonzero(result[2]==1),1000)
        self.assertEqual(result[2][-1],3);self.assertEqual(result[3][-1],-1)

    def test_crop_suppression_of_true_baseline_box_with_bad_gt_alignment(self):
        rows=[[2,0,12,10,.8,0],[3.5,0,13.5,10,.9,0]]
        a,o,status,by,selected,_=c.trace_nms(rows,[[0,-1,0],[1,0,23]],100,100)
        arrays={'candidate_rows':a,'candidate_origins':o,'nms_status':status,
                'suppressed_by_candidate':by,'selected_candidate_ids':selected}
        gt=np.array([[0,0,10,10,1,1,0,0]],dtype=float)
        report=c.per_source_diagnostics(gt,100,100,rows[:1],rows[:1],arrays)
        self.assertEqual(report['baseline_TP_trace']['0.5'],{'suppressed_by_crop_suppressor_cannot_match_same_GT':1})
        self.assertEqual(report['counts_at_fixed_max500']['0.5']['baseline']['TP'],1)
        self.assertEqual(report['counts_at_fixed_max500']['0.5']['hybrid_replay']['TP'],0)

    def test_randomized_nms_matches_frozen_11_and_suppressor_is_selected(self):
        rng=np.random.default_rng(7)
        for _ in range(20):
            left=rng.uniform(0,80,(80,2));right=np.minimum(100,left+rng.uniform(1,30,(80,2)))
            a=np.column_stack([left,right,rng.choice([.3,.7,.9],80),rng.integers(0,3,80)])
            origin=np.column_stack([rng.integers(0,2,80),np.zeros(80),np.arange(80)]).astype(int)
            rows,o,status,by,selected,summary=c.trace_nms(a.tolist(),origin,100,100)
            self.assertTrue(all(status[by[i]]==1 for i in np.flatnonzero(status==2)))
            self.assertEqual(int(np.count_nonzero(status==1)),len(selected))

    def test_npz_lossless_roundtrip_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc={'width':100,'height':100,'crops':[{'crop_index':7,'image':'a.png','source_crop_xyxy':[0,0,100,100]}]}
            arrays,summary=c.capture_parent(doc,Path(tmp),[],lambda *args:queries())
            dest=Path(tmp)/'committed'
            meta=c.save_parent(dest,arrays,{'protocol_sha256':'recipe','source_image':'a',
                'source_crop_document_sha256':'doc','old11_prediction_sha256':'old','crop_count':1})
            result,_=c.read_parent(dest,'recipe','a','doc','old')
            for key in arrays:np.testing.assert_array_equal(result[key],arrays[key])
            self.assertEqual(result['raw_queries'].dtype,np.float32)
            with self.assertRaises(FileExistsError):c.save_parent(dest,arrays,{})
            with (dest/'queries_and_trace.npz').open('ab') as f:f.write(b'corrupted')
            with self.assertRaises(ValueError):c.read_parent(dest,'recipe','a','doc','old')

    def test_interrupted_capture_reuses_committed_source_and_preserves_old_11(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'experiments/development_eval_v1_seed42';out=root/'capture';out.mkdir()
            parents=[];index={'records':[]}
            for i in range(2):
                folder=root/f'crops/source_{i}';folder.mkdir(parents=True)
                crops=[]
                for j in range(2):
                    png=folder/f'{j}.png';png.write_bytes(b'PNG fixture '+bytes([j]))
                    crops.append({'image':png.name,'image_sha256':sha256(png),'width':100,'height':100,
                                  'crop_index':j,'source_crop_xyxy':[0,0,100,100],
                                  'supervision_state':'fully_ignored_no_positive' if j else 'positive'})
                doc={'width':100,'height':100,'crops':crops}
                r={'source_image':f'images/train/{i}.jpg'}
                p={'protocol_sha256':c.SOURCE_PROTOCOL,'source_image':r['source_image'],
                    'width':100,'height':100,'baseline':[[0.,0.,10.,10.,.9,0]],
                    'hycado':[[0.,0.,10.,10.,.9,0]],'all_crops_processed':2,
                    'source_crop_document_sha256':fingerprint(doc)}
                p['content_sha256']=fingerprint(p);path=source/f'predictions/{i}.json';write_json(path,p)
                ann=source/f'reference_annotations/{i}.txt';ann.parent.mkdir(parents=True,exist_ok=True);ann.write_text('0,0,10,10,1,1,0,0\n')
                index['records'].append({'source_image':r['source_image'],'path':f'predictions/{i}.json',
                                        'sha256':sha256(path),'annotation_sha256':sha256(ann)})
                parents.append((r,doc,folder))
            before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            attempts=[]
            def stop(model,path,crop):
                attempts.append(str(path))
                if len(attempts)==3:raise RuntimeError('synthetic interruption')
                return queries()
            fake_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True))
            with patch.dict('sys.modules',{'torch':fake_torch}),patch.object(c,'predict_queries',side_effect=stop):
                with self.assertRaisesRegex(RuntimeError,'interruption'):
                    c.capture_all(root,out,parents,index,'recipe',None,None,factory=lambda *a:object())
            archive=out/'sources/0/queries_and_trace.npz';saved=archive.read_bytes()
            calls=[]
            def predict(model,path,crop):calls.append(crop['supervision_state']);return queries()
            with patch.dict('sys.modules',{'torch':fake_torch}),patch.object(c,'predict_queries',side_effect=predict):
                report=c.capture_all(root,out,parents,index,'recipe',None,None,factory=lambda *a:object())
            self.assertEqual(report['images'],2);self.assertEqual(report['crops'],4)
            self.assertEqual(report['exact_original11_images'],2)
            self.assertEqual(len(calls),2);self.assertIn('fully_ignored_no_positive',calls)
            self.assertEqual(archive.read_bytes(),saved)
            self.assertTrue(all(p.read_bytes()==data for p,data in before.items()))


@unittest.skipUnless(importlib.util.find_spec('torch'),'CPU Torch optional')
class PreprocessingTests(unittest.TestCase):
    def test_same_rgb_resize_and_float_normalization_as_11(self):
        import torch
        import cv2
        from PIL import Image
        class Model:
            def predict(self,x):
                self.input=x.clone();return torch.from_numpy(queries()).unsqueeze(0),None
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'image.png';Image.new('RGB',(23,19),(1,2,255)).save(path)
            crop={'width':23,'height':19,'image_sha256':sha256(path)};model=Model()
            result=c.predict_queries(model,path,crop,'cpu')
            with Image.open(path) as im:rgb=np.array(im.convert('RGB'))
            expected=torch.from_numpy(cv2.resize(rgb,(960,960),interpolation=cv2.INTER_LINEAR).transpose(2,0,1).copy()).unsqueeze(0).to(dtype=torch.float32)/255.
            self.assertTrue(torch.equal(model.input,expected));np.testing.assert_array_equal(result,queries())


if __name__=='__main__':unittest.main()
