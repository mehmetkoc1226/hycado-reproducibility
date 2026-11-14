import contextlib,io,unittest,hashlib,tempfile,sqlite3
from pathlib import Path
from unittest.mock import patch
import numpy as np
from pycocotools.cocoeval import COCOeval
import coco_core as c
import visdrone_preprocess as v

def reference(records,ground,det):
    images=[];ga=[];da=[];gi=di=0
    for rec,g,d in zip(records,ground,det):
        images.append({'id':rec['id'],'width':rec['width'],'height':rec['height']})
        for row in g:
            gi+=1;ga.append({'id':gi,'image_id':rec['id'],'category_id':int(row[4]),'bbox':list(row[:4]),
                            'area':float(row[2]*row[3]),'iscrowd':0})
        for row in d:
            di+=1;da.append({'id':di,'image_id':rec['id'],'category_id':int(row[5]),'bbox':list(row[:4]),
                            'area':float(row[2]*row[3]),'score':float(row[4]),'iscrowd':0})
    e=COCOeval(c.coco_object(images,ga),c.coco_object(images,da),'bbox')
    e.params.imgIds=[r['id'] for r in records];e.params.maxDets=c.CAPS
    with contextlib.redirect_stdout(io.StringIO()):e.evaluate();e.accumulate()
    return e

def streamed(records,ground,det):
    groups={}
    for rec,g,d in zip(records,ground,det):
        for e in c.one_image(rec,g,d):
            x,h,b=c.encode(e);decoded=c.decode(h,b)
            groups.setdefault((x['category_id'],x['area_index']),{})[rec['id']]=decoded
    return c.aggregate([r['id'] for r in records],lambda cat,area:groups.get((cat,area),{}))

class CocoTests(unittest.TestCase):
    def equal(self,recs,gt,dt):
        expected=reference(recs,gt,dt)
        results,p,r=streamed(recs,gt,dt)
        np.testing.assert_allclose(p,expected.eval['precision'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(r,expected.eval['recall'],rtol=0,atol=1e-12)
        return results

    def test_perfect_and_area_boundaries(self):
        rec=[{'id':9,'width':500,'height':500}]
        gt=[[[1,1,32,32,1],[100,100,96,96,2],[250,250,120,120,3]]]
        dt=[[[1,1,32,32,.9,1],[100,100,96,96,.8,2],[250,250,120,120,.7,3]]]
        d=self.equal(rec,gt,dt)
        for k in ('AP','AP_small','AP_medium','AP_large'):self.assertAlmostEqual(d['100'][k],100)

    def test_empty_predictions_and_no_gt_images(self):
        r=[{'id':1,'width':100,'height':100},{'id':6,'width':100,'height':100}]
        d=self.equal(r,[[[0,0,10,10,1]],[]],[[],[[20,20,10,10,.7,1]]])
        self.assertEqual(d['100']['AP'],0)

    def test_equal_scores_duplicate_boxes_and_classes(self):
        r=[{'id':1,'width':200,'height':200},{'id':4,'width':200,'height':200}]
        g=[[[10,10,20,20,1]],[[10,10,20,20,1],[80,80,20,20,2]]]
        d=[[[0,0,5,5,.5,1],[10,10,20,20,.5,1],[10,10,20,20,.5,1]],
           [[10,10,20,20,.5,1],[80,80,20,20,.5,2],[0,0,5,5,.5,2]]]
        self.equal(r,g,d)

    def test_caps_100_and_500(self):
        r=[{'id':2,'width':5000,'height':100}]
        g=[[[i*25,0,10,10,1] for i in range(150)]]
        d=[[[i*25,0,10,10,1-i/1000,1] for i in range(150)]]
        m=self.equal(r,g,d)
        self.assertAlmostEqual(m['100']['AR'],100*100/150)
        self.assertAlmostEqual(m['500']['AR'],100)
        self.assertGreater(m['500']['AP'],m['100']['AP'])

    def test_random_multiclass_reference_parity(self):
        rng=np.random.default_rng(20261112)
        recs=[{'id':k,'width':200,'height':200} for k in [1,3,8,12]]
        gt=[];dt=[]
        for _ in recs:
            g=[];d=[]
            for j in range(15):
                x,y=rng.uniform(0,130,2);w,h=rng.uniform(3,60,2);cls=j%3+1
                g.append([x,y,w,h,cls]);d.append([x+1,y,w,h,.7,cls])
            d.extend([[1,2,4,5,.7,1],[100,100,25,30,.8,2]])
            gt.append(g);dt.append(d)
        self.equal(recs,gt,dt)

    def test_cache_tampering_is_rejected(self):
        e=c.one_image({'id':1,'width':20,'height':20},[[0,0,4,4,1]],[])[0]
        _,h,b=c.encode(e)
        with self.assertRaises(ValueError):c.decode('wrong',b)

    def test_primary_and_secondary_caps_equal_separate_upstream_runs(self):
        rec={'id':1,'width':5000,'height':100}
        gt=[[i*25,0,10,10,1] for i in range(140)]
        dt=[[i*25,0,10,10,1-i/1000,1] for i in range(140)]
        _,p,r=streamed([rec],[gt],[dt])
        for caps,selected in [([1,10,100],[0,1,2]),([1,10,500],[0,1,3])]:
            g,d=c.objects(rec,gt,dt);e=COCOeval(g,d,'bbox');e.params.maxDets=caps
            with contextlib.redirect_stdout(io.StringIO()):e.evaluate();e.accumulate()
            np.testing.assert_allclose(p[:,:,:,:,selected],e.eval['precision'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(r[:,:,:,selected],e.eval['recall'],rtol=0,atol=1e-12)

    def test_original_ignore_preprocessing_excludes_regions(self):
        raw=v.raw_annotation('1,1,15,15,0,0,0,0\n3,3,5,5,1,1,0,0\n30,30,5,5,1,1,0,0\n')
        integral=v.ignore_integral(raw,50,50)
        normal=v.prepare_groundtruth(raw,integral)
        self.assertEqual(len(normal),1)
        det=v.outside_ignore(v.xyxy_to_det([[3,3,8,8,.9,0],[30,30,35,35,.8,0]]),integral)
        self.assertEqual(len(det),1);self.assertEqual(det[0,0],30)

    def test_streamed_runner_resume_and_missing_cache_entry(self):
        import coco_runner as runner
        class FakeSources:
            def __init__(self,root):pass
            def get(self,split,rec):return {'baseline':[[30,30,35,35,.8,0]],'cascade':[[30,30,35,35,.8,0]]}
            def close(self):pass
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);ann=root/'annotations';ann.mkdir();p=ann/'one.txt'
            p.write_text('30,30,5,5,1,1,0,0\n');before=p.read_bytes()
            rec={'id':1,'image':'one.jpg','width':100,'height':100,'annotation':'one.txt',
                 'annotation_sha256':hashlib.sha256(before).hexdigest()}
            sets={'fixture':{'records':[rec],'methods':['baseline','cascade'],'annotations':ann}}
            with patch.object(runner,'Sources',FakeSources),contextlib.redirect_stdout(io.StringIO()):
                a=runner.calculate(root,root,sets,'full','sig')[0]
                b=runner.calculate(root,root,sets,'full','sig')[0]
                self.assertEqual(a,b);self.assertAlmostEqual(a['fixture']['baseline']['100']['AP'],100)
                with contextlib.closing(sqlite3.connect(root/'evaluation_cache.sqlite3')) as db:
                    with db:
                        db.execute("DELETE FROM entries WHERE method='baseline' AND category=1 AND area=0")
                with self.assertRaises(ValueError):runner.calculate(root,root,sets,'full','sig')
            self.assertEqual(p.read_bytes(),before)

    def test_sahi_archive_over_1000_is_preserved_with_exact_index_count(self):
        import input_inventory as inventory
        rec={'width':100,'height':100,'image_sha256':'a'*64}
        rows=[[1,1,2,2,.5,0] for _ in range(1018)]
        body={'image_sha256':rec['image_sha256'],'detections':rows}
        entry={'source_image':'images/val/dense.jpg','detections':1018}
        self.assertEqual(inventory.validate_p2_rows(body,rec,'sahi640',entry),1018)
        self.assertEqual(len(rows),1018)
        with self.assertRaisesRegex(ValueError,'frozen index'):
            inventory.validate_p2_rows(body,rec,'sahi640',{**entry,'detections':1017})
        for bad in ([1,1,2,2,float('nan'),0],[1,1,1,2,.5,0],[1,1,2,2,.5,10]):
            invalid={'image_sha256':rec['image_sha256'],'detections':rows[:-1]+[bad]}
            with self.assertRaises(ValueError):inventory.validate_p2_rows(invalid,rec,'sahi640',entry)

    def test_yolo_and_cascade_caps_remain_enforced(self):
        import input_inventory as inventory
        rec={'width':100,'height':100,'image_sha256':'a'*64}
        body={'image_sha256':rec['image_sha256'],'detections':[[1,1,2,2,.5,0] for _ in range(1001)]}
        entry={'source_image':'images/val/dense.jpg','detections':1001}
        for method in ('yolo800_repeat','yolo1280','yolo1536'):
            with self.assertRaisesRegex(ValueError,'cap exceeded'):
                inventory.validate_p2_rows(body,rec,method,entry)
        with self.assertRaisesRegex(ValueError,'cap exceeded'):inventory.validate_rows(body,rec)
        with self.assertRaisesRegex(ValueError,'Unknown P2'):
            inventory.validate_p2_rows(body,rec,'unrecognized',entry)

    def test_p2_source_adapter_reads_long_sahi_and_checks_source_hash(self):
        import coco_runner as runner
        rec={'id':1,'image':'images/val/dense.jpg','width':100,'height':100,'image_sha256':'b'*64}
        protocol=runner.read(runner.HERE/'references/p2_val/official_val_prediction_index.json')['protocol_sha256']
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);src=object.__new__(runner.Sources)
            src.p2=root/'p2';src.old=root/'old';src.p2.mkdir();src.old.mkdir()
            src.pidx={}
            long_rows=[[1,1,2,2,.5,0] for _ in range(1018)]
            for method in ('yolo800_repeat','yolo1280','yolo1536','sahi640','sahi800'):
                rows=long_rows if method=='sahi640' else []
                path=src.p2/(method+'.json')
                runner.write(path,{'image_sha256':rec['image_sha256'],'protocol_sha256':protocol,'detections':rows})
                src.pidx[method]={rec['image']:{'source_image':rec['image'],'path':path.name,
                                              'sha256':runner.sha(path),'detections':len(rows)}}
            legacy={'image_sha256':rec['image_sha256'],'protocol_sha256':'legacy','baseline':[],'hybrid_a025':[]}
            legacy['content_sha256']=runner.fp(legacy)
            path=src.old/'legacy.json';runner.write(path,legacy)
            src.oidx={rec['image']:{'path':path.name,'sha256':runner.sha(path)}}
            src.frozen={'original_protocol_content_sha256':'legacy'}
            before={p:p.read_bytes() for p in root.rglob('*.json')}
            answer=src.get('val',rec)
            self.assertEqual(answer['sahi640'],long_rows)
            self.assertEqual({p:p.read_bytes() for p in before},before)
            broken=src.p2/'sahi640.json';broken.write_bytes(broken.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'identity differs'):src.get('val',rec)

    def test_coco_keeps_sahi_tail_before_per_category_caps(self):
        # The minority-class TP appears beyond global rank 1000. It must
        # survive the archive adapter and COCO's separate per-category caps.
        rec={'id':1,'width':100,'height':100}
        xyxy=[[0,0,5,5,.9,0] for _ in range(1000)]+[[30,30,35,35,.5,1] for _ in range(18)]
        dt=v.xyxy_to_det(xyxy).tolist()
        results=self.equal([rec],[[[70,70,5,5,1],[30,30,5,5,2]]],[dt])
        for cap in ('100','500'):
            cls=next(x for x in results[cap]['per_class'] if x['category_id']==2)
            self.assertAlmostEqual(cls['AP'],100)

if __name__=='__main__':unittest.main()
