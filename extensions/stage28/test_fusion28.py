"""Synthetic tests; no project files, GPU, model weights or network access."""
import contextlib,copy,io,json,math,sqlite3,tempfile,unittest,zlib
from pathlib import Path
import numpy as np
import fusion28 as f
import coco_core as c
import uncertainty_core as u
from pycocotools.cocoeval import COCOeval

def literal_nms(rows,width,height,threshold=.7,limit=1000):
    ordered=f.canonical(rows,width,height);kept=[]
    def iou(a,b):
        inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
    for row in ordered:
        if not any(row[5]==k[5] and iou(row,k)>threshold for k in kept):kept.append(row)
        if len(kept)==limit:break
    return kept

def upstream(data,cap=100):
    images=[];gts=[];dts=[];g=0;d=0
    for rec,gt,dt in data:
        images.append({'id':rec['id'],'width':rec['width'],'height':rec['height']})
        for row in gt:
            g+=1;gts.append({'id':g,'image_id':rec['id'],'category_id':int(row[4]),'bbox':row[:4],
                             'area':row[2]*row[3],'iscrowd':0})
        for row in dt:
            d+=1;dts.append({'id':d,'image_id':rec['id'],'category_id':int(row[5]),'bbox':row[:4],
                             'area':row[2]*row[3],'score':row[4],'iscrowd':0})
    e=COCOeval(c.coco_object(images,gts),c.coco_object(images,dts),'bbox')
    e.params.maxDets=[1,10,cap]
    with contextlib.redirect_stdout(io.StringIO()):e.evaluate();e.accumulate()
    p=e.eval['precision'][:,:,:,0,-1];r=e.eval['recall'][:,:,0,-1]
    return np.array([c.mean(p),c.mean(p[0]),c.mean(p[5]),c.mean(r)])

class FusionTests(unittest.TestCase):
    def test_fusion_native_coordinates_and_scores_unchanged(self):
        a=[[0,0,20,20,.9,0],[70,70,100,100,.2,1]]
        b=[[0,0,20,20,.8,0],[110,100,130,120,.7,0]]
        old=copy.deepcopy([a,b]);r=f.fuse(a,b,200,200)
        self.assertEqual([a,b],old)
        self.assertEqual(r,[[0.,0.,20.,20.,.9,0.],[110.,100.,130.,120.,.7,0.],[70.,70.,100.,100.,.2,1.]])

    def test_nms_strict_boundary_and_class_isolation(self):
        a=[0,0,17,10,.9,0];b=[3,0,20,10,.8,0] # IoU exactly 14/20=.7: both survive.
        self.assertEqual(len(f.nms([a,b],100,100)),2)
        b[0]=2.999;self.assertEqual(len(f.nms([a,b],100,100)),1)
        b[5]=1;self.assertEqual(len(f.nms([a,b],100,100)),2)

    def test_stable_ties_clipping_and_global_cap(self):
        rows=[[5,5,15,15,.5,1],[-10,-10,10,10,.5,0],[20,20,20,30,.8,0]]
        self.assertEqual(f.nms(rows,100,100,limit=1),[[0.,0.,10.,10.,.5,0.]])
        self.assertEqual(f.fuse([],[],100,100),[])

    def test_randomized_nms_matches_independent_literal(self):
        rng=np.random.default_rng(42)
        for _ in range(40):
            rows=[]
            for i in range(80):
                xy=rng.uniform(-10,95,2);wh=rng.uniform(1,50,2)
                rows.append([*xy,*(xy+wh),float(rng.integers(0,6)/5),int(rng.integers(0,3))])
            self.assertEqual(f.nms(rows,100,100,limit=35),literal_nms(rows,100,100,limit=35))

    def test_validation_rejects_nonfinite_class_geometry_and_cap(self):
        rec={'width':100,'height':100}
        for rows in [[[0,0,10,10,float('nan'),0]],[[0,0,10,10,.5,10]],[[0,0,0,10,.5,0]],[[0,0,101,10,.5,0]]]:
            with self.assertRaises(ValueError):f.validate_rows(rows,rec)
        with self.assertRaises(ValueError):f.validate_rows([[0,0,10,10,.5,0]]*1001,rec)

    def test_readonly_missing_and_connection_released(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'source.sqlite3'
            with self.assertRaises(ValueError):f.readonly(p)
            self.assertFalse(p.exists())
            with contextlib.closing(sqlite3.connect(p)) as db:
                db.execute('CREATE TABLE x(v INTEGER)');db.commit()
            before=f.sha(p)
            with contextlib.closing(f.readonly(p)) as db:
                with self.assertRaises(sqlite3.OperationalError):db.execute('INSERT INTO x VALUES(1)')
            self.assertEqual(f.sha(p),before);p.unlink() # Fails on Windows if a handle leaked.

    def test_source_payload_digest_and_image_binding(self):
        rec={'stem':'one','width':100,'height':100,'image_sha256':'a'}
        body={'image_sha256':'a','detections':[[0,0,10,10,.5,0]]}
        raw=json.dumps(body).encode();h=__import__('hashlib').sha256(raw).hexdigest()
        with contextlib.closing(sqlite3.connect(':memory:')) as db:
            db.execute('CREATE TABLE prediction(method TEXT,stem TEXT,digest TEXT,payload BLOB)')
            db.execute('INSERT INTO prediction VALUES(?,?,?,?)',('yolo800','one',h,zlib.compress(raw)))
            self.assertEqual(f.source_body(db,'yolo800',rec,h),body['detections'])
            with self.assertRaises(ValueError):f.source_body(db,'yolo800',rec,'wrong')
            with self.assertRaises(ValueError):f.source_body(db,'yolo800',dict(rec,image_sha256='b'),h)

    def test_state_resume_signature_and_marker_tampering(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'state.sqlite3'
            with contextlib.closing(f.state(p,'fixed')) as db:
                rec={'id':1,'width':100,'height':100}
                entries=c.one_image(rec,[[0,0,10,10,1]],[[0,0,10,10,.9,1]])
                f.save_entries(db,'fusion',rec,entries);self.assertTrue(f.entries_done(db,'fusion',1))
            with contextlib.closing(f.state(p,'fixed')) as db:
                self.assertTrue(f.entries_done(db,'fusion',1));db.execute("UPDATE done SET marker='bad'");db.commit()
                with self.assertRaises(ValueError):f.entries_done(db,'fusion',1)
            with self.assertRaises(ValueError):f.state(p,'changed')
            p.unlink()

    def test_packed_prediction_and_coco_payload_tamper(self):
        h,b=f.pack({'detections':[]});self.assertEqual(f.unpack(h,b),{'detections':[]})
        with self.assertRaises(ValueError):f.unpack('bad',b)
        entry=c.one_image({'id':1,'width':100,'height':100},[[0,0,10,10,1]],[])[0]
        d,h,b=c.encode(entry)
        with self.assertRaises(ValueError):c.decode(h,b+b'bad')

    def test_coco_caps_and_empty_images_match_upstream(self):
        data=[({'id':1,'width':200,'height':200},[[150,150,10,10,1]],
               [[i,0,1,1,.9-i*.0001,1] for i in range(105)]+[[150,150,10,10,.1,1]]),
              ({'id':2,'width':200,'height':200},[],[])]
        entries={}
        for rec,gt,dt in data:
            for e in c.one_image(rec,gt,dt):
                d,h,b=c.encode(e);entries[(rec['id'],d['category_id'],d['area_index'])]=c.decode(h,b)
        values=c.aggregate([1,2],lambda cat,area:{im:e for (im,k,a),e in entries.items() if k==cat and a==area})[0]
        for cap in [100,500]:
            got=[values[str(cap)][k] for k in ['AP','AP50','AP75','AR']]
            np.testing.assert_allclose(got,upstream(data,cap),rtol=0,atol=1e-10)
        self.assertLess(values['100']['AP'],values['500']['AP'])

    def test_bootstrap_equal_score_image_blocks_match_explicit_replication(self):
        data=[({'id':1,'width':100,'height':100},[[0,0,10,10,1]],[[0,0,10,10,.5,1],[50,50,10,10,.5,1]]),
              ({'id':2,'width':100,'height':100},[[20,20,10,10,1]],[[50,50,10,10,.5,1],[20,20,10,10,.5,1]]),
              ({'id':3,'width':100,'height':100},[],[[50,50,10,10,.5,1]])]
        entries={}
        for rec,gt,dt in data:
            for e in c.one_image(rec,gt,dt):
                d,h,b=c.encode(e)
                if d['category_id']==1 and d['area_index']==0:entries[rec['id']]=c.decode(h,b)
        block=u.compile_category(entries,[1,2,3]);frequency=np.array([2,1,2])
        got=u.evaluate([[block]],frequency)[0];expanded=[]
        for item,count in zip(data,frequency):
            for _ in range(count):expanded.append((dict(item[0],id=len(expanded)+1),item[1],item[2]))
        np.testing.assert_allclose(got,upstream(expanded),rtol=0,atol=1e-10)

    def test_ignore_preprocessing_is_same_for_gt_and_detections(self):
        import visdrone_preprocess as v
        raw=v.raw_annotation('1,1,30,30,0,0,0,0\n2,2,10,10,1,1,0,0\n60,60,10,10,1,1,0,0')
        integral=v.ignore_integral(raw,100,100)
        gt=v.prepare_groundtruth(raw,integral);self.assertEqual(len(gt),1)
        dt=v.outside_ignore(v.xyxy_to_det([[2,2,12,12,.9,0],[60,60,70,70,.8,0]]),integral)
        self.assertEqual(len(dt),1);self.assertEqual(dt[0,0],60)

    def test_retained_draws_groups_and_annotation_names(self):
        import run_fusion28 as r
        import resampling as b
        rows=r.records();groups=f.read(r.REF/'bootstrap_groups.json')
        self.assertEqual(b.make_groups(rows),groups)
        self.assertEqual(len(groups['labels']),61)
        old=r.archived_draws();self.assertEqual(old['sahi640'].shape,(1000,4))
        np.testing.assert_array_equal(b.frequencies(groups,'cluster',12),b.frequencies(groups,'cluster',12))

    def test_atomic_write_and_lock_reusable_after_exit(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'result.json';f.write(path,{'a':1});f.write(path,{'a':2})
            self.assertEqual(f.read(path),{'a':2});self.assertFalse(path.with_name(path.name+'.partial').exists())
            with f.run_lock(Path(t)/'run.lock'):pass
            with f.run_lock(Path(t)/'run.lock'):pass

    def test_end_to_end_freeze_matching_bootstrap_and_resume(self):
        import run_fusion28 as r
        import resampling as rs
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);out=root/'out';out.mkdir();refs=root/'refs';refs.mkdir();ann=root/'annotations';ann.mkdir()
            rows=[];outputs={};pred=[]
            for i in range(1,5):
                stem=f'{i:07d}_00001_d_0000001';text='10,10,10,10,1,1,0,0\n'
                p=ann/(stem+'.txt');p.write_text(text)
                rec={'id':i,'image':'images/test-dev/'+stem+'.jpg','stem':stem,'width':100,'height':100,
                     'image_sha256':str(i)*64,'annotation':p.name,'annotation_sha256':f.sha(p)}
                rows.append(rec)
                y8=[[10,10,20,20,.4,0]] if i%2 else []
                y12=[[50,50,60,60,.8,0],[10,10,20,20,.6,0]] if i!=4 else [[50,50,60,60,.8,0]]
                hy=y12+([[10,10,20,20,.1,0]] if i==4 else [])
                outputs[rec['stem']]={'yolo800':y8,'yolo1280':y12,'hybrid1280':hy}
                pred.append(f.fuse(y8,y12,100,100))
            class FakeSource:
                indices={18:{rr['stem']:{'variant_sha256':{m:f.fp(outputs[rr['stem']][m]) for m in ['yolo800','yolo1280']}} for rr in rows}}
                def get(self,stage,method,rec):return outputs[rec['stem']][method]
            src=FakeSource();groups=rs.make_groups(rows);f.write(refs/'bootstrap_groups.json',groups)
            dbpath=out/'fusion_and_metrics.sqlite3'
            with contextlib.closing(f.state(dbpath,'test')) as db,contextlib.redirect_stdout(io.StringIO()):
                token=r.frozen_predictions(db,src,rows,'test',out)
                self.assertEqual(r.measure(db,src,rows,ann,out),4)
                allowed={1,2,3,4};point={m:c.aggregate([1,2,3,4],lambda k,a,m=m:f.load_entries(db,m,k,a,allowed))[0] for m in r.ALL}
                f.write(refs/'22_point_results.json',{'metrics':{'testdev':point}})
                f.write(refs/'24_metrics.json',{'sahi640_coco':point['yolo1280']})
                engines={m:[[u.compile_category(f.load_entries(db,m,k,0,allowed),[1,2,3,4]) for k in range(1,11)]] for m in r.BASE}
                for n in ['24_bootstrap_draws.csv','23_bootstrap_draws.csv']:
                    with (refs/n).open('w',newline='') as stream:
                        import csv
                        writer=csv.writer(stream);writer.writerow(['draw','method']+r.KEYS)
                        for i in range(1000):
                            freq=rs.frequencies(groups,'cluster',i)
                            vals={m:u.evaluate(e,freq)[0] for m,e in engines.items()};vals['sahi640']=vals['yolo1280']
                            for m in ['yolo1280','hybrid1280','sahi640']:writer.writerow([i,m,*vals[m]])
                with patch.object(r,'REF',refs):
                    pt=r.points(db,rows,out);result=r.bootstrap(db,rows,pt,out)
                self.assertEqual(result['replicates'],1000)
                original=(out/'bootstrap_draws.csv').read_bytes()
            # Reopen all handles and resume each phase without recomputing matches/draws.
            with contextlib.closing(f.state(dbpath,'test')) as db,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(r.frozen_predictions(db,src,rows,'test',out),token)
                self.assertEqual(r.measure(db,src,rows,ann,out),4)
                with patch.object(r,'REF',refs):
                    pt=r.points(db,rows,out);again=r.bootstrap(db,rows,pt,out)
                self.assertEqual(result,again);self.assertEqual((out/'bootstrap_draws.csv').read_bytes(),original)
            dbpath.unlink() # Every SQLite handle must be closed, including on Windows.

if __name__=='__main__':unittest.main(verbosity=2)
