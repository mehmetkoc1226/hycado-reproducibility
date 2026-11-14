import contextlib,copy,io,json,sqlite3,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from pycocotools.cocoeval import COCOeval
import coco_core as c
import uncertainty_core as u
import recall_core as r
import resampling as b

def compiled(records,ground,det):
    entries={cat:{} for cat in c.CATS}
    for rec,g,d in zip(records,ground,det):
        for e in c.one_image(rec,g,d):
            row,h,payload=c.encode(e);decoded=c.decode(h,payload)
            if row['area_index']==0:entries[row['category_id']][rec['id']]=decoded
    return [u.compile_category(entries[cat],[x['id'] for x in records]) for cat in c.CATS],entries

def literal_reference(records,ground,det,frequency):
    images=[];ga=[];da=[]
    for rec,g,d,n in zip(records,ground,det,frequency):
        for _ in range(int(n)):
            image=len(images)+1;images.append({'id':image,'width':rec['width'],'height':rec['height']})
            for x in g:ga.append({'id':len(ga)+1,'image_id':image,'category_id':int(x[4]),'bbox':x[:4],'area':x[2]*x[3],'iscrowd':0})
            for x in d:da.append({'id':len(da)+1,'image_id':image,'category_id':int(x[5]),'bbox':x[:4],'area':x[2]*x[3],'iscrowd':0,'score':x[4]})
    e=COCOeval(c.coco_object(images,ga),c.coco_object(images,da),'bbox');e.params.maxDets=[1,10,100]
    with contextlib.redirect_stdout(io.StringIO()):e.evaluate();e.accumulate()
    p=e.eval['precision'];rr=e.eval['recall']
    vals=[c.mean(p[:,:,:,0,2]),c.mean(p[0,:,:,0,2]),c.mean(p[5,:,:,0,2]),c.mean(rr[:,:,0,2])]
    return np.asarray([np.nan if v is None else v for v in vals])

class AnalysisTests(unittest.TestCase):
    def fixture(self):
        recs=[{'id':i,'width':300,'height':300} for i in (1,2,3)]
        gt=[[[10,10,20,20,1]],[[50,50,40,40,2]],[]]
        dt=[[[10,10,20,20,.5,1],[0,0,5,5,.5,1]],
            [[50,50,40,40,.7,2],[50,50,40,40,.7,2]],[[0,0,5,5,.6,1]]]
        return recs,gt,dt

    def test_original_point_equals_upstream_COCO(self):
        rec,g,d=self.fixture();engine,_=compiled(rec,g,d);f=np.ones(3,dtype=np.int64)
        np.testing.assert_allclose(u.evaluate([engine],f)[0],literal_reference(rec,g,d,f),rtol=0,atol=1e-10)

    def test_literal_resampling_preserves_score_ties(self):
        rec,g,d=self.fixture();engine,_=compiled(rec,g,d)
        for f in [np.array([2,1,0]),np.array([3,0,2]),np.array([0,2,1])]:
            np.testing.assert_allclose(u.evaluate([engine],f)[0],literal_reference(rec,g,d,f),rtol=0,atol=1e-10)
        block=engine[0]
        ids=b.repeated_indices(block,np.array([2,0,0],dtype=np.int64))
        self.assertEqual(block['flags'][ids,0].tolist(),[1,0,1,0])

    def test_random_multiclass_cluster_frequencies_equal_upstream(self):
        rng=np.random.default_rng(20261113);rec=[{'id':i+1,'width':200,'height':200} for i in range(7)]
        gt=[];dt=[]
        for _ in rec:
            g=[];d=[]
            for j in range(12):
                x,y=rng.uniform(0,130,2);w,h=rng.uniform(5,50,2);cls=j%4+1
                g.append([x,y,w,h,cls]);d.append([x+float(rng.uniform(0,5)),y,w,h,float(rng.choice([.3,.5,.7])),cls])
            d.extend([[0,0,3,3,.5,1],[100,100,10,10,.7,2]])
            gt.append(g);dt.append(d)
        engine,_=compiled(rec,gt,dt)
        for _ in range(8):
            f=rng.integers(0,4,len(rec));f[0]=1
            np.testing.assert_allclose(u.evaluate([engine],f)[0],literal_reference(rec,gt,dt,f),rtol=0,atol=1e-10)

    def test_empty_predictions_and_absent_categories(self):
        rec=[{'id':1,'width':200,'height':200},{'id':2,'width':200,'height':200}]
        g=[[[0,0,10,10,1]],[]];d=[[],[[0,0,10,10,.5,2]]]
        engine,_=compiled(rec,g,d)
        np.testing.assert_allclose(u.evaluate([engine],np.array([2,1]))[0],[0,0,0,0],atol=1e-12)
        self.assertTrue(np.isnan(u.evaluate([engine],np.array([0,1]))).all())

    def test_cap100_and_minority_tail_are_not_global_truncated(self):
        rec=[{'id':1,'width':5000,'height':100}]
        g=[[[i*25,0,10,10,1] for i in range(140)]+[[0,50,10,10,2]]]
        d=[[[i*25,0,10,10,1-i/1000,1] for i in range(140)]+[[0,50,10,10,.1,2]]]
        engine,_=compiled(rec,g,d);f=np.array([2])
        np.testing.assert_allclose(u.evaluate([engine],f)[0],literal_reference(rec,g,d,f),rtol=0,atol=1e-10)

    def test_ignored_flags_equal_upstream_accumulate(self):
        rec,g,d=self.fixture();engine,entries=compiled(rec,g,d)
        e=entries[1][1];e=copy.deepcopy(e);e['dtIgnore'][:,1]=True
        entries[1][1]=e;f=np.array([2,0,1])
        block=u.compile_category(entries[1],[1,2,3]);actual=u.category_values(block,f)
        clones={};ids=[]
        for old,n in zip([1,2,3],f):
            for _ in range(int(n)):
                k=len(ids)+1;ids.append(k)
                if old in entries[1]:clones[k]=copy.deepcopy(entries[1][old])
        p,rr=c.accumulate_one(ids,1,0,clones)
        expected=[c.mean(p[:,:,2]),c.mean(p[0,:,2]),c.mean(p[5,:,2]),c.mean(rr[:,2])]
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10)

    def test_frozen_groups_and_draws_match_existing_scheme(self):
        import run_analysis as runner
        records=sorted(runner.read(runner.HERE/'references/dataset_manifest.json')['records'],key=lambda x:x['image'])
        groups=runner.read(runner.HERE/'references/bootstrap_groups.json')
        self.assertEqual(b.make_groups(records),groups);self.assertEqual(len(groups['labels']),61)
        f=b.frequencies(groups,'cluster',4)
        np.testing.assert_array_equal(f,b.frequencies(groups,'cluster',4))
        ids=np.asarray(groups['image_to_group'])
        for k in range(61):self.assertEqual(len(np.unique(f[ids==k])),1)
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261102,0,4])))
        expected=np.bincount(rng.integers(0,61,size=61),minlength=61)[ids]
        np.testing.assert_array_equal(f,expected)

    def test_recall_uses_whole_score_ties(self):
        block=r.compile_curve([.9,.9],[[1,1],[0,0]],[0,0],[1,0])
        values,curves=r.curve_values(block,np.ones(2,dtype=np.int64),full=True)
        self.assertEqual(curves[0]['TP'],[0,1]);self.assertEqual(curves[0]['FP'],[0,1])
        self.assertEqual(values[0,0],100)
        block=r.compile_curve([.9,.9],[[1,1],[0,0]],[0,0],[1])
        values,_=r.curve_values(block,np.ones(1,dtype=np.int64))
        self.assertEqual(values[0,0],0);self.assertEqual(values[0,1],100)

    def test_weighted_recall_equals_literal_replication(self):
        scores=np.array([.9,.5,.9,.3,.5]);flags=np.array([[1,1],[0,-1],[0,0],[1,0],[1,1]],dtype=np.int8)
        images=np.array([0,0,1,1,2]);counts=np.array([1,1,1]);frequency=np.array([2,0,3])
        block=r.compile_curve(scores,flags,images,counts);actual,_=r.curve_values(block,frequency)
        expanded=[];ef=[];ei=[];ec=[]
        for i,n in enumerate(frequency):
            for _ in range(n):
                idx=np.flatnonzero(images==i);expanded.extend(scores[idx]);ef.extend(flags[idx]);ei.extend([len(ec)]*len(idx));ec.append(counts[i])
        expanded=r.compile_curve(expanded,ef,ei,ec)
        expected,_=r.curve_values(expanded,np.ones(len(ec),dtype=np.int64))
        np.testing.assert_array_equal(actual,expected)

    def test_empty_recall_curve_is_zero_when_targets_exist(self):
        block=r.compile_curve([],np.empty((0,2),dtype=np.int8),[],[1,1])
        values,curves=r.curve_values(block,np.ones(2,dtype=np.int64),full=True)
        np.testing.assert_array_equal(values,np.zeros((2,5)));self.assertEqual(curves[0]['scores'],[None])

    def test_draw_cache_resume_and_corruption_guards(self):
        import run_analysis as runner
        with tempfile.TemporaryDirectory() as f:
            path=Path(f)/'state.sqlite3'
            with contextlib.closing(runner.open_state(path,'sig')) as db:
                self.assertIsNone(runner.get_draw(db,0));runner.save_draw(db,0,{'a':[1,2]})
            with contextlib.closing(runner.open_state(path,'sig')) as db:
                self.assertEqual(runner.get_draw(db,0),{'a':[1,2]})
                with db:db.execute("UPDATE draws SET payload='{}' WHERE draw=0")
                with self.assertRaisesRegex(ValueError,'payload changed'):runner.get_draw(db,0)
            # A rejected connection is explicitly closed by open_state.
            with self.assertRaisesRegex(ValueError,'different inputs'):runner.open_state(path,'wrong')

    def test_coco_cache_readonly_loader_and_coverage_guard(self):
        import run_analysis as runner
        rec,g,d=self.fixture();_,entries=compiled(rec,g,d);ids=[1,2,3]
        with tempfile.TemporaryDirectory() as f:
            path=Path(f)/'matches.sqlite3'
            with contextlib.closing(sqlite3.connect(path)) as db:
                db.execute('CREATE TABLE binding(id INTEGER PRIMARY KEY,signature TEXT)');db.execute('INSERT INTO binding VALUES(1,?)',(runner.fp({'signature':'source','mode':'full'}),))
                db.execute('CREATE TABLE entries(split TEXT,method TEXT,image_id INTEGER,category INTEGER,area INTEGER,digest TEXT,payload BLOB)')
                db.execute('CREATE TABLE done(split TEXT,method TEXT,image_id INTEGER,marker TEXT)')
                for method in runner.METHODS:
                    for image in ids:
                        for cat in c.CATS:
                            e=entries[cat].get(image)
                            if e is None:continue
                            e=copy.deepcopy(e);e['aRng']=c.AREAS[0]
                            _,h,payload=c.encode(e);db.execute('INSERT INTO entries VALUES(?,?,?,?,?,?,?)',('testdev',method,image,cat,0,h,payload))
                        parts=list(db.execute('SELECT category,area,digest FROM entries WHERE method=? AND image_id=? ORDER BY category,area',(method,image)))
                        db.execute('INSERT INTO done VALUES(?,?,?,?)',('testdev',method,image,runner.fp(parts)))
                db.commit()
            before=path.read_bytes()
            with contextlib.closing(runner.readonly(path)) as db:
                runner.validate_coco_cache(db,'source',ids)
                blocks=[u.compile_category(runner.load_category(db,runner.METHODS[0],cat,ids),ids) for cat in c.CATS]
                np.testing.assert_allclose(u.evaluate([blocks],np.ones(3,dtype=np.int64))[0],literal_reference(rec,g,d,[1,1,1]),rtol=0,atol=1e-10)
                with self.assertRaises(sqlite3.OperationalError):db.execute('DELETE FROM entries')
            self.assertEqual(path.read_bytes(),before)
            with contextlib.closing(sqlite3.connect(path)) as db:
                with db:db.execute('DELETE FROM entries WHERE category=1 AND image_id=1')
            with contextlib.closing(runner.readonly(path)) as db:
                with self.assertRaisesRegex(ValueError,'coverage changed'):runner.validate_coco_cache(db,'source',ids)

    def test_exported_intervals_and_curves_have_expected_coverage(self):
        import run_analysis as runner
        point=np.array([[20.,30.,21.,40.],[20.4,30.5,21.3,43.],[20.45,30.55,21.35,43.1]])
        rp=np.zeros((3,2,5));curves=[]
        for _ in range(3):
            curves.append([{'scores':[None,.5],'TP':[0,1],'FP':[0,1],'images':2,'normal_GT':2,'recall_percent':[0.,50.],'FP_per_image':[0.,.5]} for _ in range(2)])
        samples=np.stack([point+i*.01 for i in range(8)]);recall=np.stack([rp for _ in range(8)])
        with tempfile.TemporaryDirectory() as f,patch.object(runner,'DRAWS',8):
            out=Path(f);ci=runner.finish(point,rp,samples,recall,curves,out)
            self.assertAlmostEqual(ci['differences_pp']['hybrid1280_minus_yolo1280']['AP']['lower95'],.4)
            self.assertEqual(len(ci['recall_at_FP']),2)
            for name in ['intervals.json','recall_curves.csv','recall_budgets.csv','bootstrap_draws.csv']:self.assertTrue((out/name).is_file())

    def test_frozen_matching_SQLite_adapter_and_count_parity(self):
        import hashlib
        import run_analysis as runner
        rec=runner.read(runner.HERE/'references/dataset_manifest.json')['records'][0]
        labels={x['stem']:x['sha256'] for x in runner.read(runner.HERE/'references/annotation_index.json')['records']}
        original_read=runner.read
        expected={m:{'counts_at_fixed_max500':{t:{'TP':1,'FP':0,'ignored_detections':0} for t in ['0.5','0.75']}} for m in runner.METHODS}
        def read_fixture(path):
            return {'methods':expected} if Path(path).name=='stage20_metrics.json' else original_read(path)
        part={'gt_denominators':np.array([74707]+[0]*9,dtype=np.int64)}
        for m in runner.METHODS:
            part[m+'_scores']=np.array([.9]);part[m+'_classes']=np.array([1],dtype=np.int8)
            part[m+'_flags']=np.ones((1,10),dtype=np.int8)
        stream=io.BytesIO();np.savez_compressed(stream,**part);payload=stream.getvalue()
        with tempfile.TemporaryDirectory() as f:
            path=Path(f)/'frozen.sqlite3'
            with contextlib.closing(sqlite3.connect(path)) as db:
                db.execute('CREATE TABLE matching(stem TEXT PRIMARY KEY,ann_sha TEXT,digest TEXT,payload BLOB)')
                db.execute('INSERT INTO matching VALUES(?,?,?,?)',(rec['stem'],labels[rec['stem']],hashlib.sha256(payload).hexdigest(),payload));db.commit()
            before=path.read_bytes()
            with contextlib.closing(runner.readonly(path)) as db,patch.object(runner,'read',read_fixture),contextlib.redirect_stdout(io.StringIO()):
                engines=runner.compile_recall(db,[rec])
                self.assertEqual(len(engines),3)
                values,_=r.curve_values(engines[0],np.ones(1,dtype=np.int64))
                np.testing.assert_allclose(values,100/74707,rtol=0,atol=1e-12)
            self.assertEqual(path.read_bytes(),before)

if __name__=='__main__':unittest.main()
