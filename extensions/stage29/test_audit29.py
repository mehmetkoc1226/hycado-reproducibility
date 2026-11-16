import contextlib,hashlib,io,json,sqlite3,tempfile,unittest
from pathlib import Path
import numpy as np
import frozen_cpu as f
import trace29 as t
import audit29 as a

def raw_fixture(rects):
    q=np.zeros((len(rects),300,14),dtype=np.float32)
    for i in range(len(rects)):
        q[i,0,:4]=[.5,.5,1,1];q[i,0,4]=.4
        q[i,1,:4]=[.5,.5,.2,.2];q[i,1,5]=.8
        q[i,2,:4]=[-1,-1,.1,.1];q[i,2,4]=.8 # Clips to zero area; removed.
        q[i,3,:4]=[.5,.5,.1,.1];q[i,3,4:6]=[.5,.5] # Class tie: first index.
    return q

def literal_nms(rows):
    ans=[]
    for r in sorted(rows,key=lambda x:(-x[4],x[5],*x[:4])):
        bad=False
        for k in ans:
            inter=max(0,min(r[2],k[2])-max(r[0],k[0]))*max(0,min(r[3],k[3])-max(r[1],k[1]))
            area=(r[2]-r[0])*(r[3]-r[1])+(k[2]-k[0])*(k[3]-k[1])-inter
            if r[5]==k[5] and inter/area>.7:bad=True;break
        if not bad:ans.append(r)
        if len(ans)==1000:break
    return ans

def independent_decode(q,rect):
    result=[]
    for row in np.asarray(q,dtype=np.float64):
        cl=int(np.argmax(row[4:]));score=float(row[4+cl])
        if score<=.001:continue
        x,y,xx,yy=rect;cx,cy,w,h=map(float,row[:4])
        values=[max(0,min(1,cx-w/2))*(xx-x)+x,max(0,min(1,cy-h/2))*(yy-y)+y,
                max(0,min(1,cx+w/2))*(xx-x)+x,max(0,min(1,cy+h/2))*(yy-y)+y,score*.25,cl]
        if values[2]>values[0] and values[3]>values[1]:result.append(values)
    return result

class AuditTests(unittest.TestCase):
    def test_decode_matches_independent_retained_math(self):
        rect=[10,20,90,100];q=raw_fixture([rect])[0]
        rows,ids=t.decode(q,rect);rows[:,4]*=.25
        self.assertEqual(rows.tolist(),independent_decode(q,rect));self.assertEqual(ids.tolist(),[0,1,3])
        self.assertEqual(int(rows[-1,5]),0)

    def test_strict_raw_score_and_zero_area_filter(self):
        q=np.zeros((300,14),np.float64);q[:3,:4]=[.5,.5,.2,.2]
        q[0,4]=.001;q[1,4]=.00100000001;q[2,4]=.5;q[2,2:4]=0
        rows,ids=t.decode(q,[0,0,100,100]);self.assertEqual(ids.tolist(),[1])

    def test_trace_matches_literal_nms_and_preserves_exact_tie_origin(self):
        rows=np.array([[0,0,10,10,.2,0],[0,0,10,10,.2,0],[0,0,10,10,.2,1],[20,20,30,30,.1,0]],float)
        origins=np.array([[0,-1,0],[1,0,0],[1,0,1],[1,0,2]],np.int32)
        ids,status,by=t.trace(rows,origins,100,100)
        self.assertEqual(rows[ids].tolist(),literal_nms(rows.tolist()))
        self.assertEqual(ids.tolist(),[0,2,3]);self.assertEqual(status[1],2);self.assertEqual(by[1],0)

    def test_random_trace_parity(self):
        rng=np.random.default_rng(29)
        for _ in range(20):
            xy=rng.uniform(0,60,(70,2));wh=rng.uniform(1,35,(70,2))
            rows=np.column_stack((xy,xy+wh,rng.integers(0,5,70)/4,rng.integers(0,3,70)))
            origins=np.column_stack((np.ones(70,int),np.zeros(70,int),np.arange(70)))
            ids,status,by=t.trace(rows,origins,100,100)
            self.assertEqual(rows[ids].tolist(),literal_nms(rows.tolist()))

    def test_ratio_histogram_edges_empty_groups_and_display_filter(self):
        rec={'width':100,'height':100,'stem':'one','image':'one.jpg'}
        rects=[[0,0,100,100]];raw=raw_fixture(rects);base=[]
        expected=t.replay(raw,rects,rects,base,rec)
        s=t.image_summary(rec,raw,rects,rects,base,expected)
        g=s['distributions']['retained_crop'];self.assertEqual(g['count'],3)
        self.assertEqual(g['ratio_ge_0_95'],1);self.assertEqual(sum(g['area_ratio_histogram']),3)
        self.assertEqual(g['touches_all_four_crop_edges'],1)
        self.assertLessEqual(g['max_scaled_score'],.25)
        empty=t.group_stats([],[],[]);self.assertEqual(empty['count'],0);self.assertEqual(sum(empty['area_ratio_histogram']),0)

    def test_invalid_plans_rejected(self):
        rec={'width':100,'height':100}
        for plan in [[[0,0,0,10]],[[-1,0,10,10]],[[0.,0,10,10]],[[0,0,10,10],[0,0,10,10]]]:
            with self.assertRaises(ValueError):t.plan(plan,rec)

    def test_raw_shape_hash_union_and_probability_tampering(self):
        rects=[[0,0,100,100]];raw=raw_fixture(rects)
        stream=io.BytesIO();np.savez_compressed(stream,rects=np.array(rects,np.int32),queries=raw);payload=stream.getvalue();h=hashlib.sha256(payload).hexdigest()
        with contextlib.closing(sqlite3.connect(':memory:')) as db:
            db.execute('CREATE TABLE crop_raw(stem TEXT,digest TEXT,payload BLOB)');db.execute('INSERT INTO crop_raw VALUES(?,?,?)',('one',h,payload))
            rec={'stem':'one'};np.testing.assert_array_equal(t.raw_queries(db,rec,rects,h),raw)
            with self.assertRaises(ValueError):t.raw_queries(db,rec,rects,'bad')
            with self.assertRaises(ValueError):t.raw_queries(db,rec,[[0,0,50,50]],h)
            with self.assertRaises(ValueError):t.raw_queries(db,{'stem':'missing'},rects,h)
        raw[0,0,4]=1.1
        with self.assertRaises(ValueError):t.decode(raw[0],rects[0])

    def test_readonly_database_handle_released_on_windows(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'source.sqlite3'
            with self.assertRaises(ValueError):f.readonly(path)
            self.assertFalse(path.exists())
            with contextlib.closing(sqlite3.connect(path)) as db:db.execute('CREATE TABLE x(v)');db.commit()
            h=f.sha(path)
            with contextlib.closing(f.readonly(path)) as db:
                with self.assertRaises(sqlite3.OperationalError):db.execute('INSERT INTO x VALUES(1)')
            self.assertEqual(h,f.sha(path));path.unlink()

    def test_full_audit_summary_resume_and_source_binding(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'out';out.mkdir();paths=[root/'18.sqlite3',root/'20.sqlite3']
            rows=[];idx18={};idx20={}
            with contextlib.ExitStack() as stack:
                db18,db20=[stack.enter_context(contextlib.closing(sqlite3.connect(p))) for p in paths]
                for db in [db18,db20]:db.execute('CREATE TABLE prediction(method TEXT,stem TEXT,digest TEXT,payload BLOB,PRIMARY KEY(method,stem))')
                db20.execute('CREATE TABLE crop_raw(stem TEXT PRIMARY KEY,digest TEXT,payload BLOB)')
                def save(db,method,rec,dets,rects,index):
                    body={'image_sha256':rec['image_sha256'],'detections':dets,'rects':rects,'baseline_sha256':None}
                    plain=json.dumps(body,separators=(',',':')).encode();h=hashlib.sha256(plain).hexdigest()
                    import zlib
                    db.execute('INSERT INTO prediction VALUES(?,?,?,?)',(method,rec['stem'],h,zlib.compress(plain)))
                    index.setdefault(rec['stem'],{'variant_sha256':{}})['variant_sha256'][method]=h
                for i in range(3):
                    rec={'stem':str(i),'image':f'{i}.jpg','image_sha256':str(i)*64,'width':100,'height':100};rows.append(rec)
                    r8=[[0,0,80,80]];r12=[[20,20,100,100]];union=r8+r12;raw=raw_fixture(union)
                    y8=[[30,30,40,40,.9,0]];y12=[[32,32,42,42,.95,0]]
                    lookup={tuple(r):j for j,r in enumerate(union)}
                    def independent_fuse(rects,base):
                        more=[row for rect in rects for row in independent_decode(raw[lookup[tuple(rect)]],rect)]
                        return literal_nms(list(base)+more)
                    h8=independent_fuse(r8,y8);h12=independent_fuse(r12,y12);mixed=independent_fuse(r8,y12)
                    for m,dets,rects in [('yolo800',y8,None),('yolo1280',y12,None),('hybrid_a025',h8,r8)]:save(db18,m,rec,dets,rects,idx18)
                    for m,dets,rects in [('yolo1280',y12,None),('hybrid800_reference',h8,r8),('hybrid1280',h12,r12),('yolo1280_rois800',mixed,r8)]:save(db20,m,rec,dets,rects,idx20)
                    stream=io.BytesIO();np.savez_compressed(stream,rects=np.array(union,np.int32),queries=raw);p=stream.getvalue();h=hashlib.sha256(p).hexdigest()
                    db20.execute('INSERT INTO crop_raw VALUES(?,?,?)',(rec['stem'],h,p));idx20[rec['stem']].update({'raw_sha256':h,'crops800':1,'crops1280':1})
                db18.commit();db20.commit()
            before=[f.sha(p) for p in paths]
            with contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
                db18,db20=[stack.enter_context(contextlib.closing(f.readonly(p))) for p in paths]
                cache=stack.enter_context(contextlib.closing(a.open_state(out/'audit.sqlite3','test')))
                summaries=a.perform(rows,db18,db20,idx18,idx20,cache,'test',out)
                result=a.aggregate(summaries,out)
                self.assertEqual(result['images'],3);self.assertEqual(result['total_union_crops'],6)
                saved=(out/'geometry_summary.json').read_bytes()
                again=a.perform(rows,db18,db20,idx18,idx20,cache,'test',out);a.aggregate(again,out)
                self.assertEqual(saved,(out/'geometry_summary.json').read_bytes())
                cache.execute("UPDATE summaries SET digest='bad' WHERE stem='0'");cache.commit()
                with self.assertRaises(ValueError):a.perform(rows,db18,db20,idx18,idx20,cache,'test',out)
                with self.assertRaises(ValueError):a.open_state(out/'audit.sqlite3','changed')
            self.assertEqual([f.sha(p) for p in paths],before)
            for p in paths:p.unlink()
            (out/'audit.sqlite3').unlink()

    def test_raw_replay_detects_changed_prediction(self):
        rec={'width':100,'height':100,'stem':'one','image':'one.jpg'};rects=[[0,0,100,100]];raw=raw_fixture(rects)
        with self.assertRaises(ValueError):t.image_summary(rec,raw,rects,rects,[],[])

    def test_global_cap_and_cross_source_suppression(self):
        rows=[[0,0,10,10,.1,0],[0,0,10,10,.2,0]];o=[[0,-1,0],[1,0,0]]
        ids,status,by=t.trace(rows,o,100,100);self.assertEqual(ids.tolist(),[1]);self.assertEqual(by[0],1)
        rows=[[i*2,0,i*2+1,1,1-i*.00001,0] for i in range(1002)];o=[[1,0,i] for i in range(1002)]
        ids,status,by=t.trace(rows,o,2100,10);self.assertEqual(len(ids),1000);self.assertEqual(np.count_nonzero(status==3),2)

if __name__=='__main__':unittest.main(verbosity=2)
