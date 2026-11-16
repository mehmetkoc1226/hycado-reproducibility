"""Synthetic fixed-grid and COCO provenance tests; no project input needed."""
import contextlib,hashlib,io,json,sqlite3,tempfile,unittest,zlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import frozen_cpu as f
import trace29 as t
import alpha30_core as a
import coco_core as c
import run_alpha30 as run

def queries(rects):
    q=np.zeros((len(rects),300,14),np.float32)
    for i in range(len(rects)):
        q[i,0,:4]=[.5,.5,.2,.2];q[i,0,4]=.8
        q[i,1,:4]=[.5,.5,1,1];q[i,1,5]=.3
        q[i,2,:4]=[.1,.1,.1,.1];q[i,2,4]=.002
    return q

def literal(rows):
    kept=[]
    for row in sorted(rows,key=lambda r:(-r[4],r[5],*r[:4])):
        reject=False
        for k in kept:
            inter=max(0,min(row[2],k[2])-max(row[0],k[0]))*max(0,min(row[3],k[3])-max(row[1],k[1]))
            union=(row[2]-row[0])*(row[3]-row[1])+(k[2]-k[0])*(k[3]-k[1])-inter
            if row[5]==k[5] and inter/union>.7:reject=True;break
        if not reject:kept.append(row)
        if len(kept)==1000:break
    return kept

class AlphaTests(unittest.TestCase):
    def test_fixed_grid_native_geometry_and_true_global_only_endpoint(self):
        rec={'id':1,'width':100,'height':100};rects=[[0,0,100,100]];raw=queries(rects)
        base=[[40,40,60,60,.4,0]];before=json.dumps(base)
        result=a.variants(raw,rects,rects,base,rec)
        self.assertEqual(result['a000']['detections'],base);self.assertEqual(json.dumps(base),before)
        mapped,ids=t.decode(raw[0],rects[0])
        for method,alpha in a.GRID:
            if alpha==0:continue
            rows=mapped.copy();rows[:,4]*=alpha
            self.assertEqual(result[method]['detections'],literal(base+rows.tolist()))
        # Large alpha changes cross-source suppression; it must not merely rescale final old boxes.
        self.assertNotEqual(result['a0125']['detections'][0][4],result['a100']['detections'][0][4])

    def test_alpha025_source_origins_and_area_ratios(self):
        rec={'id':1,'width':100,'height':100};rects=[[0,0,100,100]]
        out=a.variants(queries(rects),rects,rects,[[40,40,60,60,.9,0]],rec)['a025']
        self.assertEqual(len(out['origins']),len(out['detections']))
        for r,o,ratio in zip(out['detections'],out['origins'],out['box_crop_area_ratios']):
            if o[0]==0:self.assertIsNone(ratio)
            else:self.assertGreater(ratio,0);self.assertLessEqual(ratio,1);self.assertLessEqual(r[4],.25)

    def test_COCO_dtIds_map_back_through_class_sorting(self):
        rec={'id':1,'width':200,'height':200,'image':'one.jpg','image_sha256':'a'}
        doc={'detections':[[100,100,120,120,.9,1],[50,50,70,70,.2,0],[10,10,30,30,.1,0]],
             'origins':[[0,-1,0],[1,0,0],[1,1,0]],'box_crop_area_ratios':[None,1.,1.],
             'rects':[[50,50,70,70],[10,10,30,30]]}
        dt,ids=a.filtered_detections(doc,None);entries=c.one_image(rec,[[10,10,20,20,1]],dt.tolist())
        result=a.geometry_diagnostic(doc,rec,entries,dt,ids)
        for iou in ['0.5','0.75']:
            self.assertEqual(result['counts'][iou]['crop_all'],{'TP':1,'FP':1,'ignored':0})
        status={x['crop_index']:x['COCO_IoU_status']['0.5'] for x in result['examples_display_crop_filling']}
        self.assertEqual(status,{0:'FP',1:'TP'})

    def test_ignore_filter_preserves_native_source_ID(self):
        import visdrone_preprocess as v
        rec={'id':1,'width':100,'height':100,'image':'one.jpg','image_sha256':'a'}
        raw=v.raw_annotation('1,1,30,30,0,0,0,0\n60,60,10,10,1,1,0,0')
        integral=v.ignore_integral(raw,100,100)
        doc={'detections':[[2,2,12,12,.2,0],[60,60,70,70,.1,0]],'origins':[[1,0,0],[1,1,0]],
             'box_crop_area_ratios':[1.,1.],'rects':[[2,2,12,12],[60,60,70,70]]}
        dt,ids=a.filtered_detections(doc,integral);self.assertEqual(ids.tolist(),[1])
        entries=c.one_image(rec,[[60,60,10,10,1]],dt.tolist())
        result=a.geometry_diagnostic(doc,rec,entries,dt,ids)
        self.assertEqual(result['counts']['0.5']['crop_all']['TP'],1)
        self.assertEqual(result['examples_display_crop_filling'][0]['crop_index'],1)

    def test_per_class_top100_diagnostic_cap(self):
        rec={'id':1,'width':500,'height':500,'image':'one.jpg','image_sha256':'a'}
        rows=[[i*2,0,i*2+1,1,.2-i*.0001,0] for i in range(105)]+[[400,400,410,410,.05,0]]
        doc={'detections':rows,'origins':[[1,i,0] for i in range(len(rows))],
             'box_crop_area_ratios':[1.]*len(rows),'rects':[r[:4] for r in rows]}
        dt,ids=a.filtered_detections(doc,None);entries=c.one_image(rec,[[400,400,10,10,1]],dt.tolist())
        d=a.geometry_diagnostic(doc,rec,entries,dt,ids)
        self.assertEqual(d['counts']['0.5']['crop_all'],{'TP':0,'FP':100,'ignored':0})

    def test_cache_resume_tampering_and_atomic_coupled_diagnostic(self):
        rec={'id':1,'width':100,'height':100,'image_sha256':'a'}
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'cache.sqlite3'
            with contextlib.closing(a.state(path,'fixed')) as db:
                doc={'signature':'fixed','image_id':1,'image_sha256':'a','alpha':.5,'detections':[]}
                h,p=f.pack(doc)
                with db:db.execute('INSERT INTO predictions VALUES(?,?,?,?)',('a050',1,h,p))
                self.assertEqual(a.load_prediction(db,'a050',rec,'fixed'),doc)
                e=c.one_image(rec,[[0,0,10,10,1]],[])
                with self.assertRaises(ValueError):a.save_evaluation(db,'a025',rec,e,None)
                self.assertFalse(a.done(db,'a025',1))
                self.assertEqual(db.execute("SELECT count(*) FROM entries WHERE method='a025'").fetchone()[0],0)
                a.save_evaluation(db,'a050',rec,e);self.assertTrue(a.done(db,'a050',1))
            with contextlib.closing(a.state(path,'fixed')) as db:
                self.assertTrue(a.done(db,'a050',1))
                with db:db.execute("UPDATE done SET marker='bad'")
                with self.assertRaises(ValueError):a.done(db,'a050',1)
            with self.assertRaises(ValueError):a.state(path,'changed')
            path.unlink()

    def test_end_to_end_freeze_metrics_geometry_and_reopen_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'out';out.mkdir();ann=root/'annotations';ann.mkdir();source=root/'source.sqlite3'
            records=[];idx={}
            with contextlib.closing(sqlite3.connect(source)) as db:
                db.execute('CREATE TABLE prediction(method TEXT,stem TEXT,digest TEXT,payload BLOB,PRIMARY KEY(method,stem))')
                db.execute('CREATE TABLE crop_raw(stem TEXT PRIMARY KEY,digest TEXT,payload BLOB)')
                for i in range(1,4):
                    p=ann/(str(i)+'.txt');p.write_text('40,40,20,20,1,1,0,0\n')
                    rec={'id':i,'stem':str(i),'image':str(i)+'.jpg','image_sha256':str(i)*64,'width':100,'height':100,
                         'annotation':p.name,'annotation_sha256':f.sha(p)};records.append(rec)
                    rects=[[0,0,100,100]];raw=queries(rects);base=[[40,40,60,60,.35,0]] if i!=3 else []
                    variants=a.variants(raw,rects,rects,base,rec)
                    hashes={}
                    for method,rows,plan in [('yolo1280',base,None),('hybrid1280',variants['a025']['detections'],rects),('hybrid800_reference',variants['a025']['detections'],rects)]:
                        body={'image_sha256':rec['image_sha256'],'detections':rows,'rects':plan}
                        payload=json.dumps(body,separators=(',',':')).encode();h=hashlib.sha256(payload).hexdigest();hashes[method]=h
                        db.execute('INSERT INTO prediction VALUES(?,?,?,?)',(method,rec['stem'],h,zlib.compress(payload)))
                    stream=io.BytesIO();np.savez_compressed(stream,rects=np.asarray(rects,np.int32),queries=raw);payload=stream.getvalue();h=hashlib.sha256(payload).hexdigest()
                    db.execute('INSERT INTO crop_raw VALUES(?,?,?)',(rec['stem'],h,payload));idx[rec['stem']]={'variant_sha256':hashes,'raw_sha256':h}
                db.commit()
            oldsha=f.sha(source);path=out/'cache.sqlite3'
            with contextlib.closing(f.readonly(source)) as src,contextlib.closing(a.state(path,'fixed')) as db,contextlib.redirect_stdout(io.StringIO()):
                token=run.freeze(src,db,records,idx,'fixed',out)
                self.assertEqual(run.measure(db,records,ann,'fixed',out),3)
                ids=[r['id'] for r in records];allowed=set(ids)
                # Generate baseline fixture via direct per-image upstream evaluation.
                baseline={}
                for rec in records:
                    doc=a.load_prediction(db,'a000',rec,'fixed');dt,_=a.filtered_detections(doc,None)
                    for e in c.one_image(rec,[[40,40,20,20,1]],dt.tolist()):
                        d,h,p=c.encode(e);baseline[(rec['id'],d['category_id'],d['area_index'])]=c.decode(h,p)
                base=c.aggregate(ids,lambda cat,area:{im:e for (im,k,ar),e in baseline.items() if k==cat and ar==area})[0]
                parent=c.aggregate(ids,lambda cat,area:a.load_entries(db,'a025',cat,area,allowed))[0]
                expected={'yolo1280':base,'hybrid1280':parent}
                point=run.summarize_metrics(db,records,out,expected)
                geometry=run.summarize_geometry(db,records,'fixed',out)
                self.assertEqual(point['alpha025_max_abs_difference_pp'],0)
                self.assertEqual(geometry['images'],3)
                csv_before=(out/'alpha_metrics.csv').read_bytes()
            with contextlib.closing(f.readonly(source)) as src,contextlib.closing(a.state(path,'fixed')) as db,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run.freeze(src,db,records,idx,'fixed',out),token)
                self.assertEqual(run.measure(db,records,ann,'fixed',out),3)
                again=run.summarize_metrics(db,records,out,expected)
                self.assertEqual(again,point);self.assertEqual((out/'alpha_metrics.csv').read_bytes(),csv_before)
                self.assertEqual(run.summarize_geometry(db,records,'fixed',out),geometry)
            self.assertEqual(f.sha(source),oldsha);source.unlink();path.unlink()

    def test_retained29_gate_covers_exact_raw_index(self):
        gate=f.read(run.REF/'29_result.json');ix=f.read(run.REF/'29_verified_raw_index.json')
        self.assertEqual(gate['status'],'RAW29_AUDIT_COMPLETE');self.assertTrue(gate['alpha_sensitivity_possible_without_inference'])
        self.assertEqual(f.fp(ix),gate['verified_raw_index_sha256'])
        refs={Path(r['image']).stem:r for r in f.read(run.REF/'stage20/prediction_index.json')['records']}
        for row in ix['records']:self.assertEqual(row['raw_archive_sha256'],refs[Path(row['image']).stem]['raw_sha256'])

if __name__=='__main__':unittest.main(verbosity=2)
