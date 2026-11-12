"""Fixed 1280 test-dev extension; two crop geometries, cached global detections."""
from __future__ import annotations
import argparse
from contextlib import closing
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback
import zipfile

import frozen_support as s

HERE=Path(__file__).resolve().parent
RUN="testdev1280_extension_v1_seed42"
OLD18="testdev_frozen_v1_seed42"
OLD19="full1280_development_v1_seed42"
METHODS=("yolo1280","hybrid1280","yolo1280_rois800","hybrid800_reference")
COUNT=1610
DRAWS=1000
PILOT_N=8
REPORT="20_testdev1280_rapor.zip"


def readonly(path,expected):
    s.check_file(path,expected)
    db=sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True,timeout=10)
    db.execute("PRAGMA query_only=ON")
    return db


def bind_source(folder,expected):
    docs={}
    for name,digest in expected.items():
        if name=="database_sha256":continue
        s.check_file(folder/name,digest)
        docs[name]=s.read(folder/name)
    signature=s.fingerprint(docs["protocol.json"])
    for name,doc in docs.items():
        if name not in ("protocol.json","dataset_manifest.json"):
            s.require(doc.get("protocol_sha256")==signature,f"Kaynak protokol baglantisi farkli: {folder/name}")
    s.require(docs["result.json"]["status"]=="COMPLETED","Onceki deney tamamlanmamis.")
    return docs,signature


def prepare(root,out,dataset_override=None):
    ref=s.read(HERE/"evidence/source_bindings.json")
    old=root/"experiments"/OLD18
    dev=root/"experiments"/OLD19
    a,sig18=bind_source(old,ref["stage18"])
    b,sig19=bind_source(dev,ref["stage19"])
    f,_,weights,_,_=s.scan_project(root)
    env=s.environment()
    s.require(env==a["protocol.json"]["environment"]==b["protocol.json"]["environment"],
              "GPU/Python/surucu ortami onceki 18 ve 19 ile farkli.")
    s.require(s.fingerprint(a["dataset_manifest.json"])==a["protocol.json"]["dataset_manifest_sha256"],
              "18 test-dev manifest baglantisi farkli.")
    records=a["dataset_manifest.json"]["records"]
    s.require(len(records)==COUNT and len({r["image"] for r in records})==COUNT and
              len({r["stem"] for r in records})==COUNT,"1610 test-dev kaydi gerekli.")
    labels={r["stem"]:r["sha256"] for r in a["annotation_index.json"]["records"]}
    index={r["image"]:r for r in a["prediction_index.json"]["records"]}
    s.require(set(labels)=={r["stem"] for r in records} and set(index)=={r["image"] for r in records},
              "18 test-dev indeks kapsami farkli.")
    sha=a["dataset_manifest.json"]["official_zip_sha256"]
    dataset=old/("dataset_"+sha[:16]) if sha else Path()
    if dataset_override is not None:
        dataset=Path(dataset_override).resolve()
    if not (dataset/"images").is_dir() or not (dataset/"annotations").is_dir():
        # Require an explicit path only for non-default/moved data; no new download/extraction.
        raise FileNotFoundError("18. deneyin acilmis test-dev klasoru bulunamadi. --dataset ile images/annotations iceren klasoru belirtin.")
    protocol={"id":RUN,"source_bindings":ref,"source18_protocol_sha256":sig18,"source19_protocol_sha256":sig19,
              "runner_sha256":s.sha256(Path(__file__)),"support_sha256":s.sha256(HERE/"frozen_support.py"),
              "parity_records_sha256":s.sha256(HERE/"evidence/stage19_parity.json"),
              "frozen_evidence_sha256":s.sha256(s.FROZEN),"records_sha256":s.fingerprint(records),
              "environment":env,"methods":list(METHODS),"images":COUNT,"alpha":.25,
              "crop_policy":f["crop_policy"],"crop_inference":f["crop_inference"],"fusion":f["fusion"],
              "primary":"hybrid1280 minus yolo1280 toolkit AP",
              "secondary":["yolo1280_rois800 minus yolo1280","hybrid1280 minus yolo1280_rois800"],
              "scope":"exploratory extension; test-dev results of original HyCADO800 already examined",
              "training":False,"parameter_search":False,"new_global_inference":"3 development images in preflight only",
              "new_crop_inference":"union of ordered 800/1280 rectangles per image; identical rectangles reuse the exact same batch-one query",
              "bootstrap":{"draws":DRAWS,"mode":"paired filename/hash proxy clusters","seed":20261102},
              "timing":"progress/estimate only; not a controlled latency benchmark",
              "annotation_policy":"new output sets frozen before new metric matching; existing test labels/metrics were used in stage18"}
    s.bind_protocol(out,"protocol.json",protocol)
    return f,weights,records,labels,index,dataset,a,b,old,dev,ref,protocol


def progress(out,phase,done,total):
    if done==1 or done%25==0 or done==total:
        print(f"20 {phase}: {done}/{total}",flush=True)
        s.write_json(out/"progress.json",{"phase":phase,"done":done,"total":total,"utc":s.utc_now()})


def rects(rec,base,f):
    from hycado_start.crop_data import select_clusters
    _,clusters=select_clusters({"width":rec["width"],"height":rec["height"],"detections":base},f["crop_policy"])
    return [list(c["rect"]) for c in clusters]


def source_predictions(db,rec,index):
    expected=index[rec["image"]]["variant_sha256"]
    docs={}
    for method in ("yolo800","yolo1280","hybrid_a025"):
        entry=db.execute("SELECT digest FROM prediction WHERE method=? AND stem=?",(method,rec["stem"])).fetchone()
        s.require(entry is not None and entry[0]==expected[method],f"18 tahmin indeksi farkli: {method}/{rec['stem']}")
        docs[method]=s.cache_rows(db,method,rec)
        s.require(docs[method] is not None,"Kaynak tahmin eksik.")
    return docs


def get_raw(db,rec,union):
    import numpy as np
    row=db.execute("SELECT digest,payload FROM crop_raw WHERE stem=?",(rec["stem"],)).fetchone()
    if row is None:return None
    s.require(hashlib.sha256(row[1]).hexdigest()==row[0],"Ham query arsivi hash farkli.")
    with np.load(io.BytesIO(row[1]),allow_pickle=False) as z:
        s.require(z["rects"].tolist()==union and z["queries"].shape==(len(union),300,14) and
                  z["queries"].dtype==np.float32 and np.isfinite(z["queries"]).all(),"Ham query bicimi veya geometri farkli.")
        return z["queries"].copy()


def save_raw(db,rec,union,queries):
    import numpy as np
    stream=io.BytesIO()
    np.savez_compressed(stream,rects=np.asarray(union,dtype=np.int32).reshape(-1,4),queries=queries)
    payload=stream.getvalue()
    with db:db.execute("INSERT INTO crop_raw VALUES(?,?,?)",(rec["stem"],hashlib.sha256(payload).hexdigest(),payload))


def fuse(raw,union,ordered,base,rec):
    from hycado_start.dev_evaluation import map_queries
    from hycado_start.nms_equivalence import fast_nms
    lookup={tuple(rect):i for i,rect in enumerate(union)}
    rows=list(base)
    for rect in ordered:
        more=map_queries(raw[lookup[tuple(rect)]],{"source_crop_xyxy":rect})
        for row in more:row[4]*=.25
        rows.extend(more)
    return fast_nms(rows,rec["width"],rec["height"])


def crop_phase(db,source,records,index,f,weights,dataset,out):
    import numpy as np
    import torch
    from hycado_start.crop_capture import load_model
    from hycado_start.runtime_benchmark import direct_queries
    model=None;elapsed=0.;new=0
    try:
        for i,rec in enumerate(records,1):
            docs=source_predictions(source,rec,index)
            base800=docs["yolo800"]["detections"];base1280=docs["yolo1280"]["detections"]
            r800=rects(rec,base800,f);r1280=rects(rec,base1280,f)
            s.require(r800==docs["hybrid_a025"]["rects"],"18 crop geometrisi yeniden elde edilemedi.")
            union=[list(r) for r in dict.fromkeys(tuple(r) for r in r800+r1280)]
            raw=get_raw(db,rec,union)
            if raw is None:
                if union and model is None:model=load_model(weights["crop"],weights["official"],"cuda")
                start=time.perf_counter()
                rgb=s.rgb_image(s.image_path(dataset,rec),rec)
                raw=np.stack([direct_queries(model,np.array(rgb.crop(tuple(r)))) for r in union]) if union else np.empty((0,300,14),dtype=np.float32)
                # Every reconstructed original HyCADO output must match stage18 exactly.
                s.require(fuse(raw,union,r800,base800,rec)==docs["hybrid_a025"]["detections"],
                          f"18 orijinal HyCADO800 birebir eslik farkli: {rec['image']}")
                save_raw(db,rec,union,raw)
                elapsed+=time.perf_counter()-start;new+=1
            else:
                s.require(fuse(raw,union,r800,base800,rec)==docs["hybrid_a025"]["detections"],
                          "Korunan ham query orijinal HyCADO800'u uretmiyor.")
            variants={"yolo1280":(base1280,None),
                      "hybrid800_reference":(docs["hybrid_a025"]["detections"],r800),
                      "hybrid1280":(fuse(raw,union,r1280,base1280,rec),r1280),
                      "yolo1280_rois800":(fuse(raw,union,r800,base1280,rec),r800)}
            for method,(rows,plan) in variants.items():
                prior=s.cache_rows(db,method,rec)
                binding=s.fingerprint(base1280 if method!="hybrid800_reference" else base800)
                if prior is None:s.cache_prediction(db,method,rec,rows,rects=plan,baseline_sha=binding)
                else:s.require(prior["detections"]==rows and prior["rects"]==plan and prior["baseline_sha256"]==binding,
                               f"Kayitli son tahmin yeniden uretilemiyor: {method}/{rec['stem']}")
            progress(out,"CROP",i,len(records))
        return elapsed,new
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()


def development_parity(f,weights,dev,out,b):
    import torch
    from hycado_start import proposals
    from hycado_start.runtime_benchmark import backend_state,restore_backend
    from hycado_start.crop_capture import load_model
    state=backend_state();predict=None;model=None
    records=s.read(HERE/"evidence/stage19_parity.json")
    try:
        with closing(readonly(dev/"predictions.sqlite3",b["result.json"]["prediction_database_sha256"])) as db:
            predict=proposals.make_predictor(proposals.load_ema(weights["yolo"],f["stage1_protocol_sha256"]),
                                             {"inference":f["stage1_inference"]|{"imgsz":1280}},out)
            pathroot=dev.parents[1]/"experiments/stage1_v1_seed42/data"
            for rec in records:
                path=s.safe(pathroot,rec["data_image"]);s.check_file(path,rec["image_sha256"])
                w,h,rows,_=predict(path)
                prior=s.cache_rows(db,"yolo1280",rec)
                s.require((w,h)==(rec["width"],rec["height"]) and prior is not None and rows==prior["detections"],
                          "19 YOLO1280 birebir esligi basarisiz.")
            del predict;predict=None;gc.collect();torch.cuda.empty_cache()
            model=load_model(weights["crop"],weights["official"],"cuda")
            for rec in records:
                base=s.cache_rows(db,"yolo1280",rec)["detections"]
                plan=rects(rec,base,f)
                path=s.safe(pathroot,rec["data_image"])
                rows=s.crop_rows(model,s.rgb_image(path,rec),rec,base,plan)
                prior=s.cache_rows(db,"hybrid1280",rec)
                s.require(prior is not None and rows==prior["detections"] and plan==prior["rects"],"19 HyCADO1280 birebir esligi basarisiz.")
        print("20 DEV eslik: 3 YOLO1280 ve 3 HyCADO1280 birebir eslesti.",flush=True)
    finally:
        if predict is not None:del predict
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache();restore_backend(state)


def freeze(db,records,signature,out):
    s.require(db.execute("SELECT count(*) FROM prediction").fetchone()[0]==len(records)*len(METHODS) and
              db.execute("SELECT count(*) FROM crop_raw").fetchone()[0]==len(records),"Tum test-dev tahminleri gerekli.")
    entries=[]
    for rec in records:
        variants={m:s.cache_rows(db,m,rec) for m in METHODS}
        s.require(all(x is not None for x in variants.values()),"Yontem kaydi eksik.")
        entries.append({"image":rec["image"],"crops800":len(variants["yolo1280_rois800"]["rects"]),
                        "crops1280":len(variants["hybrid1280"]["rects"]),
                        "raw_sha256":db.execute("SELECT digest FROM crop_raw WHERE stem=?",(rec["stem"],)).fetchone()[0],
                        "variant_sha256":dict(db.execute("SELECT method,digest FROM prediction WHERE stem=? ORDER BY method",(rec["stem"],)))})
    doc={"protocol_sha256":signature,"records":entries}
    s.bind_protocol(out,"prediction_index.json",doc)
    return doc


def measure(db,records,labels,dataset,oldmetrics,signature,out):
    import numpy as np
    import csv
    from hycado_start.score_ablation import aggregate
    s.METHODS=METHODS;parts=[]
    for i,rec in enumerate(records,1):
        ann=dataset/"annotations"/rec["annotation"]
        s.check_file(ann,labels[rec["stem"]])
        saved=db.execute("SELECT ann_sha,digest,payload FROM matching WHERE stem=?",(rec["stem"],)).fetchone()
        if saved:
            s.require(saved[0]==labels[rec["stem"]] and hashlib.sha256(saved[2]).hexdigest()==saved[1],"Etiket/matching kaydi farkli.")
            with np.load(io.BytesIO(saved[2]),allow_pickle=False) as z:part={k:z[k] for k in z.files}
        else:
            rows={m:s.cache_rows(db,m,rec)["detections"] for m in METHODS}
            part=s.metric_part(rec,ann.read_bytes(),rows)
            stream=io.BytesIO();np.savez_compressed(stream,**part);payload=stream.getvalue()
            with db:db.execute("INSERT INTO matching VALUES(?,?,?,?)",(rec["stem"],labels[rec["stem"]],hashlib.sha256(payload).hexdigest(),payload))
        parts.append(part);progress(out,"METRIK",i,len(records))
    metrics={m:aggregate(parts,m) for m in METHODS}
    from hycado_start.score_ablation import compare_metrics
    compare_metrics(metrics["yolo1280"],oldmetrics["methods"]["yolo1280"])
    compare_metrics(metrics["hybrid800_reference"],oldmetrics["methods"]["hybrid_a025"])
    for new,old in (("yolo1280","yolo1280"),("hybrid800_reference","hybrid_a025")):
        s.require(metrics[new]["counts_at_fixed_max500"]==oldmetrics["methods"][old]["counts_at_fixed_max500"],
                  "18 TP/FP metrik esligi farkli.")
    report={"protocol_sha256":signature,"images":len(records),"methods":metrics,"source18_metric_parity":True,
            "metric":"pinned toolkit-aligned Python AP in percentage points; not COCO mAP",
            "separate_reference_binary_validation":False}
    s.write_json(out/"metrics.json",report)
    keys=["AP","AP50","AP75","AR1","AR10","AR100","AR500"]
    with (out/"metrics.csv").open("w",newline="",encoding="utf-8-sig") as file:
        writer=csv.writer(file);writer.writerow(["method",*keys])
        for m in METHODS:writer.writerow([m,*(metrics[m][k] for k in keys)])
    with (out/"per_class.csv").open("w",newline="",encoding="utf-8-sig") as file:
        writer=csv.DictWriter(file,fieldnames=["method",*metrics[METHODS[0]]["per_class"][0]])
        writer.writeheader()
        for m in METHODS:writer.writerows({"method":m,**row} for row in metrics[m]["per_class"])
    return parts,metrics


def bootstrap(db,records,parts,metrics,out):
    import numpy as np
    from hycado_start import bootstrap_statistics as b
    groups=b.make_groups(records);s.write_json(out/"bootstrap_groups.json",groups)
    engine=b.compile_engine(parts,METHODS);point=b.evaluate(engine,np.ones(len(records),dtype=np.int64))
    expected=np.array([[metrics[m][k] for k in b.KEYS] for m in METHODS])
    s.require(np.allclose(point,expected,atol=1e-8,rtol=0),"Bootstrap nokta esligi basarisiz.")
    samples=[]
    for i in range(DRAWS):
        saved=db.execute("SELECT payload FROM bootstrap WHERE draw=?",(i,)).fetchone()
        if saved:
            e=json.loads(saved[0])
            s.require(e["sha256"]==s.fingerprint({"draw":i,"value":e["value"]}),"Bootstrap kayit hash farkli.")
            v=np.asarray(e["value"],dtype=float) if e["value"] is not None else np.full(point.shape,np.nan)
            s.require(v.shape==point.shape,"Bootstrap kayit bicimi farkli.")
        else:
            v=b.evaluate(engine,b.frequencies(groups,"cluster",i,seed=20261102))
            value=v.tolist() if np.isfinite(v).all() else None
            e={"value":value,"sha256":s.fingerprint({"draw":i,"value":value})}
            with db:db.execute("INSERT INTO bootstrap VALUES(?,?)",(i,json.dumps(e,allow_nan=False)))
            progress(out,"BOOTSTRAP",i+1,DRAWS)
        samples.append(v)
    arr=np.stack(samples);valid=np.isfinite(arr).all(axis=(1,2));arr=arr[valid]
    s.require(len(arr)>=2,"Yeterli tanimli bootstrap tekrari yok.")
    contrasts={}
    for a,bid in ((1,0),(2,0),(1,2)):
        diff=arr[:,a,:]-arr[:,bid,:];low,high=np.quantile(diff,[.025,.975],axis=0,method="linear")
        contrasts[METHODS[a]+"_minus_"+METHODS[bid]]={
            key:{"point":float(point[a,j]-point[bid,j]),"lower95":float(low[j]),"upper95":float(high[j])}
            for j,key in enumerate(b.KEYS)}
    doc={"valid_replicates":int(valid.sum()),"replicates_requested":DRAWS,"proxy_cluster_count":len(groups["labels"]),
         "differences_pp":contrasts,"primary_contrast":"hybrid1280_minus_yolo1280",
         "scope":"fixed models, exploratory previously examined test-dev; sample uncertainty only",
         "multiplicity":"marginal 95% intervals, no multiple-comparison correction",
         "interval":"paired proxy-cluster percentile; groups are not verified independent scenes"}
    s.write_json(out/"intervals.json",doc)
    return doc


def bundle(out):
    files=["protocol.json","20_onkontrol.json","prediction_index.json","metrics.json","metrics.csv","per_class.csv",
           "intervals.json","bootstrap_groups.json","result.json"]
    temp=out/(REPORT+".partial")
    with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(out/name,name)
    os.replace(temp,out/REPORT)


def run(args,root,out):
    f,weights,records,labels,index,dataset,a,b,old,dev,ref,protocol=prepare(root,out,args.dataset)
    signature=s.fingerprint(protocol)
    with closing(readonly(old/"predictions.sqlite3",ref["stage18"]["database_sha256"])) as source:
        s.require(source.execute("SELECT signature FROM binding WHERE id=1").fetchone()==(protocol["source18_protocol_sha256"],),
                  "18 tahmin kasasi protokol farkli.")
        with closing(s.open_cache(out/"predictions.sqlite3",signature)) as db:
            with db:db.execute("CREATE TABLE IF NOT EXISTS crop_raw (stem TEXT PRIMARY KEY,digest TEXT NOT NULL,payload BLOB NOT NULL)")
            if args.mode=="preflight":
                target=out/"20_onkontrol.json"
                if target.exists():
                    pre=s.read(target);s.require(pre["status"]=="PREFLIGHT_OK" and pre["protocol_sha256"]==signature,"Onkontrol farkli.")
                    print(f"ON KONTROL ONCEDEN TAMAM: {target}",flush=True);return
                development_parity(f,weights,dev,out,b)
                sample=sorted(records,key=lambda r:s.fingerprint({"20_pilot":r["image"]}))[:PILOT_N]
                elapsed,new=crop_phase(db,source,sample,index,f,weights,dataset,out)
                estimate=elapsed/new*(len(records)-len(sample))/3600 if new else None
                pre={"status":"PREFLIGHT_OK","protocol_sha256":signature,"development_exact_images":3,
                     "testdev_pilot_images":len(sample),"original_hycado800_replay_exact":True,
                     "testdev_annotation_rows_parsed":False,"estimated_remaining_inference_hours":estimate,
                     "estimate_excludes":"CPU metrics/bootstrap and setup; approximate subset estimate",
                     "training":False,"cached_global_predictions_reused":True}
                s.write_json(target,pre)
                print(f"ON KONTROL TAMAM: {target}",flush=True)
                if estimate is not None:print(f"Kalan crop cikarimi ~{estimate:.2f} saat; CPU metrik/GA haric.",flush=True)
                return
            pre=s.read(out/"20_onkontrol.json")
            s.require(pre["status"]=="PREFLIGHT_OK" and pre["protocol_sha256"]==signature,"Once 20A basarili olmali.")
            crop_phase(db,source,records,index,f,weights,dataset,out)
            frozen=freeze(db,records,signature,out)
            parts,metrics=measure(db,records,labels,dataset,a["metrics.json"],signature,out)
            ci=bootstrap(db,records,parts,metrics,out)
    result={"status":"COMPLETED","protocol_sha256":signature,"images":COUNT,
            "primary_AP_difference_pp":ci["differences_pp"]["hybrid1280_minus_yolo1280"]["AP"],
            "secondary_oldroi_AP_difference_pp":ci["differences_pp"]["yolo1280_rois800_minus_yolo1280"]["AP"],
            "crop_calls800":sum(r["crops800"] for r in frozen["records"]),
            "crop_calls1280":sum(r["crops1280"] for r in frozen["records"]),
            "source18_original_outputs_and_metrics_reproduced":True,
            "training":False,"parameter_search":False,"independent_untouched_test":False,
            "latency_benchmark":False,"completed_utc":s.utc_now(),
            "prediction_database_sha256":s.sha256(out/"predictions.sqlite3")}
    s.write_json(out/"result.json",result);bundle(out)
    print(f"SONUC: COMPLETED\nGONDERILECEK RAPOR: {out/REPORT}",flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("mode",choices=["preflight","full"])
    ap.add_argument("--project",required=True,type=Path);ap.add_argument("--dataset",type=Path,default=None)
    args=ap.parse_args();root=args.project.resolve()
    if not (root/"experiments").is_dir():print(f"Proje klasoru bulunamadi: {root}");return 1
    out=root/"experiments"/RUN;out.mkdir(exist_ok=True)
    try:
        with s.training_lock(out/"run.lock"):
            try:run(args,root,out)
            except (Exception,KeyboardInterrupt) as exc:
                s.write_json(out/"error.json",{"status":"STOPPED","error":str(exc),"traceback":traceback.format_exc()});raise
        return 0
    except (Exception,KeyboardInterrupt):
        traceback.print_exc();print(f"DURDU. Hata raporu: {out/'error.json'}",flush=True);return 1

if __name__=="__main__":raise SystemExit(main())
