"""Development-only full HyCADO-1280. Frozen recipe; cached YOLO; new 1280-guided crops."""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path, PurePosixPath
import sqlite3
import time
import traceback
from contextlib import closing
import zipfile
import os

import frozen_support as s

HERE = Path(__file__).resolve().parent
N = 647
DRAWS = 1000
BASELINE_AP = 31.65130516071972
METHODS = ("yolo1280", "yolo1280_nms_only", "hybrid1280")
RUN = "full1280_development_v1_seed42"
REPORT = "19_tam1280_gelistirme_rapor.zip"


def prepare(root, out):
    f = s.read(s.FROZEN)
    ref = s.read(HERE/"evidence/development_refs.json")
    pilot = s.read(HERE/"evidence/previous_pilot_outputs.json")
    for name, digest in f["engine_sha256"].items():
        s.check_file(HERE/"engine/hycado_start"/name, digest)
    for name, version in f["runtime_versions"].items():
        s.require(importlib.metadata.version(name)==version,
                  f"Paket surumu farkli: {name}; gereken {version}. Projenin .venv ortamini kullanin.")
    stage = root/"experiments/stage1_v1_seed42"
    crop = root/"experiments/rtdetr_crop_v1_seed42"
    p2 = root/"experiments/p2_strong_baselines_v1_seed42"
    manifest_path = stage/"data/manifest.json"
    s.check_file(manifest_path, ref["manifest"])
    manifest = s.read(manifest_path)
    raw = sorted((r for r in manifest["records"] if r["split"]=="val"), key=lambda r:r["source_image"])
    s.require(len(raw)==N and len({r["source_image"] for r in raw})==N and
              all(r["source_split"]=="train" and r["source_image"].startswith("images/train/") and
                  PurePosixPath(r["image"]).parts[:2]==("images","val") for r in raw),
              "647 TRAIN-ici gelistirme kaydi gerekli.")
    p = s.read(p2/"development_protocol.json")
    s.require(s.fingerprint(p)==ref["p2_protocol"] and p["manifest_sha256"]==ref["manifest"] and
              p["checkpoint_sha256"]==ref["checkpoint"] and p["training"] is False and
              p["official_val_used"] is False, "P2 gelistirme protokolu farkli.")
    s.check_file(p2/"development_prediction_index.json",ref["p2_index"])
    idx=s.read(p2/"development_prediction_index.json")
    entries=idx["methods"]["yolo1280"]
    s.require(idx["protocol_sha256"]==ref["p2_protocol"] and
              [x["source_image"] for x in entries]==[r["source_image"] for r in raw],
              "P2 YOLO1280 indeksi farkli.")
    weights={}
    for kind,folder,key in (("yolo",stage,"stage1_protocol_sha256"),("crop",crop,"crop_training_protocol_sha256")):
        proto=s.read(folder/"protocol.json")
        s.require(s.fingerprint(proto)==f[key],f"Egitim protokolu farkli: {kind}")
        path=root/"release_stage/checkpoints"/f"{kind}_epoch_0080.pt"
        if not path.exists():
            path=s.safe(folder,s.read(folder/"checkpoint_index.json")["latest"]["path"])
        s.check_file(path,f["checkpoint_sha256"][kind])
        weights[kind]=path
        if kind=="crop":
            official=root/"work/fresh_official_pretrained/rtdetr-l.pt"
            s.check_file(official,proto["initial_weights_sha256"])
            weights["official"]=official
            f["architecture_checkpoint_sha256"]=proto["initial_weights_sha256"]
    env=s.environment()
    records=[]
    for i,(r,entry) in enumerate(zip(raw,entries),1):
        s.check_file(s.safe(stage/"data",r["image"]),r["image_sha256"])
        s.check_file(s.safe(p2,entry["path"]),entry["sha256"])
        doc=s.read(s.safe(p2,entry["path"]))
        s.require(doc["content_sha256"]==s.fingerprint({k:v for k,v in doc.items() if k!="content_sha256"}) and
                  doc["method"]=="yolo1280" and doc["source_image"]==r["source_image"] and
                  doc["protocol_sha256"]==ref["p2_protocol"] and
                  doc["image_sha256"]==r["image_sha256"] and len(doc["detections"])==entry["detections"],
                  f"YOLO-1280 tahmin kimligi farkli: {r['source_image']}")
        records.append({"image":r["source_image"],"stem":PurePosixPath(r["source_image"]).stem,
                        "width":doc["width"],"height":doc["height"],"image_sha256":r["image_sha256"],
                        "annotation_sha256":r["annotation_sha256"],"data_image":r["image"],
                        "p2_path":entry["path"],"p2_sha256":entry["sha256"]})
        if i==1 or i%200==0 or i==N:
            print(f"19 kaynak denetimi: {i}/{N}",flush=True)
    s.require(set(pilot).issubset({r["image"] for r in records}),"Onceki 32 goruntu pilota baglanamadi.")
    protocol={"id":RUN,"images":N,"records":records,"upstream_refs":ref,
              "frozen_evidence_sha256":s.sha256(s.FROZEN),
              "source_sha256":{name:s.sha256(HERE/name) for name in ("run1280.py","frozen_support.py","evidence/previous_pilot_outputs.json")},
              "environment":env,"stage1_inference":f["stage1_inference"]|{"imgsz":1280},
              "crop_policy":f["crop_policy"],"crop_inference":f["crop_inference"],
              "crop_score_multiplier":.25,"fusion":f["fusion"],
              "checkpoint_sha256":f["checkpoint_sha256"]|{"architecture":f["architecture_checkpoint_sha256"]},
              "methods":list(METHODS),"training":False,"parameter_search":False,
              "official_val_images_used":False,"testdev_used":False,
              "scope":"previously examined TRAIN-internal development; exploratory",
              "annotation_policy":"all 647 output sets frozen before raw annotation parsing",
              "primary_endpoint":"hybrid1280 minus cached YOLO1280 toolkit AP",
              "bootstrap":{"draws":DRAWS,"seed":20261102,"mode":"paired filename/hash proxy clusters"},
              "timing_scope":"progress/estimate only, not a latency benchmark"}
    s.bind_protocol(out,"protocol.json",protocol)
    return f,records,weights,pilot,stage/"data",p2,protocol


def progress(out,phase,done,total,new):
    if new==1 or done%25==0 or done==total:
        print(f"19 {phase}: {done}/{total}",flush=True)
        s.write_json(out/"progress.json",{"phase":phase,"done":done,"total":total,"utc":s.utc_now()})


def baseline(rec,p2):
    s.check_file(s.safe(p2,rec["p2_path"]),rec["p2_sha256"])
    return s.read(s.safe(p2,rec["p2_path"]))["detections"]


def image_path(data,rec):
    path=s.safe(data,rec["data_image"])
    s.check_file(path,rec["image_sha256"])
    return path


def rects_for(rec,base,f):
    from hycado_start.crop_data import select_clusters
    _,clusters=select_clusters({"width":rec["width"],"height":rec["height"],"detections":base},f["crop_policy"])
    return [list(c["rect"]) for c in clusters]


def yolo_parity(records,f,weights,data,p2,out):
    import torch
    from hycado_start import proposals
    from hycado_start.runtime_benchmark import backend_state,restore_backend
    state=backend_state()
    predict=proposals.make_predictor(proposals.load_ema(weights["yolo"],f["stage1_protocol_sha256"]),
                                     {"inference":f["stage1_inference"]|{"imgsz":1280}},out)
    try:
        for rec in records[:3]:
            w,h,rows,_=predict(image_path(data,rec))
            s.require((w,h)==(rec["width"],rec["height"]) and rows==baseline(rec,p2),
                      f"YOLO1280 birebir eslik basarisiz: {rec['image']}")
        print("YOLO1280 smoke: 3/3 birebir eslesti.",flush=True)
    finally:
        del predict
        gc.collect()
        torch.cuda.empty_cache()
        restore_backend(state)


def predict_crops(db,records,f,weights,data,p2,pilot,out):
    import torch
    from hycado_start.crop_capture import load_model
    from hycado_start.nms_equivalence import fast_nms
    pending=[r for r in records if s.cache_rows(db,"hybrid1280",r) is None]
    if not pending:
        print("19 crop: tamamlanan tahminler korunuyor.",flush=True)
        return 0.,0
    model=load_model(weights["crop"],weights["official"],"cuda")
    started=time.perf_counter();calls=0
    try:
        for i,rec in enumerate(pending,1):
            base=baseline(rec,p2)
            rects=rects_for(rec,base,f)
            rows=s.crop_rows(model,s.rgb_image(image_path(data,rec),rec),rec,base,rects)
            if rec["image"] in pilot:
                expected=pilot[rec["image"]]
                s.require(s.fingerprint({"detections":rows,"rects":rects})==expected["sha256"] and
                          len(rects)==expected["crop_count"],
                          f"Onceki tam1280 pilotuyla birebir eslik basarisiz: {rec['image']}")
            # Each image commits atomically. A failed commit is recomputed on resume.
            s.cache_prediction(db,"hybrid1280",rec,rows,rects=rects,baseline_sha=s.fingerprint(base))
            calls+=len(rects)
            if i==1 or i%25==0 or i==len(pending):
                elapsed=time.perf_counter()-started
                eta=elapsed/i*(len(pending)-i)/60
                print(f"19 HyCADO1280: {len(records)-len(pending)+i}/{len(records)}; yeni crop={calls}; kalan cikarim ~{eta:.1f} dk",flush=True)
                s.write_json(out/"progress.json",{"phase":"CROP","done":len(records)-len(pending)+i,
                             "total":len(records),"new_crop_calls":calls,"remaining_inference_minutes":eta})
        return time.perf_counter()-started,calls
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def freeze(db,records,f,p2,signature,out):
    from hycado_start.nms_equivalence import fast_nms
    entries=[]
    for rec in records:
        base=baseline(rec,p2)
        for method,rows in (("yolo1280",base),("yolo1280_nms_only",fast_nms(base,rec["width"],rec["height"]))):
            prior=s.cache_rows(db,method,rec)
            if prior is None:
                s.cache_prediction(db,method,rec,rows)
            else:
                s.require(prior["detections"]==rows,f"Korunan baseline farkli: {rec['stem']}")
        crop=s.cache_rows(db,"hybrid1280",rec)
        s.require(crop is not None and crop["rects"]==rects_for(rec,base,f) and
                  crop["baseline_sha256"]==s.fingerprint(base),f"Crop tahmini/plan eksik: {rec['stem']}")
        entries.append({"image":rec["image"],"crops":len(crop["rects"]),
                        "variant_sha256":dict(db.execute("SELECT method,digest FROM prediction WHERE stem=? ORDER BY method",(rec["stem"],)))})
    s.require(db.execute("SELECT count(*) FROM prediction").fetchone()[0]==len(records)*len(METHODS),"Tahmin sayisi farkli.")
    index={"protocol_sha256":signature,"records":entries}
    s.bind_protocol(out,"prediction_index.json",index)
    return index


def measure(db,records,root,out,signature):
    import numpy as np
    from hycado_start.score_ablation import aggregate
    parts=[]
    s.METHODS=METHODS  # Shared exact matching implementation; no change to scoring rules.
    for i,rec in enumerate(records,1):
        ann=root/"data/prepared/visdrone_annotations/train"/(rec["stem"]+".txt")
        s.check_file(ann,rec["annotation_sha256"])
        item=db.execute("SELECT ann_sha,digest,payload FROM matching WHERE stem=?",(rec["stem"],)).fetchone()
        if item:
            s.require(item[0]==rec["annotation_sha256"] and hashlib.sha256(item[2]).hexdigest()==item[1],
                      "Kayitli annotation/matching farkli.")
            with np.load(io.BytesIO(item[2]),allow_pickle=False) as data:
                part={k:data[k] for k in data.files}
        else:
            rows={m:s.cache_rows(db,m,rec)["detections"] for m in METHODS}
            part=s.metric_part(rec,ann.read_bytes(),rows)
            stream=io.BytesIO()
            np.savez_compressed(stream,**part)
            payload=stream.getvalue()
            with db:
                db.execute("INSERT INTO matching VALUES(?,?,?,?)",(rec["stem"],rec["annotation_sha256"],hashlib.sha256(payload).hexdigest(),payload))
        parts.append(part)
        progress(out,"METRIK",i,len(records),i)
    metrics={m:aggregate(parts,m) for m in METHODS}
    s.require(abs(metrics["yolo1280"]["AP"]-BASELINE_AP)<1e-8,
              "Eski YOLO1280 gelistirme AP'si yeniden elde edilemedi.")
    report={"protocol_sha256":signature,"methods":metrics,"images":len(records),
            "metric":"same pinned VisDrone DET toolkit-aligned Python AP; percent units; not COCO mAP",
            "separate_reference_binary_validation":False}
    s.write_json(out/"metrics.json",report)
    with (out/"metrics.csv").open("w",encoding="utf-8-sig",newline="") as file:
        keys=["AP","AP50","AP75","AR1","AR10","AR100","AR500"]
        writer=csv.writer(file);writer.writerow(["method",*keys])
        for m in METHODS:writer.writerow([m,*(metrics[m][k] for k in keys)])
    print(f"NOKTA SONUCU: HyCADO1280 - YOLO1280 = {metrics['hybrid1280']['AP']-metrics['yolo1280']['AP']:+.3f} AP",flush=True)
    return parts,metrics


def bootstrap(db,records,parts,metrics,out):
    import numpy as np
    from hycado_start import bootstrap_statistics as b
    groups=b.make_groups(records)
    s.write_json(out/"bootstrap_groups.json",groups)
    engine=b.compile_engine(parts,METHODS)
    point=b.evaluate(engine,np.ones(len(records),dtype=np.int64))
    expected=np.array([[metrics[m][k] for k in b.KEYS] for m in METHODS])
    s.require(np.allclose(point,expected,atol=1e-8,rtol=0),"Bootstrap nokta esligi basarisiz.")
    draws=[]
    for i in range(DRAWS):
        saved=db.execute("SELECT payload FROM bootstrap WHERE draw=?",(i,)).fetchone()
        if saved:
            envelope=json.loads(saved[0])
            s.require(envelope["sha256"]==s.fingerprint({"draw":i,"value":envelope["value"]}),"Bootstrap kaydi bozulmus.")
            v=np.asarray(envelope["value"],dtype=float) if envelope["value"] is not None else np.full(point.shape,np.nan)
            s.require(v.shape==point.shape,"Bootstrap kayit bicimi farkli.")
        else:
            v=b.evaluate(engine,b.frequencies(groups,"cluster",i,seed=20261102))
            payload=v.tolist() if np.isfinite(v).all() else None
            envelope={"value":payload,"sha256":s.fingerprint({"draw":i,"value":payload})}
            with db:db.execute("INSERT INTO bootstrap VALUES(?,?)",(i,json.dumps(envelope,allow_nan=False)))
            progress(out,"BOOTSTRAP",i+1,DRAWS,i+1)
        draws.append(v)
    arr=np.stack(draws);valid=np.isfinite(arr).all(axis=(1,2));arr=arr[valid]
    s.require(len(arr)>=2,"Tanimli bootstrap tekrar sayisi yetersiz.")
    contrasts={}
    for j in (0,1):
        diff=arr[:,2,:]-arr[:,j,:]
        low,high=np.quantile(diff,[.025,.975],axis=0,method="linear")
        contrasts["hybrid1280_minus_"+METHODS[j]]={
            k:{"point":float(point[2,i]-point[j,i]),"lower95":float(low[i]),"upper95":float(high[i])}
            for i,k in enumerate(b.KEYS)}
    result={"replicates_requested":DRAWS,"valid_replicates":int(valid.sum()),"proxy_cluster_count":len(groups["labels"]),
            "differences_pp":contrasts,"scope":"exploratory, previously examined TRAIN development, fixed weights",
            "interval":"paired proxy-cluster percentile 95%; no training seed uncertainty"}
    s.write_json(out/"intervals.json",result)
    return result


def bundle(out):
    files=["protocol.json","19_onkontrol.json","prediction_index.json","metrics.json","metrics.csv",
           "intervals.json","bootstrap_groups.json","result.json"]
    temp=out/(REPORT+".partial")
    with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED) as z:
        for n in files:z.write(out/n,n)
    os.replace(temp,out/REPORT)


def run(args,root,out):
    f,records,weights,pilot,data,p2,protocol=prepare(root,out)
    signature=s.fingerprint(protocol)
    check=out/"19_onkontrol.json"
    with closing(s.open_cache(out/"predictions.sqlite3",signature)) as db:
        if args.mode=="preflight":
            if check.exists():
                prior=s.read(check)
                s.require(prior["status"]=="PREFLIGHT_OK" and prior["protocol_sha256"]==signature,"Onkontrol protokolu farkli.")
                print(f"ON KONTROL ONCEDEN TAMAM: {check}",flush=True)
                return
            sample=sorted((r for r in records if r["image"] in pilot),key=lambda r:r["image"])[:8]
            s.require(len(sample)==8,"Onceki pilotla ortak 8 goruntu gerekli.")
            yolo_parity(sample,f,weights,data,p2,out)
            pending_count=sum(s.cache_rows(db,"hybrid1280",r) is None for r in sample)
            secs,calls=predict_crops(db,sample,f,weights,data,p2,pilot,out)
            for r in sample:
                cached=s.cache_rows(db,"hybrid1280",r)
                s.require(s.fingerprint({"detections":cached["detections"],"rects":cached["rects"]})==pilot[r["image"]]["sha256"],
                          "Korunan pilot tahmini onceki pilotla eslesmedi.")
            s.write_json(check,{"status":"PREFLIGHT_OK","protocol_sha256":signature,
                         "fresh_yolo1280_exact_images":3,"crop_pilot_exact_images":len(sample),
                         "estimated_remaining_inference_hours":secs/pending_count*(len(records)-len(sample))/3600 if pending_count else None,
                         "estimate_excludes":"setup, CPU metrics/bootstrap; nonrepresentative pilot subset",
                         "new_crop_calls":calls,"annotation_rows_parsed":False})
            estimate=s.read(check)["estimated_remaining_inference_hours"]
            print(f"ON KONTROL TAMAM: {check}",flush=True)
            if estimate is not None:
                print(f"Kalan cikarim tahmini ~{estimate:.2f} saat (CPU metrik/GA haric).",flush=True)
            else:
                print("Pilot tahminleri kayittan korundu; bu cagrida sure olculmedi.",flush=True)
            return
        pre=s.read(check)
        s.require(pre["status"]=="PREFLIGHT_OK" and pre["protocol_sha256"]==signature,"Once 19A onkontrol gerekli.")
        predict_crops(db,records,f,weights,data,p2,pilot,out)
        index=freeze(db,records,f,p2,signature,out)
        parts,metrics=measure(db,records,root,out,signature)
        ci=bootstrap(db,records,parts,metrics,out)
    s.write_json(out/"result.json",{"status":"COMPLETED","protocol_sha256":signature,"images":N,
              "crops":sum(r["crops"] for r in index["records"]),
              "AP_gain_vs_yolo1280_pp":metrics["hybrid1280"]["AP"]-metrics["yolo1280"]["AP"],
              "AP_gain_vs_nms_only_pp":metrics["hybrid1280"]["AP"]-metrics["yolo1280_nms_only"]["AP"],
              "primary_AP_interval":ci["differences_pp"]["hybrid1280_minus_yolo1280"]["AP"],
              "training":False,"official_val_images_used":False,"testdev_used":False,
              "independent_test":False,"latency_measured":False,
              "completed_utc":s.utc_now(),"prediction_database_sha256":s.sha256(out/"predictions.sqlite3")})
    bundle(out)
    print(f"SONUC: COMPLETED\nGONDERILECEK RAPOR: {out/REPORT}",flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=["preflight","full"])
    ap.add_argument("--project",required=True,type=Path)
    args=ap.parse_args();root=args.project.resolve();out=root/"experiments"/RUN
    if not (root/"experiments").is_dir():
        print(f"Proje klasoru bulunamadi: {root}")
        return 1
    out.mkdir(exist_ok=True)
    try:
        with s.training_lock(out/"run.lock"):
            try:run(args,root,out)
            except (Exception,KeyboardInterrupt) as exc:
                s.write_json(out/"error.json",{"status":"STOPPED","error":str(exc),"traceback":traceback.format_exc()})
                raise
        return 0
    except (Exception,KeyboardInterrupt):
        traceback.print_exc()
        print(f"DURDU. Hata raporu: {out/'error.json'}",flush=True)
        return 1

if __name__=="__main__":
    raise SystemExit(main())
