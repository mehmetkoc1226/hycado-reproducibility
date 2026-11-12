"""Read-only replay on the 647 TRAIN-internal development images."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import traceback
import zipfile

import numpy as np

OUT = 'p4_yolo1280_cpu_fusion_v1_seed42'
VARIANTS = ('yolo1280', 'yolo1280_nms_only', 'hycado800_a025', 'yolo1280_old_rois_a025')
P2 = 'p2_strong_baselines_v1_seed42'
PILOT_A = 'p3_crop_budget_cpu_v1_seed42'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(tmp,path)


def checked_file(path, expected):
    if sha(path) != expected:
        raise ValueError('Kaynak SHA256 farkli: '+str(path))
    return path


def safe(base, rel):
    p = PurePosixPath(rel)
    if not p.parts or p.is_absolute() or '..' in p.parts or '\\' in rel or ':' in rel:
        raise ValueError('Beklenmeyen kaynak yolu: '+rel)
    target = base.joinpath(*p.parts).resolve()
    if not target.is_relative_to(base.resolve()):
        raise ValueError('Kaynak yolu klasor disinda')
    return target


def variant_rows(yolo, baseline, candidates, origins, w, h, nms):
    a = np.asarray(candidates)
    o = np.asarray(origins)
    if (a.dtype != np.float64 or a.ndim != 2 or a.shape[1] != 6 or
            o.shape != (len(a),3) or not np.isfinite(a).all() or not np.isin(o[:,0],[0,1]).all()):
        raise ValueError('11C aday dizisi gecersiz')
    if a[o[:,0]==0].tolist() != baseline:
        raise ValueError('11C ve 11 YOLO-800 adaylari farkli')
    crop = a[o[:,0]==1].copy()
    if np.any((crop[:,4]<=.001)|(crop[:,4]>1)):
        raise ValueError('11C crop skor araligi farkli')
    crop[:,4] *= .25  # Do not apply a second confidence cutoff.
    c = crop.tolist()
    return dict(zip(VARIANTS, (yolo, nms(yolo,w,h), nms(baseline+c,w,h), nms(yolo+c,w,h))))


def match(s, variants, annotation, w, h):
    raw=s.raw_annotation(annotation.read_text(encoding='utf-8-sig'))
    integral=s.ignore_integral(raw,w,h); gt=s.prepare_groundtruth(raw,integral)
    result={'gt_denominators':np.array([np.count_nonzero(gt[:,5]==c) for c in range(1,11)],dtype=np.int64)}
    result['presence']=(result['gt_denominators']>0).astype(np.int64)
    for name,rows in variants.items():
        det=s.outside_ignore(s.xyxy_to_det(rows),integral)[:500]
        result[name+'_scores']=det[:,4].copy()
        result[name+'_classes']=det[:,5].astype(np.int8)
        result[name+'_flags']=np.column_stack([s.fast_flags(det,gt,float(t)) for t in s.THRESHOLDS])
    return result


def run(root,out):
    refs=read(Path(__file__).with_name('source_hashes.json'))
    for name,digest in refs['code'].items():checked_file(root/'hycado_start'/name,digest)
    sys.path.insert(0,str(root))
    from hycado_start import score_ablation as s, dev_evaluation as old
    source,capture,index,cidx=s.validate_capture(root)
    exp=root/'experiments';p2=exp/P2
    p2m=read(checked_file(p2/'development_metrics.json',refs['p2_metrics']))
    pa=read(checked_file(exp/PILOT_A/'metrics.json',refs['pilot_a_metrics']))
    manifest=checked_file(exp/'stage1_v1_seed42/data/manifest.json',refs['manifest'])
    records=sorted((r for r in read(manifest)['records'] if r['split']=='val'),key=lambda r:r['source_image'])
    pp=read(p2/'development_protocol.json');pi_path=p2/'development_prediction_index.json';pi=read(pi_path)
    p2_sig=fingerprint(pp)
    if (pp['id']!='hycado_p2_development_v1' or pp['manifest_sha256']!=refs['manifest'] or
            pp['checkpoint_sha256']!=refs['checkpoint'] or pp['script_sha256']!=refs['p2_script'] or
            pp['official_val_used'] is not False or pp['training'] is not False or pp['images']!=647 or
            pi['protocol_sha256']!=p2_sig or pi['images_per_method']!=647 or pi['yolo800_exact_parity'] is not True):
        raise ValueError('P2 gelistirme protokolu/indeksi uyumsuz')
    p2rows=pi['methods']['yolo1280'];ids=[r['source_image'] for r in records]
    oldmap={r['source_image']:r for r in index['records']};cmap={r['source_image']:r for r in cidx['records']}
    if (len(ids)!=647 or len(set(ids))!=647 or any(not x.startswith('images/train/') for x in ids) or
            [r['source_image'] for r in p2rows]!=ids or set(ids)!=set(oldmap) or set(ids)!=set(cmap)):
        raise ValueError('Ayni 647 gelistirme goruntusu gerekli')
    protocol={'id':'p4_yolo1280_old_roi_cpu_replay_v1','images':647,'crop_weight':.25,
              'scope':'TRAIN-internal development, exploratory; original YOLO-800 ROIs',
              'script_sha256':sha(Path(__file__)),'refs':refs,'p2_index_sha256':sha(pi_path),
              'p2_protocol_sha256':p2_sig,'capture_index_sha256':s.CAPTURE_INDEX,
              'variants':list(VARIANTS),'training':False,'inference':False,'latency_measured':False}
    proto=out/'protocol.json'
    if proto.exists() and read(proto)!=protocol:raise ValueError('Bu cikti klasorunde farkli protokol var')
    write(proto,protocol);sig=fingerprint(protocol)
    parts=[];crops=0;nms_changed=0;reused=0
    for i,(record,new) in enumerate(zip(records,p2rows),1):
        ident=record['source_image'];item=oldmap[ident];cap=cmap[ident]
        predpath=checked_file(safe(source,item['path']),item['sha256'])
        metapath=checked_file(safe(capture,cap['path']),cap['sha256'])
        archive=checked_file(metapath.parent/'queries_and_trace.npz',cap['archive_sha256'])
        p2path=checked_file(safe(p2,new['path']),new['sha256'])
        annotation=checked_file(source/'reference_annotations'/(predpath.stem+'.txt'),item['annotation_sha256'])
        if item['annotation_sha256']!=record['annotation_sha256']:raise ValueError('P2 ve 11 GT farkli: '+ident)
        p=read(p2path);body={k:v for k,v in p.items() if k!='content_sha256'}
        prior=old.cached_payload(predpath,s.SOURCE_PROTOCOL,ident);meta=read(metapath)
        w,h=prior['width'],prior['height'];count=meta['crop_count'];yolo=p['detections']
        if (p['content_sha256']!=fingerprint(body) or p['protocol_sha256']!=p2_sig or p['method']!='yolo1280' or
                p['source_image']!=ident or p['image_sha256']!=record['image_sha256'] or
                (p['width'],p['height'])!=(w,h) or len(yolo)!=new['detections'] or len(yolo)>1000):
            raise ValueError('YOLO-1280 tahmin kimligi farkli: '+ident)
        if (meta['source_image']!=ident or meta['protocol_sha256']!=s.CAPTURE_SIGNATURE or
                meta['old11_prediction_sha256']!=item['sha256'] or meta['archive_sha256']!=cap['archive_sha256'] or
                meta['source_crop_document_sha256']!=prior['source_crop_document_sha256'] or
                count!=prior['all_crops_processed'] or meta['diagnostics']['original11_exactly_reproduced'] is not True):
            raise ValueError('Ham crop kaydi farkli: '+ident)
        clean,dropped=old.canonical_detections(yolo,w,h)
        if clean!=yolo or dropped:raise ValueError('YOLO-1280 kutulari gecersiz: '+ident)
        folder=out/'sources'/predpath.stem
        bindings={'protocol':sig,'source_image':ident,'yolo1280':new['sha256'],'archive':cap['archive_sha256'],
                  'original':item['sha256'],'annotation':item['annotation_sha256'],'capture':cap['sha256']}
        if folder.exists():
            saved=read(folder/'summary.json')
            if saved['bindings']!=bindings or saved['content_sha256']!=fingerprint({k:v for k,v in saved.items() if k!='content_sha256'}):
                raise ValueError('Kayitli CPU ozeti farkli: '+ident)
            with np.load(checked_file(folder/'matches.npz',saved['npz_sha256']),allow_pickle=False) as z:data={k:z[k] for k in z.files}
            reused+=1
        else:
            with np.load(archive,allow_pickle=False) as z:
                variants=variant_rows(yolo,prior['baseline'],z['candidate_rows'],z['candidate_origins'],w,h,old.classwise_nms)
            data=match(s,variants,annotation,w,h);data['nms_changed']=np.array(variants['yolo1280_nms_only']!=yolo)
            s.save_cache(folder,data,bindings)
        parts.append(data);crops+=count;nms_changed+=int(data['nms_changed'])
        if i==1 or i%25==0 or i==647:
            print(f'CPU fuzyon: {i}/647 | kayittan {reused}',flush=True)
            write(out/'progress.json',{'images_done':i,'images_total':647,'reused':reused})
    if (len(parts),crops)!=(647,10682):raise ValueError('Toplam goruntu/crop kapsami farkli')
    metrics={name:s.aggregate(parts,name) for name in VARIANTS}
    s.compare_metrics(metrics['yolo1280'],p2m['methods']['yolo1280'])
    s.compare_metrics(metrics['hycado800_a025'],pa['all_a025'])
    fusion=metrics[VARIANTS[-1]]['AP'];base=metrics['yolo1280']['AP'];control=metrics['yolo1280_nms_only']['AP']
    write(out/'metrics.json',metrics)
    return {'status':'COMPLETE','images':647,'crops':crops,'nms_only_changed_images':nms_changed,
            'baseline_checks':'P2 YOLO-1280 and Pilot A all crops matched',
            'AP_gain_vs_yolo1280_pp':fusion-base,'AP_gain_vs_nms_only_pp':fusion-control,
            'training':False,'inference':False,'latency_measured':False,'official_val_used':False}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=Path(r'D:\HyCADO_Temiz_Baslangic'))
    root=ap.parse_args().project.resolve();out=root/'experiments'/OUT;out.mkdir(parents=True,exist_ok=True)
    try: result=run(root,out)
    except (Exception,KeyboardInterrupt) as e:
        traceback.print_exc();result={'status':'ERROR','error':str(e),'traceback':traceback.format_exc()}
    write(out/'result.json',result)
    target=out/'14_yolo1280_cpu_fuzyon_rapor.zip';tmp=target.with_suffix('.partial')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for n in ('protocol.json','progress.json','metrics.json','result.json'):
            if (out/n).is_file():z.write(out/n,n)
    os.replace(tmp,target)
    print('SONUC:',result['status'],'\nGONDERILECEK RAPOR:',target,flush=True)
    return 0 if result['status']=='COMPLETE' else 1

if __name__=='__main__':raise SystemExit(main())
