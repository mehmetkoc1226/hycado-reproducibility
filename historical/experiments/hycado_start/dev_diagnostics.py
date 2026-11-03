"""CPU-only read-only analysis of completed 11 predictions; no threshold tuning.

Final cached detections cannot reconstruct raw crop queries or establish the
causal effect of NMS. All source-attribution fields describe observed retention,
not a counterfactual fusion experiment. No Torch or inference model is loaded.
"""
from __future__ import annotations
from collections import Counter
import contextlib
import csv
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import traceback
import zipfile
import numpy as np

from .common import ROOT, write_json, sha256, utc_now
from .stage1 import fingerprint, training_lock
from .crop_loader import safe_path, read_json
from .dev_evaluation import cached_payload
from .dev_metrics import raw_annotation, xyxy_to_det, ignore_integral, outside_ignore, prepare_groundtruth, overlaps, match_flags

SOURCE_PROTOCOL = '57553b9f0b71d6da544687d4581bd4125dbfc9198e9e4862a4bc9f73dcd97126'
SOURCE_INDEX = 'cadc323f6f3542eee3d22f3a5d7bfc1ddd349504ac75fe4f808999fcc18ee944'
BINS = [0., .01, .05, .1, .25, .5, .75, .9, 1.]


def detailed_matches(det, gt, threshold):
    """Return flags plus original GT row identities; cross-check old 11 matcher."""
    flags = np.zeros(len(det), dtype=np.int8)
    matched_ids = np.full(len(det), -1, dtype=np.int64)
    for cls in range(1,11):
        did = np.flatnonzero(det[:,5] == cls)
        gid = np.flatnonzero(gt[:,5] == cls)
        gid = gid[np.argsort(gt[gid,4], kind='stable')]
        state = -gt[gid,4].astype(np.int8)
        matrix = overlaps(det[did,:5],gt[gid,:5])
        for j, d in enumerate(did):
            available = np.flatnonzero((state == 0) & (matrix[j] >= threshold))
            if len(available):
                best_overlap = matrix[j,available].max()
                best = available[matrix[j,available] == best_overlap][-1]
                flags[d] = 1; matched_ids[d] = gid[best]; state[best] = 1
            elif np.any((state == -1) & (matrix[j] >= threshold)):
                flags[d] = -1
        expected = match_flags(det[did,:5],gt[gt[:,5] == cls,:5],threshold)
        if not np.array_equal(flags[did],expected):
            raise ValueError('Ayrintili eslesme, 11 eslesmesiyle tutarsiz')
    return flags, matched_ids


def false_positive_reasons(det, gt, flags, threshold):
    normal = gt[(gt[:,4] == 0) & (gt[:,5] >= 1) & (gt[:,5] <= 10)]
    ids = np.flatnonzero(flags == 0)
    result = [''] * len(det)
    matrix = overlaps(det[ids,:5],normal[:,:5])
    for j,d in enumerate(ids):
        same = normal[:,5] == det[d,5]
        best_same = float(matrix[j,same].max()) if same.any() else 0.
        best_other = float(matrix[j,~same].max()) if (~same).any() else 0.
        if best_same >= threshold: reason = 'duplicate_same_class_overlap'
        elif best_other >= threshold: reason = 'other_class_overlap'
        elif best_same >= .1: reason = 'same_class_localization_0.1_to_threshold'
        else: reason = 'background_or_unexplained'
        result[d] = reason
    return result


def counts(flags):
    return {'TP': int(np.count_nonzero(flags == 1)), 'FP': int(np.count_nonzero(flags == 0)),
            'ignored_detections': int(np.count_nonzero(flags == -1))}


def analyze_image(gt, baseline_all, hybrid_all, threshold):
    base, hybrid = baseline_all[:500], hybrid_all[:500]
    data={}; rows=[]
    for variant,det in [('baseline',base),('hycado',hybrid)]:
        flags,ids=detailed_matches(det,gt,threshold)
        reasons=false_positive_reasons(det,gt,flags,threshold)
        data[variant]={'flags':flags,'ids':ids,'reasons':reasons}
        for rank,(box,flag,reason) in enumerate(zip(det,flags,reasons),1):
            # Plain JSON-compatible data; no GT box coordinates need to be uploaded.
            rows.append({'variant':variant,'rank':rank,'score':float(box[4]),'class_id':int(box[5]),
                         'match_flag':int(flag),'fp_reason':reason})
    base_ids=set(map(int,data['baseline']['ids'][data['baseline']['flags']==1]))
    hybrid_ids=set(map(int,data['hycado']['ids'][data['hycado']['flags']==1]))
    gained, lost = hybrid_ids-base_ids, base_ids-hybrid_ids
    hybrid_position={tuple(row):i for i,row in enumerate(hybrid_all)}
    lost_observations=Counter()
    for d in np.flatnonzero(data['baseline']['flags']==1):
        if int(data['baseline']['ids'][d]) not in lost: continue
        position=hybrid_position.get(tuple(base[d]))
        if position is None: reason='baseline_TP_box_absent_from_final_hybrid'
        elif position>=500: reason='baseline_TP_box_below_global500_cutoff'
        else: reason='baseline_TP_box_within500_but_GT_unmatched_after_greedy_matching'
        lost_observations[reason]+=1
    normal_count=int(np.count_nonzero((gt[:,4]==0) & (gt[:,5]>=1) & (gt[:,5]<=10)))
    summary={'normal_GT':normal_count,'gained_GT':len(gained),'lost_GT':len(lost),
             'shared_matched_GT':len(base_ids & hybrid_ids),
             'lost_GT_observations':dict(lost_observations)}
    base_keys=set(map(tuple,baseline_all)); hybrid_keys=set(map(tuple,hybrid_all))
    summary['post_ignore_final_box_retention']={
        'baseline_boxes':len(baseline_all),'hybrid_boxes':len(hybrid_all),
        'exact_baseline_rows_retained':len(base_keys & hybrid_keys),
        'baseline_rows_absent':len(base_keys-hybrid_keys),
        'hybrid_rows_not_exactly_in_baseline':len(hybrid_keys-base_keys)}
    for variant,det in [('baseline',base),('hycado',hybrid)]:
        flags=data[variant]['flags']
        summary[variant]={**counts(flags),'FN_normal_GT':normal_count-int(np.count_nonzero(flags==1)),
            'FP_reasons':dict(Counter(x for x in data[variant]['reasons'] if x)),
            'rank_prefixes':{str(k):counts(flags[:k]) for k in [1,10,100,500]}}
    return summary,rows


def initialize_totals():
    return {v:{'counts':Counter(),'FP_reasons':Counter(),
               'rank_prefixes':{str(k):Counter() for k in [1,10,100,500]},
               'score_bins':[Counter() for _ in range(len(BINS)-1)],
               'per_class':{str(c):Counter() for c in range(1,11)}} for v in ['baseline','hycado']}


def accumulate(totals, summary, rows):
    for v in ['baseline','hycado']:
        totals[v]['counts'].update({k:summary[v][k] for k in ['TP','FP','ignored_detections','FN_normal_GT']})
        totals[v]['FP_reasons'].update(summary[v]['FP_reasons'])
        for k,values in summary[v]['rank_prefixes'].items():totals[v]['rank_prefixes'][k].update(values)
    for row in rows:
        flag='TP' if row['match_flag']==1 else 'FP' if row['match_flag']==0 else 'ignored_detections'
        idx=min(len(BINS)-2,int(np.searchsorted(BINS,row['score'],side='right'))-1)
        bucket=totals[row['variant']]['score_bins'][max(0,idx)]
        bucket[flag]+=1
        totals[row['variant']]['per_class'][str(row['class_id'])][flag]+=1


def finalize(totals):
    result={}
    for variant,t in totals.items():
        count=dict(t['counts']);den=count.get('TP',0)+count.get('FP',0)
        count['precision_at_fixed_max500']=count.get('TP',0)/den if den else None
        recall_den=count.get('TP',0)+count.get('FN_normal_GT',0)
        count['recall_normal_GT_at_fixed_max500']=count.get('TP',0)/recall_den if recall_den else None
        bins=[]
        for i,b in enumerate(t['score_bins']):
            denom=b['TP']+b['FP']
            bins.append({'lower_inclusive':BINS[i],'upper':BINS[i+1],
                         'upper_inclusive':i==len(BINS)-2,**dict(b),
                         'observed_precision':b['TP']/denom if denom else None})
        result[variant]={'counts':count,'FP_reasons':dict(t['FP_reasons']),
            'rank_prefixes':{k:dict(v) for k,v in t['rank_prefixes'].items()},
            'score_bins':bins,'per_class':{k:dict(v) for k,v in t['per_class'].items()}}
    return result


def validate_source(source):
    cfg=read_json(source/'protocol.json');index=read_json(source/'prediction_index.json')
    result=read_json(source/'result.json')
    if (fingerprint(cfg)!=SOURCE_PROTOCOL or sha256(source/'prediction_index.json')!=SOURCE_INDEX or
            index['protocol_sha256']!=SOURCE_PROTOCOL or result.get('status')!='COMPLETED' or
            result.get('development_images')!=647 or result.get('crops')!=10682 or
            result.get('official_val_used') is not False):
        raise ValueError('Tamamlanan sabit 11 sonucu gerekli')
    if importlib.metadata.version('numpy')!=cfg['runtime_versions']['numpy']:
        raise ValueError('Sabit NumPy surumu farkli')
    for name,digest in cfg['code_hashes'].items():
        if Path(name).name!=name or sha256(Path(__file__).with_name(name))!=digest:
            raise ValueError('11 olcum kaynak kodu degismis: '+name)
    records=index['records']
    if (len(records)!=647 or len({r['source_image'] for r in records})!=647 or
            any(not r['source_image'].startswith('images/train/') for r in records)):
        raise ValueError('647 benzersiz resmi TRAIN gelistirme kaydi gerekli')
    return cfg,index


def diagnostic_analysis(source,output,index,signature):
    thresholds=[.5,.75]
    totals={str(t):initialize_totals() for t in thresholds}
    transitions={str(t):Counter() for t in thresholds};observations={str(t):Counter() for t in thresholds}
    retention=Counter();image_rows=[];fingerprints=[];crop_count=0
    source=Path(source);output=Path(output)
    for i,item in enumerate(index['records'],1):
        cache=safe_path(source,item['path'])
        annotation=source/'reference_annotations'/(cache.stem+'.txt')
        if sha256(cache)!=item['sha256'] or sha256(annotation)!=item['annotation_sha256']:
            raise ValueError('Kaynak tahmin/annotation degismis: '+item['source_image'])
        p=cached_payload(cache,signature,item['source_image'])
        crop_count+=p['all_crops_processed']
        raw=raw_annotation(annotation.read_text(encoding='utf-8-sig'))
        integral=ignore_integral(raw,p['width'],p['height']);gt=prepare_groundtruth(raw,integral)
        dets={v:outside_ignore(xyxy_to_det(p[v]),integral) for v in ['baseline','hycado']}
        del integral
        for threshold in thresholds:
            summary,rows=analyze_image(gt,dets['baseline'],dets['hycado'],threshold)
            accumulate(totals[str(threshold)],summary,rows)
            transitions[str(threshold)].update({k:summary[k] for k in ['normal_GT','gained_GT','lost_GT','shared_matched_GT']})
            observations[str(threshold)].update(summary['lost_GT_observations'])
            if threshold==.5:retention.update(summary['post_ignore_final_box_retention'])
            image_rows.append({'source_image':item['source_image'],'IoU':threshold,
                               'baseline_TP':summary['baseline']['TP'],'baseline_FP':summary['baseline']['FP'],
                               'hycado_TP':summary['hycado']['TP'],'hycado_FP':summary['hycado']['FP'],
                               'gained_GT':summary['gained_GT'],'lost_GT':summary['lost_GT']})
        fingerprints.extend([(cache,item['sha256']),(annotation,item['annotation_sha256'])])
        if i==1 or i%25==0 or i==len(index['records']):
            print(f'CPU TANI: {i}/{len(index["records"])} goruntu; esikler degistirilmiyor.',flush=True)
            write_json(output/'progress.json',{'status':'ANALYZING','images_done':i,'images_total':len(index['records'])})
    for path,digest in fingerprints:
        if sha256(path)!=digest:raise ValueError('Kaynak dosya analiz sirasinda degismis')
    result={'source_protocol_sha256':signature,'images':len(index['records']),'crops_in_source_cache':crop_count,
        'training_started':False,'inference_started':False,'official_val_used':False,
        'parameter_search':False,'official_binary_parity_verified':False,
        'meaning':'post-fusion diagnostic observations; not causal NMS attribution or new AP evaluation',
        'rank_cap':500,'FP_reason_policy':'same-class duplicate; otherwise other-class overlap; otherwise same-class IoU>=0.1; otherwise background/unexplained',
        'raw_crop_queries_available':False,'per_IoU':{
            key:{'detectors':finalize(totals[key]),'GT_transitions':dict(transitions[key]),
                 'lost_GT_observations':dict(observations[key])} for key in totals},
        'post_ignore_final_box_retention':dict(retention),'source_files_rehashed_after_analysis':True}
    write_json(output/'diagnostics.json',result)
    with (output/'per_image.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['source_image','IoU','baseline_TP','baseline_FP','hycado_TP','hycado_FP','gained_GT','lost_GT']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(image_rows)
    return result


def bundle(output):
    import os
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/'11B_tani_rapor.zip';fd,name=tempfile.mkstemp(dir=output,prefix='11B_',suffix='.zip.tmp');os.close(fd)
    try:
        with zipfile.ZipFile(name,'w',zipfile.ZIP_DEFLATED) as z:
            for n in ['protocol.json','result.json','progress.json','diagnostics.json','per_image.csv','source_metrics.json']:
                if (output/n).exists():z.write(output/n,n)
            if (output/'console.log').exists():
                with (output/'console.log').open('rb') as f:
                    f.seek(max(0,(output/'console.log').stat().st_size-50000));z.writestr('console_tail.log',f.read())
        os.replace(name,target)
    finally:Path(name).unlink(missing_ok=True)
    return target


def run():
    from .__main__ import _Tee
    source=ROOT/'experiments/development_eval_v1_seed42'
    output=ROOT/'experiments/development_diagnostics_v1_seed42';output.mkdir(parents=True,exist_ok=True)
    with training_lock(output/'.diagnostic.lock'),training_lock(source/'.evaluation.lock'):
        with (output/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    print('11B CPU TANI:',utc_now(),flush=True)
                    print('Model yuklenmez; egitim/cikarim/esik aramasi yapilmaz.',flush=True)
                    cfg,index=validate_source(source)
                    names=['protocol.json','prediction_index.json','result.json','metrics.json','metrics.csv']
                    before={n:sha256(source/n) for n in names}
                    config={'id':'hycado_development_diagnostics_v1','date':'2026-10-29',
                            'source_protocol_sha256':SOURCE_PROTOCOL,'source_metadata_hashes':before,
                            'fixed_IoUs':[.5,.75],'fixed_global_cap':500,'score_histogram_edges':BINS,
                            'parameter_search':False,'training_started':False,'inference_started':False,
                            'numpy_version':importlib.metadata.version('numpy'),
                            'code_sha256':sha256(__file__)}
                    if (output/'protocol.json').exists() and read_json(output/'protocol.json')!=config:
                        raise ValueError('11B kodu veya kaynak kimligi degismis; raporlar karistirilmadi')
                    write_json(output/'protocol.json',config)
                    for n in ['diagnostics.json','per_image.csv']:(output/n).unlink(missing_ok=True)
                    write_json(output/'result.json',{'status':'RUNNING'})
                    data=diagnostic_analysis(source,output,index,SOURCE_PROTOCOL)
                    if data['images']!=647 or data['crops_in_source_cache']!=10682:
                        raise ValueError('647 goruntu / 10682 crop kaynak sayaci farkli')
                    if before!={n:sha256(source/n) for n in names}:raise ValueError('11 kaynak metaverisi degismis')
                    write_json(output/'source_metrics.json',read_json(source/'metrics.json'))
                    result={'status':'COMPLETED','images':647,'source_11_unchanged':True,
                            'training_started':False,'inference_started':False,'official_val_used':False}
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();result={'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',
                        'error_type':type(exc).__name__,'error':str(exc)}
                write_json(output/'result.json',result);write_json(output/'progress.json',result)
                print('SONUC:',result['status'],flush=True)
        print('GONDERILECEK RAPOR:',bundle(output),flush=True)
        return result['status']=='COMPLETED'


def main():
    if '--report-only' in sys.argv:
        print(bundle(ROOT/'experiments/development_diagnostics_v1_seed42'),flush=True);return 0
    return 0 if run() else 1


if __name__=='__main__':raise SystemExit(main())
