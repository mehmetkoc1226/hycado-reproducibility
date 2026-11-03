"""Read-only size/GT-count diagnostics of frozen 12 VAL predictions.

Descriptive recall at fixed global max500, not AP by area. GT is matched once
on the full image before size grouping; FP is never assigned a GT size.
"""
from __future__ import annotations
import contextlib
import csv
from pathlib import Path
import sys
import traceback
from collections import Counter
import numpy as np
from .common import ROOT, sha256, write_json, utc_now
from .stage1 import fingerprint, training_lock
from .crop_loader import safe_path, read_json
from . import optimized_runtime as o
from .dev_diagnostics import detailed_matches, counts
from .dev_metrics import raw_annotation, ignore_integral, prepare_groundtruth, outside_ignore, xyxy_to_det

OUT = 'size_diagnostics_v1_seed42'
THRESHOLDS = (.5, .75)
AREAS = ['nonpositive_box', 'lt16_squared', '16_to_32_squared', '32_to_96_squared', 'ge96_squared']
GROUPS = ['zero_GT', '1_to_30_GT', '31_to_100_GT', '101_plus_GT']
COUNT_KEYS = ['normal_GT', 'baseline_matched_GT', 'hybrid_matched_GT', 'gained_GT', 'lost_GT', 'shared_matched_GT']
FILES15 = {
    'protocol.json':'ba50e80e4541156a1e86b1223e5226d9054b8a0e83efbfc514275cfae399225c',
    'result.json':'99a37f171007b45b83b1ffcb884d8f24ca694fae9a0a436fb621eff47d12da21',
    'summary.json':'a4d244cd95c4e2bd45b6c5c9adf979c43b7e0611b10feb5ca9882038f8b5ffc5'}


def source(root):
    cfg12, base, w1, w2, official, upstream = o.sources(root)
    folder = root/'experiments'/o.OUT
    for name,digest in FILES15.items():
        if sha256(folder/name) != digest: raise ValueError('Dogrulanmis 15 kaydi farkli: '+name)
    cfg15 = read_json(folder/'protocol.json')
    if cfg15['code_sha256'] != sha256(o.__file__) or cfg15['upstream_metadata_sha256'] != upstream:
        raise ValueError('15 kodu/kaynaklari degismis')
    result = read_json(folder/'result.json')
    if (result.get('status') != 'COMPLETED' or result.get('all_measured_outputs_exact_original12') is not True or
            result.get('profile_raw_queries_and_outputs_exact12') is not True):
        raise ValueError('Tamamlanan 15 eslik kontrolu gerekli')
    phases = [o.r.validate_phase(read_json(folder/f'pass_{p+1}_{v}.json'),cfg15,p,v) for p,v in o.r.PHASES]
    profile = o.validate_profile(read_json(folder/'crop_profile.json'),cfg15)
    if o.summary(phases,profile,read_json(root/'experiments'/o.r.OUT/'summary.json')) != read_json(folder/'summary.json'):
        raise ValueError('15 ozeti ham kayitlarla farkli')
    for name in list(FILES15)+['crop_profile.json']+[f'pass_{p+1}_{v}.json' for p,v in o.r.PHASES]:
        upstream[(folder/name).relative_to(root).as_posix()] = sha256(folder/name)
    # All source validation is CPU file hashing/JSON; no model is loaded.
    return cfg12, base, upstream


def size_group(width, height):
    if not np.isfinite([width,height]).all(): raise ValueError('GT kutu boyutu sonlu olmali')
    if width <= 0 or height <= 0: return 'nonpositive_box'
    area = width*height
    if area < 16**2: return 'lt16_squared'
    if area < 32**2: return '16_to_32_squared'
    if area < 96**2: return '32_to_96_squared'
    return 'ge96_squared'


def object_count_group(n):
    if n < 0: raise ValueError('GT sayisi negatif olamaz')
    if n == 0: return 'zero_GT'
    if n <= 30: return '1_to_30_GT'
    if n <= 100: return '31_to_100_GT'
    return '101_plus_GT'


def finish(values):
    if any(not isinstance(values.get(k,0),(int,np.integer)) or isinstance(values.get(k,0),bool) for k in COUNT_KEYS):
        raise ValueError('GT sayilari tamsayi olmali')
    out = {k:int(values.get(k,0)) for k in COUNT_KEYS}; den = out['normal_GT']
    b,h = out['baseline_matched_GT'],out['hybrid_matched_GT']
    if (any(x<0 for x in out.values()) or max(b,h)>den or
            b != out['shared_matched_GT']+out['lost_GT'] or h != out['shared_matched_GT']+out['gained_GT']):
        raise ValueError('GT eslesme sayilari tutarsiz')
    out.update({'baseline_recall_percent':100*b/den if den else None,
                'hybrid_recall_percent':100*h/den if den else None,
                'delta_recall_pp':100*(h-b)/den if den else None})
    return out


def subset_counts(ids, baseline_ids, hybrid_ids):
    ids = set(map(int,ids)); b = ids & baseline_ids; h = ids & hybrid_ids
    return finish({'normal_GT':len(ids),'baseline_matched_GT':len(b),'hybrid_matched_GT':len(h),
                   'gained_GT':len(h-b),'lost_GT':len(b-h),'shared_matched_GT':len(b&h)})


def analyze(raw, payload):
    integral = ignore_integral(raw,payload['width'],payload['height']);gt = prepare_groundtruth(raw,integral)
    normal = (gt[:,4] == 0) & (gt[:,5] >= 1) & (gt[:,5] <= 10)
    ids = np.flatnonzero(normal); sizes = np.array([size_group(*g[2:4]) for g in gt],dtype=object)
    dets = {v:outside_ignore(xyxy_to_det(payload[v]),integral)[:500] for v in ['baseline','hybrid_a025']}
    del integral
    output = {'normal_GT':int(len(ids)), 'object_count_group':object_count_group(len(ids)), 'thresholds':{}}
    for threshold in THRESHOLDS:
        matched={}; totals={}; per_class_counts={}
        for v,det in dets.items():
            flags, matched_ids = detailed_matches(det,gt,threshold)
            matched[v] = set(map(int,matched_ids[flags==1]))
            if not matched[v].issubset(set(map(int,ids))): raise ValueError('Eslesen GT normal olmali')
            totals[v] = counts(flags)
            if len(matched[v]) != totals[v]['TP']: raise ValueError('Bir GT birden fazla TP olamaz')
            per_class_counts[v] = {str(c):counts(flags[det[:,5]==c]) for c in range(1,11)}
        b,h = matched['baseline'],matched['hybrid_a025']; rows=[]
        for c in range(11):
            class_ids = ids if c==0 else np.flatnonzero(normal & (gt[:,5]==c))
            for area in ['all']+AREAS:
                selected = class_ids if area=='all' else class_ids[sizes[class_ids]==area]
                rows.append({'class_id':c,'area_group':area,**subset_counts(selected,b,h)})
        overall = subset_counts(ids,b,h)
        if overall['baseline_matched_GT'] != totals['baseline']['TP'] or overall['hybrid_matched_GT'] != totals['hybrid_a025']['TP']:
            raise ValueError('Genel TP ve eslesen GT sayisi farkli')
        output['thresholds'][str(threshold)] = {'overall':overall,'detections':totals,
                         'per_class_detection_counts':per_class_counts,'class_size_rows':rows}
    return output


def validate_analysis(data):
    n = data['normal_GT']
    if not isinstance(n,int) or n<0 or data['object_count_group']!=object_count_group(n): raise ValueError('GT sayi grubu farkli')
    if set(data['thresholds']) != {str(t) for t in THRESHOLDS}: raise ValueError('Tani esikleri farkli')
    for t,item in data['thresholds'].items():
        if finish(item['overall'])!=item['overall'] or item['overall']['normal_GT']!=n: raise ValueError('Genel GT ozeti farkli')
        rows=item['class_size_rows']; pairs={(x['class_id'],x['area_group']) for x in rows}
        if len(rows)!=66 or pairs!={(c,s) for c in range(11) for s in ['all']+AREAS}: raise ValueError('Boyut/sinif kapsami farkli')
        table={(x['class_id'],x['area_group']):x for x in rows}
        for row in rows:
            if any(row[k]!=v for k,v in finish(row).items()): raise ValueError('Boyut eslesme ozeti farkli')
        for c in range(11):
            for k in COUNT_KEYS:
                if sum(table[c,s][k] for s in AREAS)!=table[c,'all'][k]: raise ValueError('Boyut bolmeleri tum GT yi kapsamali')
        for s in ['all']+AREAS:
            for k in COUNT_KEYS:
                if sum(table[c,s][k] for c in range(1,11))!=table[0,s][k]: raise ValueError('Sinif toplami farkli')
        if {k:table[0,'all'][k] for k in COUNT_KEYS} != {k:item['overall'][k] for k in COUNT_KEYS}: raise ValueError('Genel/boyut toplami farkli')
        for variant,key in [('baseline','baseline_matched_GT'),('hybrid_a025','hybrid_matched_GT')]:
            d=item['detections'][variant]
            if set(d)!={'TP','FP','ignored_detections'} or any(not isinstance(x,int) or x<0 for x in d.values()): raise ValueError('Gecersiz TP/FP sayisi')
            if d['TP']!=item['overall'][key]: raise ValueError('TP/GT sayisi farkli')
            pc=item['per_class_detection_counts'][variant]
            if set(pc)!={str(c) for c in range(1,11)}: raise ValueError('Sinif TP/FP kapsami farkli')
            for values in pc.values():
                if set(values)!=set(d) or any(not isinstance(x,int) or x<0 for x in values.values()):raise ValueError('Gecersiz sinif TP/FP sayisi')
            for k in d:
                if sum(pc[str(c)][k] for c in range(1,11))!=d[k]: raise ValueError('Sinif TP/FP toplami farkli')
            for c in range(1,11):
                if pc[str(c)]['TP']!=table[c,'all'][key]: raise ValueError('Sinif TP/GT toplami farkli')
    return data


def save_cache(path, data, bindings):
    validate_analysis(data)
    body={'bindings':bindings,'analysis':data};body['content_sha256']=fingerprint(body)
    write_json(path,body)


def load_cache(path, bindings):
    p=read_json(path)
    if p['content_sha256']!=fingerprint({k:v for k,v in p.items() if k!='content_sha256'}) or p['bindings']!=bindings:
        raise ValueError('16 goruntu kaydi/kaynagi farkli')
    return validate_analysis(p['analysis'])


def combine(parts, reference_metrics):
    result={'images':len(parts),'normal_GT':sum(p['normal_GT'] for p in parts),'thresholds':{}}
    for t in [str(v) for v in THRESHOLDS]:
        sizes=Counter(); densities={g:[] for g in GROUPS}; detections={v:Counter() for v in ['baseline','hybrid_a025']}
        for part in parts:
            item=part['thresholds'][t];densities[part['object_count_group']].append(item)
            for row in item['class_size_rows']:
                for k in COUNT_KEYS:sizes[row['class_id'],row['area_group'],k]+=row[k]
            for v in detections:detections[v].update(item['detections'][v])
        class_rows=[{'class_id':c,'area_group':s,**finish({k:sizes[c,s,k] for k in COUNT_KEYS})} for c in range(11) for s in ['all']+AREAS]
        grouped=[]
        for group,items in densities.items():
            sums=Counter();dt={v:Counter() for v in detections}
            for item in items:
                sums.update({k:item['overall'][k] for k in COUNT_KEYS})
                for v in dt:dt[v].update(item['detections'][v])
            grouped.append({'object_count_group':group,'images':len(items),**finish(sums),
                            'detection_counts':{v:{k:int(dt[v][k]) for k in ['TP','FP','ignored_detections']} for v in dt}})
        for v,d in detections.items():
            if dict(d)!=reference_metrics['variants'][v]['counts_at_fixed_max500'][t]:
                raise ValueError('16 TP/FP sonucu tamamlanan 12 ile farkli: '+v+' IoU='+t)
        result['thresholds'][t]={'size_rows_all_classes':[x for x in class_rows if x['class_id']==0],
                     'per_class_size_rows':[x for x in class_rows if x['class_id']!=0],
                     'object_count_groups':grouped,'detection_counts':{v:dict(d) for v,d in detections.items()}}
    result.update({'original12_TP_FP_counts_equal':True,'new_AP_computed':False,
       'scope':'descriptive post hoc recall on frozen official VAL outputs; normal target GT only; fixed global max500 after ignore filtering',
       'area_definition':'original GT width*height in source pixels; positive-area disjoint bins <256, [256,1024), [1024,9216), >=9216; nonpositive dimensions separate',
       'matching':'full image, class-aware greedy original matcher first; GT assigned to size groups after matching; no size-specific rematching',
       'FP_note':'FP reported overall/by class/by image object-count group; no FP assigned to a GT-size bin',
       'not_official_area_AP':True,'parameter_search':False,'training_started':False,'inference_started':False,
       'uncertainty_not_measured':True,'no_independent_confirmation_claim':True})
    return result


def process(root, base, cfg12, output, config):
    records=cfg12['records'];index=read_json(base/'prediction_index.json')['records']
    if len(index)!=len(records) or {x['source_image'] for x in index}!={x['image'] for x in records}:raise ValueError('VAL kapsami farkli')
    mapping={x['source_image']:x for x in index};parts=[];bindings_index=[];immutable={}
    for i,record in enumerate(records,1):
        item=mapping[record['image']];path=safe_path(base,item['path'])
        anno=root/'data/prepared/visdrone_annotations/val'/(Path(record['image']).stem+'.txt')
        if sha256(path)!=item['sha256'] or sha256(anno)!=record['annotation_sha256'] or item['annotation_sha256']!=record['annotation_sha256']:
            raise ValueError('VAL kaynak tahmin/etiket degismis: '+record['image'])
        bindings={'protocol_sha256':fingerprint(config),'record':record,'prediction_sha256':item['sha256'],'annotation_sha256':record['annotation_sha256']}
        cache=output/'images'/(Path(record['image']).stem+'.json')
        if cache.exists():data=load_cache(cache,bindings)
        else:
            p=o.r.val.checked_json(path,o.r.P12,record)
            data=analyze(raw_annotation(anno.read_text(encoding='utf-8-sig')),p);save_cache(cache,data,bindings)
        parts.append(data);immutable[str(path)]=item['sha256'];immutable[str(anno)]=record['annotation_sha256']
        bindings_index.append({'source_image':record['image'],'path':cache.relative_to(output).as_posix(),'sha256':sha256(cache)})
        if i==1 or i%25==0 or i==len(records):
            print(f'16 CPU BOYUT ANALIZI: {i}/{len(records)}',flush=True)
            write_json(output/'progress.json',{'status':'ANALYZING','images_done':i,'images_total':len(records)})
    for path,digest in immutable.items():
        if sha256(path)!=digest:raise ValueError('Kaynak analiz sirasinda degismis')
    report=combine(parts,read_json(base/'metrics.json'))
    write_json(output/'cache_index.json',{'protocol_sha256':fingerprint(config),'records':bindings_index})
    write_json(output/'summary.json',report)
    fields=['IoU','class_id','area_group']+COUNT_KEYS+['baseline_recall_percent','hybrid_recall_percent','delta_recall_pp']
    with (output/'size_recall.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for t,item in report['thresholds'].items():
            for row in item['size_rows_all_classes']+item['per_class_size_rows']:w.writerow({'IoU':t,**row})
    with (output/'object_count_groups.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['IoU','object_count_group','images']+COUNT_KEYS+['baseline_recall_percent','hybrid_recall_percent','delta_recall_pp','baseline_FP','hybrid_FP']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for t,item in report['thresholds'].items():
            for row in item['object_count_groups']:
                w.writerow({'IoU':t,**{k:v for k,v in row.items() if k!='detection_counts'},
                     'baseline_FP':row['detection_counts']['baseline']['FP'],'hybrid_FP':row['detection_counts']['hybrid_a025']['FP']})
    return report


def small_backup(root, output, report=False):
    import os,shutil,tempfile
    target=root.parent/'HyCADO_Yedekleri/16_Boyut'
    files=[root/'hycado_start/size_diagnostics.py']
    files += [root/name for name in ['16_NESNE_BOYUTU_ANALIZ.cmd','16_RAPOR_AL.cmd','16_ONCE_OKUYUN.md','SHA256SUMS_16.txt'] if (root/name).exists()]
    if report:files += [p for p in output.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.zip']]
    for p in files:
        digest=sha256(p);dest=target/digest[:12]/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists() or sha256(dest)!=digest:
            fd,name=tempfile.mkstemp(dir=dest.parent,suffix='.tmp');os.close(fd)
            try:
                shutil.copy2(p,name)
                if sha256(name)!=digest:raise ValueError('16 kucuk yedek kopyasi farkli')
                os.replace(name,dest)
            finally:Path(name).unlink(missing_ok=True)
        if sha256(dest)!=digest:raise ValueError('16 kucuk yedek dogrulanamadi')
    return str(target)


def bundle(output):
    import zipfile,os,tempfile
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=output,prefix='16_',suffix='.zip.tmp');os.close(fd)
    target=output/'16_boyut_rapor.zip'
    try:
        with zipfile.ZipFile(name,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.glob('*.json'))+sorted(output.glob('*.csv'))+sorted((output/'images').glob('*.json')):z.write(p,p.relative_to(output).as_posix())
            if (output/'console.log').exists():
                with (output/'console.log').open('rb') as f:
                    f.seek(max(0,(output/'console.log').stat().st_size-60000));z.writestr('console_tail.log',f.read())
        os.replace(name,target)
    finally:Path(name).unlink(missing_ok=True)
    return target


def run():
    from .__main__ import _Tee
    output=ROOT/'experiments'/OUT;output.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        for lock in [ROOT/'.backup.lock']+[ROOT/'experiments'/folder/name for folder,name in
            [(OUT,'.analysis.lock'),(o.OUT,'.benchmark.lock'),(o.r.OUT,'.benchmark.lock'),(o.n.OUT,'.analysis.lock'),
             (o.r.val.OUT,'.evaluation.lock'),('stage1_v1_seed42','.training.lock'),('rtdetr_crop_v1_seed42','.training.lock')]]:
            stack.enter_context(training_lock(lock))
        with (output/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    print('16 SABIT VAL NESNE BOYUTU TANISI:',utc_now(),flush=True)
                    cfg12,base,upstream=source(ROOT)
                    config={'id':'hycado_size_diagnostics_v1','date':'2026-10-30','source12_protocol_sha256':o.r.P12,
                        'source12_metadata_sha256':o.r.FILES12,'upstream_metadata_sha256':upstream,'images':548,
                        'thresholds':list(THRESHOLDS),'global_max_detections':500,'area_groups':AREAS,
                        'area_boundaries_squared_pixels':[256,1024,9216],'object_count_groups':GROUPS,
                        'object_count_boundaries':[0,30,100],'matching_before_grouping':True,
                        'GT_denominator':'normal target class GT after unchanged ignore-region filtering; excludes score0/class0/class11',
                        'scope':'post hoc descriptive diagnostics of fixed predictions; no selection or new independent confirmation',
                        'new_AP_computed':False,'training_started':False,'inference_started':False,'parameter_search':False,'code_sha256':sha256(__file__)}
                    if (output/'protocol.json').exists() and read_json(output/'protocol.json')!=config:raise ValueError('16 protokolu degismis')
                    write_json(output/'protocol.json',config);write_json(output/'result.json',{'status':'RUNNING'})
                    for name in ['summary.json','size_recall.csv','object_count_groups.csv']:(output/name).unlink(missing_ok=True)
                    backup_path=small_backup(ROOT,output)
                    report=process(ROOT,base,cfg12,output,config)
                    for p,digest in upstream.items():
                        if sha256(ROOT/p)!=digest:raise ValueError('Onceki raporlar degismis')
                    for name,digest in o.r.FILES12.items():
                        if sha256(base/name)!=digest:raise ValueError('12 kaydi degismis')
                    if report['images']!=548:raise ValueError('548 VAL goruntusu gerekli')
                    result={'status':'COMPLETED','images':548,'thresholds':list(THRESHOLDS),'original12_TP_FP_counts_equal':True,
                        'source_predictions_unchanged':True,'training_started':False,'inference_started':False,'parameter_search':False,
                        'small_backup_directory':backup_path}
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();result={'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR','error':str(exc),
                        'resume':'rerun 16; verified complete per-image caches reused; partial image analyzed again'}
                write_json(output/'result.json',result);write_json(output/'progress.json',result);print('SONUC:',result['status'],flush=True)
        report_path=bundle(output)
        try:small_backup(ROOT,output,True)
        except Exception as exc:print('16 RAPOR KUCUK YEDEGI ALINAMADI:',str(exc),flush=True)
        print('GONDERILECEK RAPOR:',report_path,flush=True)
        return result['status']=='COMPLETED'


def main():
    if '--report-only' in sys.argv:print(bundle(ROOT/'experiments'/OUT),flush=True);return 0
    return 0 if run() else 1


if __name__=='__main__':raise SystemExit(main())
