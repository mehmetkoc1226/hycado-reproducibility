"""CPU-only development study from frozen 06 predictions; never loads a model."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import html
import importlib.metadata
import json
import math
import statistics
import sys
import time
import traceback
import zipfile
from pathlib import Path, PurePosixPath

from .common import ROOT, sha256, system_info, utc_now, write_json
from .data import NAMES
from .stage1 import fingerprint, training_lock
from .proposals import load_json, validate_manifest, validate_cache, prediction_path
from .proposal_geometry import select_regions, load_targets, coverage_counts, union_area
from .merge_geometry import POLICIES, initial_clusters, merge_to_budgets, projected_views, view_counts


def definitions(configuration):
    variants = []
    for scale in configuration['context_scales']:
        prefix = f'c{scale:g}'
        variants.append({'id':prefix+'_base','context_scale':scale,'policy':'baseline','budget':None})
        for policy in configuration['policies']:
            for budget in configuration['budgets']:
                variants.append({'id':f"{prefix}_{policy['name']}_b{budget}", 'context_scale':scale,
                                 'policy':policy['name'],'budget':budget})
    return variants


def validate_sources(stage1, source):
    """Verify metadata, code, and index; prediction/label bytes checked per image."""
    p1,p6 = load_json(stage1/'protocol.json'),load_json(source/'protocol.json')
    r1,r6 = load_json(stage1/'result.json'),load_json(source/'result.json')
    checkpoint = load_json(stage1/'checkpoint_index.json')
    manifest = load_json(stage1/'data/manifest.json')
    if (p1.get('protocol_id')!='hycado_stage1_development_v1' or p1.get('epochs')!=80 or p1.get('seed')!=42 or
            r1.get('status')!='COMPLETED' or r1.get('completed_epochs')!=80 or r6.get('status')!='COMPLETED' or
            r6['predictions']['images']!=6471 or r6['coverage']['development_images']!=647 or
            r6.get('official_val_used') is not False or p6.get('weights_used')!='ema'):
        raise ValueError('Tamamlanmis Stage-1 ve 06 raporlari gerekli')
    if (p6.get('protocol_id')!='hycado_stage1_predictions_coverage_v1' or
            p6['source_stage1_protocol_sha256']!=fingerprint(p1) or
            checkpoint['protocol_sha256']!=fingerprint(p1) or checkpoint['latest']!=p6['checkpoint'] or
            p6['checkpoint']['epoch']!=80 or p6.get('official_val_used') is not False):
        raise ValueError('Kaynak protokol/checkpoint kimligi eslesmiyor')
    manifest_hash = sha256(stage1/'data/manifest.json')
    if manifest_hash!=p6['dataset_manifest_sha256'] or manifest_hash!=p1['dataset_manifest_sha256']:
        raise ValueError('Kaynak veri manifesti degismis')
    validate_manifest(manifest)
    for name,digest in {**p1['code_hashes'],**p6['code_hashes']}.items():
        if Path(name).name!=name or sha256(Path(__file__).with_name(name))!=digest:
            raise ValueError('Kaynak kod kimligi degismis: '+name)
    for name in ['numpy','Pillow']:
        if importlib.metadata.version(name)!=p6['runtime_versions'][name]:
            raise ValueError('Kullanilan CPU paket surumu degismis: '+name)
    index = load_json(source/'prediction_index.json')
    if index['protocol_sha256']!=fingerprint(p6):
        raise ValueError('Tahmin indeks protokolu uyusmuyor')
    mapping = {r['source_image']:r for r in index['records']}
    if len(mapping)!=len(manifest['records']) or len(mapping)!=len(index['records']):
        raise ValueError('Tahmin indeksinde eksik/tekrarli kaynak')
    for record in manifest['records']:
        item = mapping.get(record['source_image'])
        expected = prediction_path(source,record).relative_to(source).as_posix()
        if item is None or item['path']!=expected or item['split']!=record['split']:
            raise ValueError('Tahmin indeksi/manifest kaynak rolleri uyusmuyor')
    coverage = load_json(source/'coverage_summary.json')
    if coverage['protocol_sha256']!=fingerprint(p6) or coverage.get('official_val_used') is not False:
        raise ValueError('06 kapsam raporu kimligi uyusmuyor')
    return {'manifest':manifest,'index':mapping,'p6':p6,'coverage':coverage,
            'prediction_index_sha256':sha256(source/'prediction_index.json'),
            'coverage_summary_sha256':sha256(source/'coverage_summary.json')}


def configuration_for(sources):
    return {'protocol_id':'hycado_crop_merge_development_v1',
            'source06_protocol_sha256':fingerprint(sources['p6']),
            'source_prediction_index_sha256':sources['prediction_index_sha256'],
            'source06_coverage_sha256':sources['coverage_summary_sha256'],
            'dataset_manifest_sha256':sources['p6']['dataset_manifest_sha256'],
            'source_checkpoint':sources['p6']['checkpoint'],
            'threshold':.05,'top_k':60,'context_scales':[2.,3.],
            'selection_reason':'development coverage/cost study after 06; no official VAL access',
            'budgets':[5,10,20],'policies':copy.deepcopy(POLICIES),
            'algorithm':'deterministic greedy heap; rank worst original-member axis growth, pair hull waste, gap, stable ids',
            'budget_policy':'keep all input crop coverage; report unmet budget, never discard crops to force count',
            'projected_input_side':960,'projected_geometry':'square scale-fill; visible short side conditional on IoA90',
            'pixel_thresholds':[4,8,16], 'thresholds_are_recognition_claims':False,
            'preview_count':6,'preview_selection':'smallest SHA256(07-preview:source_image); independent of metrics',
            'official_val_used':False,'official_visdrone_metrics_computed':False,
            'model_loaded':False,'auto_select_configuration':False,
            'runtime_versions':{name:importlib.metadata.version(name) for name in ['numpy','Pillow']},
            'code_hashes':{name:sha256(Path(__file__).with_name(name)) for name in ['crop_merge.py','merge_geometry.py']}}


def measurement_path(output,record):
    return Path(output)/'measurements'/'val'/(PurePosixPath(record['image']).stem+'.json')


def input_for(record,data,source,sources):
    if record['split']!='val' or record['source_split']!='train':
        raise ValueError('07 yalniz resmi TRAIN ic gelistirme ayrimini okur')
    path = prediction_path(source,record)
    prediction_hash = sha256(path)
    if prediction_hash!=sources['index'][record['source_image']]['sha256']:
        raise ValueError('Kaynak tahmin dosyasi indeksten farkli: '+str(path))
    payload = validate_cache(load_json(path),record,fingerprint(sources['p6']))
    label = data/'labels/val'/(PurePosixPath(record['image']).stem+'.txt')
    label_hash = sha256(label)
    if label_hash!=record['label_sha256']:
        raise ValueError('Kaynak gelistirme etiketi degismis: '+str(label))
    targets = load_targets(label.read_text(encoding='utf-8'),payload['width'],payload['height'])
    if len(targets)!=record['gt_count']:
        raise ValueError('GT sayisi manifestle farkli')
    return payload,targets,prediction_hash,label_hash


def measure_one(payload,targets,configuration):
    width,height = payload['width'],payload['height']
    bases,views,merged = {},{},{}
    for scale in configuration['context_scales']:
        base = select_regions(payload['detections'],width,height,configuration['threshold'],configuration['top_k'],scale)
        bases[scale] = initial_clusters(base)
        views[scale] = projected_views(targets,base,configuration['projected_input_side'])
        for policy in configuration['policies']:
            merged[scale,policy['name']] = merge_to_budgets(base,policy,configuration['budgets'])
    rows = []
    for variant in definitions(configuration):
        scale = variant['context_scale']
        clusters = bases[scale] if variant['policy']=='baseline' else merged[scale,variant['policy']][variant['budget']]
        regions = [c['rect'] for c in clusters]
        coverage = coverage_counts(targets,regions)
        view = view_counts(targets,regions,*views[scale],configuration['projected_input_side'])
        if coverage['all']['ioa90']!=view['all']['ioa90']:
            raise RuntimeError('Uzamsal kapsam/projeksiyon esigi farkli')
        rows.append({**variant,'crop_count':len(regions),'base_crop_count':len(bases[scale]),
                     'budget_met':None if variant['budget'] is None else len(regions)<=variant['budget'],
                     'clusters':clusters,'coverage':coverage,'view':view,
                     'union_fraction':union_area(regions)/(width*height),
                     'sum_fraction':sum((r[2]-r[0])*(r[3]-r[1]) for r in regions)/(width*height),
                     'max_member_axis_growth':max([max((c['rect'][2]-c['rect'][0])/c['min_width'],
                                                       (c['rect'][3]-c['rect'][1])/c['min_height']) for c in clusters],default=1.)})
    return rows


def validate_measurement(payload,record,digest,prediction_hash,label_hash):
    body = {k:v for k,v in payload.items() if k!='content_sha256'}
    if payload.get('content_sha256')!=fingerprint(body):
        raise ValueError('07 olcum dosyasinin icerik hash degeri degismis')
    if (payload['protocol_sha256']!=digest or payload['source_image']!=record['source_image'] or
            payload['source_prediction_sha256']!=prediction_hash or payload['source_label_sha256']!=label_hash):
        raise ValueError('07 olcum kimligi/kaynagi farkli')
    return payload


def process_measurements(records,data,source,output,sources,configuration,digest):
    selected = [r for r in records if r['split']=='val']
    files,new,reused = [],0,0
    for number,record in enumerate(selected,1):
        payload,targets,prediction_hash,label_hash = input_for(record,data,source,sources)
        path = measurement_path(output,record)
        if path.exists():
            validate_measurement(load_json(path),record,digest,prediction_hash,label_hash)
            reused+=1
        else:
            measured = {'protocol_sha256':digest,'source_image':record['source_image'],
                        'source_prediction_sha256':prediction_hash,'source_label_sha256':label_hash,
                        'image_sha256':record['image_sha256'],'image':record['image'],
                        'width':payload['width'],'height':payload['height'],'target_boxes':targets,
                        'variants':measure_one(payload,targets,configuration)}
            measured['content_sha256'] = fingerprint(measured)
            write_json(path,measured)
            new+=1
        files.append(path)
        if number==1 or number%25==0 or number==len(selected):
            write_json(output/'progress.json',{'status':'MEASURING','images':number,'total':len(selected),
                                               'new':new,'reused':reused})
            print(f'Crop birlestirme: {number}/{len(selected)}; yeni={new}; kayittan={reused}',flush=True)
    return files,{'images':len(selected),'new_this_invocation':new,'reused_this_invocation':reused}


def add_counts(total,counts):
    for key,value in counts.items():
        total[key] = total.get(key,0)+value


def summarize(files,configuration,digest,reference_coverage):
    import numpy as np
    accum = {v['id']:{'definition':v,'crop_counts':[],'base_counts':[],'union':[],'sum':[],
                     'budget_met':[],'max_growth':[], 'coverage':{'all':{},'small_native_area_lt_1024':{},
                     'classes':{str(i):{} for i in range(10)}},'view':{'all':{},'small':{}},
                     'ratios':[],'lost':0,'gained':0} for v in definitions(configuration)}
    for path in files:
        for row in load_json(path)['variants']:
            a = accum[row['id']]
            a['crop_counts'].append(row['crop_count']);a['base_counts'].append(row['base_crop_count'])
            a['union'].append(row['union_fraction']);a['sum'].append(row['sum_fraction'])
            a['budget_met'].append(row['budget_met']);a['max_growth'].append(row['max_member_axis_growth'])
            for key in ['all','small_native_area_lt_1024']:
                add_counts(a['coverage'][key],row['coverage'][key])
            for key in a['coverage']['classes']:
                add_counts(a['coverage']['classes'][key],row['coverage']['classes'][key])
            for key in ['all','small']:
                add_counts(a['view'][key],row['view'][key])
            a['ratios'].extend(row['view']['small_view_ratios_vs_baseline'])
            a['lost']+=row['view']['small_lost_min8px'];a['gained']+=row['view']['small_gained_min8px']
    rows = []
    ratio = lambda numerator,denominator: numerator/denominator if denominator else None
    for a in accum.values():
        n,small = a['coverage']['all']['targets'],a['view']['small']['targets']
        count = len(a['crop_counts'])
        if not count:
            raise ValueError('Ozet icin olcum yok')
        row = {**a['definition'],'images':count,'targets':n,'small_targets':small,
               'mean_crops':statistics.mean(a['crop_counts']),
               'p95_crops':sorted(a['crop_counts'])[math.ceil(.95*count)-1],
               'crop_call_reduction_vs_base':1-ratio(sum(a['crop_counts']),sum(a['base_counts'])) if sum(a['base_counts']) else 0.,
               'budget_met_rate':None if a['definition']['budget'] is None else sum(a['budget_met'])/count,
               'budget_unmet_images':sum(v is False for v in a['budget_met']),
               'coverage_ioa90':ratio(a['coverage']['all']['ioa90'],n),
               'small_coverage_ioa90':ratio(a['coverage']['small_native_area_lt_1024']['ioa90'],small),
               'small_coverage90_min8px':ratio(a['view']['small']['visible_min8px'],small),
               'small_coverage90_min16px':ratio(a['view']['small']['visible_min16px'],small),
               'small_paired_view_ratio_p10':float(np.quantile(a['ratios'],.1)) if a['ratios'] else None,
               'small_paired_view_ratio_median':float(np.median(a['ratios'])) if a['ratios'] else None,
               'small_lost_min8px':a['lost'],'small_gained_min8px':a['gained'],
               'mean_union_fraction':statistics.mean(a['union']),'mean_sum_fraction':statistics.mean(a['sum']),
               'max_member_axis_growth':max(a['max_growth']),
               'square_input_pixels_per_image_proxy':statistics.mean(a['crop_counts'])*configuration['projected_input_side']**2,
               'counts':{'coverage':a['coverage'],'view':a['view']}}
        rows.append(row)
    # Both unmerged controls must reproduce 06 before any result is accepted.
    for row in rows:
        if row['policy']=='baseline':
            original = next(r for r in reference_coverage['configurations'] if
                            (r['threshold'],r['top_k'],r['context_scale'])==(.05,60,row['context_scale']))
            if (row['counts']['coverage']!=original['counts'] or abs(row['mean_crops']-original['mean_crops'])>1e-12 or
                    abs(row['mean_union_fraction']-original['mean_union_fraction'])>1e-12):
                raise ValueError('07 birlestirilmemis kontrol 06 sonucunu yeniden uretmedi')
    return {'protocol_sha256':digest,'configurations':rows,'unmerged_controls_match_06':True,
            'official_val_used':False,'official_visdrone_metrics_computed':False,
            'auto_selected_configuration':None,'detector_runtime_measured':False}


def write_outputs(output,summary):
    write_json(output/'merge_summary.json',summary)
    rows = [{k:v for k,v in r.items() if k!='counts'} for r in summary['configurations']]
    with (output/'merge_summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    pct = lambda v:'—' if v is None else f'{100*v:.2f}'
    cells=[]
    for r in rows:
        values=[r['id'],f"{r['mean_crops']:.2f}",r['p95_crops'],pct(r['budget_met_rate']),pct(r['crop_call_reduction_vs_base']),
                pct(r['coverage_ioa90']),pct(r['small_coverage_ioa90']),pct(r['small_coverage90_min8px']),
                pct(r['small_coverage90_min16px']), '—' if r['small_paired_view_ratio_median'] is None else f"{r['small_paired_view_ratio_median']:.3f}",
                r['small_lost_min8px'],r['small_gained_min8px']]
        cells.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    page='''<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>07 Crop birleştirme raporu</title><style>body{max-width:1450px;margin:30px auto;padding:0 20px;font:15px/1.6 system-ui;color:#20343f}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf4f8}img{max-width:100%;height:auto}figure{margin:24px 0}code{overflow-wrap:anywhere}</style><h1>07 Crop birleştirme ve büyütme kontrolü</h1><p>Resmi TRAIN içinden ayrılan geliştirme görüntüleri. Model çıkarımı ve eğitim yapılmadı. İki birleştirilmemiş kontrol 06 sonuçlarını yeniden üretti. Bunlar resmi AP, tespit recall’u veya ölçülmüş RT-DETR hızı değildir.</p><p>Tüm ayarlarda eşik 0.05 ve Top-K 60. c2/c3 başlangıç crop boyut çarpanı 2/3; b5/b10/b20 istenen crop sayısı. Hedefe ulaşmak için crop silinmez; hedefin tutma oranını ayrıca okuyun. IoA90 tek crop içinde GT alanının en az %90’ı. Küçük hedef özgün alanda &lt;1024 piksel².</p><p>8/16 px sütunları, kapsanan nesnenin görünür kısa kenarının 960×960 square scale-fill sonrası büyüklüğünü de sınar. Bu eşikler tanı amaçlıdır, tanınabilirlik garantisi değildir. Görünüm oranı, başlangıçta IoA90 ile kapsanan küçük nesneler için birleştirilmiş/en iyi başlangıç kısa kenar boyutudur; nesne ağırlıklı medyan verilir. Kaybedilen/kazanılan 8 px sayıları aynı GT nesneleri üzerinde eşleştirilmiştir.</p><div style="overflow:auto"><table><tr><th>Ayar</th><th>Ort. crop</th><th>P95</th><th>Hedef tutma %</th><th>Çağrı azalma %</th><th>IoA90 %</th><th>Küçük IoA90 %</th><th>Küçük ≥8px %</th><th>Küçük ≥16px %</th><th>Görünüm oranı medyan</th><th>8px kayıp</th><th>8px kazanım</th></tr>'''
    page+=''.join(cells)+'</table></div><p>Crop çağrısı azalması yalnızca bölge sayısı üzerinden hesaplanır; gerçek model gecikmesi değildir. Greedy yöntem global optimum garantisi vermez. Otomatik ayar seçilmedi; sonucu dondurup inceleyin.</p><h2>Önceden sabitlenen örnekler</h2><p>Sarı: crop sınırları. Camgöbeği: GT kutuları. GT yalnızca ölçüm/görselleştirmede kullanılır; birleşim kararını belirlemez. c3 başlangıcı, near_z2/b10 ve near_z3/b10 yan yana gösterilir.</p>'
    for info in summary.get('previews',[]):
        page+='<figure><img src="'+html.escape(info['path'])+'" alt="Crop geometri karşılaştırması"><figcaption>'+html.escape(info['source_image'])+'</figcaption></figure>'
    page+='<p>Protokol: <code>'+summary['protocol_sha256']+'</code></p></html>'
    (output/'merge_summary.html').write_text(page,encoding='utf-8')


def make_previews(files,data,output,count):
    from PIL import Image,ImageDraw,ImageFont
    selected=sorted(files,key=lambda p:hashlib.sha256(('07-preview:'+load_json(p)['source_image']).encode()).hexdigest())[:count]
    previews=[]
    font=ImageFont.load_default(size=16)
    for number,path in enumerate(selected,1):
        measured=load_json(path)
        source=data/measured['image']
        if sha256(source)!=measured['image_sha256']:
            raise ValueError('Onizleme goruntusunun hash degeri degismis: '+str(source))
        with Image.open(source) as im:
            im=im.convert('RGB')
            if im.size!=(measured['width'],measured['height']):
                raise ValueError('Onizleme boyutu cache ile farkli')
            scale=480/im.width; h=max(1,round(im.height*scale)); header=42
            canvas=Image.new('RGB',(1440,h+header+26),'#f2f5f7')
            for column,key in enumerate(['c3_base','c3_near_z2_b10','c3_near_z3_b10']):
                row=next(r for r in measured['variants'] if r['id']==key)
                panel=im.resize((480,h))
                draw=ImageDraw.Draw(panel)
                for box in measured['target_boxes']:
                    draw.rectangle(tuple(v*scale for v in box[:4]),outline='#24c9db',width=1)
                for c in row['clusters']:
                    draw.rectangle(tuple(v*scale for v in c['rect']),outline='#ffe343',width=2)
                canvas.paste(panel,(column*480,header))
                ImageDraw.Draw(canvas).text((column*480+8,8),f"{key}: {row['crop_count']} crops",font=font,fill='#20343f')
            ImageDraw.Draw(canvas).text((8,h+header+3),'Yellow: crops | Cyan: GT | fixed sample, not selected by performance',font=font,fill='#20343f')
            preview=output/'previews'/f'sample_{number:02d}.jpg';preview.parent.mkdir(parents=True,exist_ok=True)
            canvas.save(preview,quality=85)
        previews.append({'source_image':measured['source_image'],'path':preview.relative_to(output).as_posix(),
                         'source_image_sha256_verified':True})
    return previews


def make_bundle(output):
    allowed=['protocol.json','environment.json','result.json','progress.json','validation_record.json']
    complete=(output/'result.json').exists() and load_json(output/'result.json').get('status')=='COMPLETED'
    if complete:
        allowed+=['merge_summary.json','merge_summary.csv','merge_summary.html']
    bundle=output/'07_crop_birlestirme_rapor.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in allowed:
            if (output/name).exists():z.write(output/name,name)
        if complete:
            for path in sorted((output/'previews').glob('sample_*.jpg')):
                z.write(path,path.relative_to(output).as_posix())
        if (output/'console.log').exists():
            with (output/'console.log').open('rb') as stream:
                stream.seek(max(0,(output/'console.log').stat().st_size-150000));z.writestr('console_tail.log',stream.read())
    return bundle


def run(stage1=None,source=None,output=None):
    from .__main__ import _Tee
    stage1=Path(stage1 or ROOT/'experiments/stage1_v1_seed42').resolve()
    source=Path(source or ROOT/'experiments/stage1_predictions_v1_seed42').resolve()
    output=Path(output or ROOT/'experiments/crop_merge_v1_seed42').resolve()
    for old in [stage1,source]:
        if old==output or output.is_relative_to(old) or old.is_relative_to(output):
            raise ValueError('07 cikti klasoru kaynak klasorlerden ayri olmali')
    output.mkdir(parents=True,exist_ok=True)
    with training_lock(output/'.merge.lock'):
        with (output/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    with training_lock(source/'.prediction.lock'),training_lock(stage1/'.training.lock'):
                        print(f'\n07 CROP BIRLESTIRME: {utc_now()}\nCPU geometri olcumu; model yuklenmez.',flush=True)
                        started=time.perf_counter()
                        sources=validate_sources(stage1,source)
                        configuration=configuration_for(sources);digest=fingerprint(configuration)
                        protocol=output/'protocol.json'
                        if protocol.exists() and fingerprint(load_json(protocol))!=digest:
                            raise ValueError('07 protokolu/kodu/kaynagi degismis; eski olcumlerle karistirilmadi')
                        write_json(protocol,configuration);write_json(output/'environment.json',system_info(include_torch=False))
                        files,work=process_measurements(sources['manifest']['records'],stage1/'data',source,output,sources,configuration,digest)
                        summary=summarize(files,configuration,digest,sources['coverage'])
                        summary['previews']=make_previews(files,stage1/'data',output,configuration['preview_count'])
                        write_outputs(output,summary)
                        write_json(output/'validation_record.json',{'created_utc':utc_now(),'source_protocols_and_code_verified':True,
                                   'development_prediction_and_label_bytes_verified':len(files),
                                   'unmerged_controls_match_06':True,'checkpoint_bytes_rehashed':False,
                                   'checkpoint_identity_inherited_from_completed_06':configuration['source_checkpoint'],
                                   'preview_image_bytes_verified':len(summary['previews']),'official_val_used':False})
                        result={'status':'COMPLETED','work':work,'configurations':len(summary['configurations']),
                                'previews':len(summary['previews']),'unmerged_controls_match_06':True,
                                'elapsed_this_invocation_seconds':time.perf_counter()-started,
                                'model_loaded':False,'training_performed':False,'official_val_used':False,
                                'official_visdrone_metrics_computed':False,'auto_selected_configuration':None}
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();result={'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',
                                                 'error':str(exc),'error_type':type(exc).__name__,
                                                 'resume':'Rerun 07; verified completed image measurements are reused'}
                write_json(output/'result.json',result);write_json(output/'progress.json',result)
                print('SONUC:',result['status'],flush=True)
        print('GONDERILECEK RAPOR:',make_bundle(output),flush=True)
        return result['status']=='COMPLETED'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1-run',type=Path);parser.add_argument('--source06',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:return 0 if run(args.stage1_run,args.source06,args.output) else 1
    except Exception:traceback.print_exc();return 1


if __name__=='__main__':raise SystemExit(main())
