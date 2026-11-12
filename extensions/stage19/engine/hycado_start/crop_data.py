"""Materialize frozen-policy native crops and explicit positive/ignore annotations."""
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
import os
import shutil
import statistics
import sys
import tempfile
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path,PurePosixPath

from .common import ROOT,sha256,system_info,utc_now,write_json
from .data import NAMES
from .stage1 import fingerprint,training_lock
from .proposals import load_json,prediction_path,validate_cache
from .proposal_geometry import select_regions
from .merge_geometry import merge_to_budgets
from . import crop_merge
from .crop_labels import ANNOTATION_POLICY,canonical_source_targets,crop_annotations,validate_annotation


DEFAULT_DECISION=ROOT/'configs/HyCADO_Crop_Politikasi_v1.json'


def validate_decision(decision,p7,summary,sources):
    if (decision.get('decision_id')!='hycado_crop_policy_v1' or
            decision.get('selected_configuration_id')!='c3_near_z3_b10' or
            decision['source07_protocol_sha256']!=fingerprint(p7) or
            summary['protocol_sha256']!=fingerprint(p7) or summary.get('unmerged_controls_match_06') is not True or
            decision['source06_protocol_sha256']!=fingerprint(sources['p6']) or
            decision['source_prediction_index_sha256']!=sources['prediction_index_sha256'] or
            decision['dataset_manifest_sha256']!=sources['p6']['dataset_manifest_sha256'] or
            p7['source06_protocol_sha256']!=fingerprint(sources['p6']) or
            p7['source_prediction_index_sha256']!=sources['prediction_index_sha256'] or
            p7['source06_coverage_sha256']!=sources['coverage_summary_sha256']):
        raise ValueError('Gomulu crop karari / kaynak protokoller uyusmuyor')
    selected=next(r for r in summary['configurations'] if r['id']==decision['selected_configuration_id'])
    for key,value in decision['observed_development'].items():
        if selected[key]!=value:
            raise ValueError('Secim kaydi ile 07 gozlenen sonucu farkli: '+key)
    generation=decision['proposal_generation'];merge=decision['merge']
    policy=next(p for p in p7['policies'] if p['name']==selected['policy'])
    if (generation['threshold']!=p7['threshold'] or generation['top_k']!=p7['top_k'] or
            generation['context_scale']!=selected['context_scale'] or merge['policy']!=policy or
            merge['requested_crop_count']!=selected['budget'] or merge['requested_crop_count_is_hard_cap'] is not False or
            merge['source_code_hashes']!=p7['code_hashes'] or
            decision['stage1_reference']['checkpoint']!=sources['p6']['checkpoint']):
        raise ValueError('Secilen crop politikasinin parametreleri degismis')
    return selected


def validate_inputs(stage1,source06,source07,decision_file):
    sources=crop_merge.validate_sources(stage1,source06)
    p7=load_json(source07/'protocol.json');r7=load_json(source07/'result.json')
    summary=load_json(source07/'merge_summary.json');decision=load_json(decision_file)
    if r7.get('status')!='COMPLETED' or r7['work']['images']!=647 or r7.get('official_val_used') is not False:
        raise ValueError('Tamamlanmis 07 gelistirme olcumu gerekli')
    for name,digest in p7['code_hashes'].items():
        if Path(name).name!=name or sha256(Path(__file__).with_name(name))!=digest:
            raise ValueError('07 kaynak kod kimligi degismis: '+name)
    selected=validate_decision(decision,p7,summary,sources)
    for record in sources['manifest']['records']:
        if record['split']=='val' and not crop_merge.measurement_path(source07,record).is_file():
            raise FileNotFoundError('07 goruntu olcumu eksik; 07 klasorunu koruyun: '+record['image'])
    sources.update(decision=decision,p7=p7,selected=selected,
                   decision_sha256=sha256(decision_file),source07_summary_sha256=sha256(source07/'merge_summary.json'))
    return sources


def configuration_for(sources):
    return {'protocol_id':'hycado_crop_dataset_v1','decision':sources['decision'],
            'decision_sha256':sources['decision_sha256'],
            'source06_protocol_sha256':fingerprint(sources['p6']),
            'source_prediction_index_sha256':sources['prediction_index_sha256'],
            'source07_protocol_sha256':fingerprint(sources['p7']),
            'source07_summary_sha256':sources['source07_summary_sha256'],
            'dataset_manifest_sha256':sources['p6']['dataset_manifest_sha256'],
            'annotation_policy':copy.deepcopy(ANNOTATION_POLICY),
            'png_compress_level':1,'preview_count_per_split':3,
            'preview_selection':'first SHA256(08-preview:source_image) source ids, first lexical crop per source',
            'runtime_versions':{n:importlib.metadata.version(n) for n in ['numpy','Pillow']},
            'code_hashes':{n:sha256(Path(__file__).with_name(n)) for n in ['crop_data.py','crop_labels.py']},
            'official_val_used':False,'old_experiment_artifacts_used':False,'model_loaded':False,
            'training_started':False,'training_budget_frozen':False,
            'requires_ignore_aware_training_loader':True,'generic_dataset_yaml_exported':False}


def parent_dir(output,record):
    p=PurePosixPath(record['image'])
    if (record['split'] not in {'train','val'} or len(p.parts)!=3 or
            p.parts[:2]!=('images',record['split']) or '..' in p.parts or '\\' in str(p)):
        raise ValueError('Crop kaynak yolu/rolu gecersiz')
    return Path(output)/'items'/record['split']/p.stem


def select_clusters(payload,decision):
    g=decision['proposal_generation'];m=decision['merge']
    initial=select_regions(payload['detections'],payload['width'],payload['height'],g['threshold'],g['top_k'],g['context_scale'])
    return initial,merge_to_budgets(initial,m['policy'],[m['requested_crop_count']])[m['requested_crop_count']]


def read_source(record,data,source06,source07,sources):
    from PIL import Image
    if record['source_split']!='train':raise ValueError('Resmi VAL crop uretimine alinamaz')
    pred_path=prediction_path(source06,record);pred_hash=sha256(pred_path)
    if pred_hash!=sources['index'][record['source_image']]['sha256']:
        raise ValueError('Kaynak tahmin hash degeri degismis: '+str(pred_path))
    payload=validate_cache(load_json(pred_path),record,fingerprint(sources['p6']))
    stem=PurePosixPath(record['image']).stem;split=record['split']
    label=data/'labels'/split/(stem+'.txt');raw=data/'annotations'/split/(stem+'.txt');image=data/record['image']
    hashes={'prediction_sha256':pred_hash,'label_sha256':sha256(label),
            'annotation_sha256':sha256(raw),'image_sha256':sha256(image)}
    for key in ['label_sha256','annotation_sha256','image_sha256']:
        if hashes[key]!=record[key]:raise ValueError('Kaynak dosya degismis: '+record['image']+' / '+key)
    initial,clusters=select_clusters(payload,sources['decision'])
    if split=='val':
        measured=crop_merge.validate_measurement(load_json(crop_merge.measurement_path(source07,record)),
                                                record,fingerprint(sources['p7']),pred_hash,hashes['label_sha256'])
        expected=next(v for v in measured['variants'] if v['id']==sources['decision']['selected_configuration_id'])['clusters']
        if fingerprint(clusters)!=fingerprint(expected):
            raise ValueError('Yeniden uretilen crop geometrisi 07 ile farkli')
    annotations=canonical_source_targets(raw.read_text(encoding='utf-8-sig'),payload['width'],payload['height'],
                                         label.read_text(encoding='utf-8'),source=str(raw))
    if len(annotations['targets'])!=record['gt_count']:
        raise ValueError('Ham kaynak GT sayisi manifestle farkli')
    # No full-frame pixels are retained between parents. Native dimensions must agree.
    with Image.open(image) as im:
        if im.size!=(payload['width'],payload['height']):raise ValueError('Goruntu/06 boyutlari eslesmiyor')
    return payload,annotations,initial,clusters,hashes,image


def validate_parent(folder,record,digest,hashes):
    document=load_json(folder/'source.json')
    if document.get('content_sha256')!=fingerprint({k:v for k,v in document.items() if k!='content_sha256'}):
        raise ValueError('Crop kaynak kaydinin icerik hash degeri farkli')
    if (document['protocol_sha256']!=digest or document['source_image']!=record['source_image'] or
            document['split']!=record['split'] or document['source_hashes']!=hashes):
        raise ValueError('Kayitli crop kaynak kimligi degismis')
    expected={'source.json'}
    for item in document['crops']:
        for key,hash_key in [('image','image_sha256'),('annotation','annotation_sha256')]:
            name=item[key]
            if Path(name).name!=name or sha256(folder/name)!=item[hash_key]:
                raise ValueError('Kayitli crop dosyasi degismis/eksik: '+str(folder/name))
            expected.add(name)
        annotation=load_json(folder/item['annotation']);validate_annotation(annotation)
        if (annotation['schema']!='hycado_crop_annotation_v1' or annotation['protocol_sha256']!=digest or
                annotation['source_image']!=record['source_image'] or annotation['split']!=record['split']):
            raise ValueError('Crop anotasyon kimligi/rolu farkli')
    if {p.name for p in folder.iterdir()}!=expected:raise ValueError('Crop klasorune dosya eklenmis/silinmis')
    return document


def build_parent(output,record,payload,annotations,initial,clusters,hashes,image_path,configuration,digest):
    from PIL import Image
    destination=parent_dir(output,record)
    destination.parent.mkdir(parents=True,exist_ok=True)
    building=output/'building';building.mkdir(exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix=record['split']+'_'+destination.name+'_',dir=building))
    started=time.perf_counter();items=[];unique={i:set() for i in range(10)}
    try:
        with Image.open(image_path) as source_image:
            source_image=source_image.convert('RGB')
            for number,c in enumerate(clusters):
                rect=tuple(c['rect'])
                ann=crop_annotations(annotations['targets'],annotations['ignore_regions'],rect,
                                     configuration['annotation_policy']['minimum_visible_source_box_fraction'])
                image_name=f'crop_{number:04d}.png';ann_name=f'crop_{number:04d}.json'
                ann.update(protocol_sha256=digest,source_image=record['source_image'],split=record['split'],
                           group=record['group'],crop_index=number,original_crop_members=list(c['members']))
                source_image.crop(rect).save(temporary/image_name,format='PNG',compress_level=configuration['png_compress_level'])
                write_json(temporary/ann_name,ann)
                class_counts=Counter(t['class_id'] for t in ann['positive_targets'])
                for target in ann['positive_targets']:unique[target['class_id']].add(target['source_record_index'])
                items.append({'crop_index':number,'image':image_name,'annotation':ann_name,
                              'image_sha256':sha256(temporary/image_name),'annotation_sha256':sha256(temporary/ann_name),
                              'width':ann['width'],'height':ann['height'],'source_crop_xyxy':list(rect),
                              'positive_targets':len(ann['positive_targets']),'class_counts':dict(class_counts),
                              'ignored_boxes_by_reason':dict(Counter(r['reason'] for r in ann['ignore_regions'])),
                              'partial_positive_targets':sum(t['new_crop_truncation'] for t in ann['positive_targets']),
                              'ignore_union_fraction':ann['ignore_union_fraction'],'supervision_state':ann['supervision_state'],
                              'image_bytes':(temporary/image_name).stat().st_size,
                              'annotation_bytes':(temporary/ann_name).stat().st_size})
        document={'schema':'hycado_crop_source_v1','protocol_sha256':digest,'source_image':record['source_image'],
                  'source_image_path':record['image'],'split':record['split'],'group':record['group'],
                  'source_hashes':hashes,'width':payload['width'],'height':payload['height'],
                  'raw_source_target_count':len(annotations['targets']),
                  'raw_source_class_counts':dict(Counter(t['class_id'] for t in annotations['targets'])),
                  'unique_positive_source_targets_by_class':{str(i):len(v) for i,v in unique.items()},
                  'duplicate_target_rows_removed':annotations['duplicate_target_rows_removed'],
                  'invalid_geometry_rows_omitted':annotations['invalid_geometry_rows_omitted'],
                  'initial_crop_count':len(initial),'crop_count':len(items),'crops':items,
                  'build_seconds':time.perf_counter()-started}
        document['content_sha256']=fingerprint(document);write_json(temporary/'source.json',document)
        # Writable descriptor also works with Windows os.fsync; final directory is immutable.
        with (temporary/'source.json').open('r+b') as stream:os.fsync(stream.fileno())
        temporary.rename(destination)
        return document
    finally:
        if temporary.exists():shutil.rmtree(temporary,ignore_errors=True)


def process_sources(records,data,source06,source07,output,sources,configuration,digest):
    # Only uncommitted task-owned scratch directories are removed, under our lock.
    if (output/'building').exists():shutil.rmtree(output/'building')
    files=[];new=reused=0
    for number,record in enumerate(records,1):
        payload,annotations,initial,clusters,hashes,image=read_source(record,data,source06,source07,sources)
        destination=parent_dir(output,record)
        if destination.exists():
            validate_parent(destination,record,digest,hashes);reused+=1
        else:
            if shutil.disk_usage(output).free<512*2**20:raise OSError('D diskinde en az 512 MiB bos alan kalmali')
            build_parent(output,record,payload,annotations,initial,clusters,hashes,image,configuration,digest);new+=1
        files.append(destination/'source.json')
        if number==1 or number%50==0 or number==len(records):
            print(f"Crop verisi: {number}/{len(records)}; yeni={new}; kayittan={reused}; son={record['split']}",flush=True)
            write_json(output/'progress.json',{'status':'BUILDING_CROPS','source_images':number,
                                               'total_source_images':len(records),'new':new,'reused':reused})
    return files,{'source_images':len(records),'new_this_invocation':new,'reused_this_invocation':reused}


def finalize_dataset(files,output,sources,configuration,digest):
    aggregate={role:{'source_images':0,'raw_source_targets':0,'crop_counts':[],'positive_target_instances':0,
                    'partial_positive_target_instances':0,'class_instances':Counter(),'raw_source_classes':Counter(),
                    'unique_positive_source_targets_by_class':Counter(),'ignored_boxes_by_reason':Counter(),
                    'supervision_states':Counter(),'image_bytes':0,'annotation_bytes':0,
                    'duplicate_target_rows_removed':0,'invalid_geometry_rows_omitted':0} for role in ['train','val']}
    indexes=[];handles={}
    try:
        for role in aggregate:handles[role]=(output/f'{role}_crops.jsonl.tmp').open('w',encoding='utf-8')
        for path in files:
            parent=load_json(path);a=aggregate[parent['split']]
            a['source_images']+=1;a['raw_source_targets']+=parent['raw_source_target_count'];a['crop_counts'].append(parent['crop_count'])
            a['raw_source_classes'].update(parent['raw_source_class_counts'])
            a['unique_positive_source_targets_by_class'].update(parent['unique_positive_source_targets_by_class'])
            for key in ['duplicate_target_rows_removed','invalid_geometry_rows_omitted']:a[key]+=parent[key]
            indexes.append({'source_image':parent['source_image'],'split':parent['split'],'group':parent['group'],
                            'source_record':path.relative_to(output).as_posix(),'sha256':sha256(path),'crop_count':parent['crop_count']})
            for item in parent['crops']:
                a['positive_target_instances']+=item['positive_targets'];a['partial_positive_target_instances']+=item['partial_positive_targets']
                a['class_instances'].update(item['class_counts']);a['ignored_boxes_by_reason'].update(item['ignored_boxes_by_reason'])
                a['supervision_states'][item['supervision_state']]+=1
                for key in ['image_bytes','annotation_bytes']:a[key]+=item[key]
                relative=path.parent.relative_to(output)
                row={'schema':'hycado_crop_item_v1','protocol_sha256':digest,'source_image':parent['source_image'],
                     'split':parent['split'],'group':parent['group'],
                     **{k:v for k,v in item.items() if k not in ['image','annotation']},
                     'image':(relative/item['image']).as_posix(),'annotation':(relative/item['annotation']).as_posix(),
                     'requires_ignore_aware_loader':True}
                handles[parent['split']].write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    finally:
        for h in handles.values():h.close()
    for role in aggregate:os.replace(output/f'{role}_crops.jsonl.tmp',output/f'{role}_crops.jsonl')
    result={}
    for role,a in aggregate.items():
        nc=a.pop('crop_counts');n=sum(nc)
        result['training' if role=='train' else 'development']={**a,'crops':n,'zero_crop_source_images':nc.count(0),
            'mean_crops_per_source':statistics.mean(nc) if nc else None,
            'p95_crops_per_source':sorted(nc)[math.ceil(.95*len(nc))-1] if nc else None,
            'max_crops_per_source':max(nc,default=0),
            'unique_positive_source_targets':sum(a['unique_positive_source_targets_by_class'].values()),
            'image_and_annotation_gib':(a['image_bytes']+a['annotation_bytes'])/2**30}
    selected=sources['selected'];expected=round(selected['mean_crops']*selected['images'])
    if (result['development']['source_images']!=selected['images'] or result['development']['crops']!=expected):
        raise ValueError('Gelistirme crop sayisi secilen 07 sonucuyla eslesmedi')
    summary={'protocol_sha256':digest,'splits':result,'selected_configuration':configuration['decision']['selected_configuration_id'],
             'development_crop_geometry_matches_07':True,'official_val_used':False,'model_loaded':False,
             'training_ready_with_generic_ultralytics_loader':False,
             'requires_ignore_aware_training_loader':True,'training_budget_frozen':False,
             'counts_are_repeated_crop_instances_unless_explicitly_unique':True}
    write_json(output/'source_index.json',{'protocol_sha256':digest,'records':indexes})
    write_json(output/'dataset_summary.json',summary)
    contract={'schema':'hycado_crop_dataset_contract_v1','protocol_sha256':digest,'names':NAMES,
              'image_format':'native RGB PNG, lossless, no GT-driven pixel modification',
              'manifests':{'train':'train_crops.jsonl','development':'val_crops.jsonl'},
              'annotation_policy':configuration['annotation_policy'],
              'requires_ignore_aware_training_loader':True,
              'positive_matched_supervision_precedes_ignore':True,
              'ignore_boxes_must_not_be_silently_treated_as_background':True,
              'generic_YOLO_loader_supported':False,'model_training_started':False}
    write_json(output/'dataset_contract.json',contract)
    with (output/'class_counts.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['role','class_id','name','raw_unique_source_gt','crop_positive_instances','unique_source_gt_supervised'])
        for role,a in result.items():
            for cls,name in enumerate(NAMES):
                writer.writerow([role,cls,name,a['raw_source_classes'].get(str(cls),0),a['class_instances'].get(str(cls),0),
                                 a['unique_positive_source_targets_by_class'].get(str(cls),0)])
    return summary


def make_previews(files,output,count):
    from PIL import Image,ImageDraw,ImageFont
    selected=[]
    for role in ['train','val']:
        candidates=[p for p in files if p.parent.parent.name==role and load_json(p)['crop_count']>0]
        selected+=sorted(candidates,key=lambda p:hashlib.sha256(('08-preview:'+load_json(p)['source_image']).encode()).hexdigest())[:count]
    previews=[];font=ImageFont.load_default(size=16)
    for number,path in enumerate(selected,1):
        parent=load_json(path);item=parent['crops'][0];ann=load_json(path.parent/item['annotation'])
        with Image.open(path.parent/item['image']) as im:
            im=im.convert('RGB');scale=min(800/im.width,600/im.height,2.)
            panel=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))))
            draw=ImageDraw.Draw(panel)
            for row in ann['ignore_regions']:
                draw.rectangle(tuple(v*scale for v in row['box_xyxy']),outline='#ff8a24',width=2)
            for row in ann['positive_targets']:
                draw.rectangle(tuple(v*scale for v in row['box_xyxy']),outline='#24e9e9',width=2)
            canvas=Image.new('RGB',(max(720,panel.width),panel.height+70),'#eef3f7');canvas.paste(panel,(0,38))
            title=f"{parent['split']} | {Path(parent['source_image']).stem} | crop 0"
            ImageDraw.Draw(canvas).text((8,8),title,font=font,fill='#20343f')
            ImageDraw.Draw(canvas).text((8,panel.height+44),'Cyan: positive target | Orange: ignore / partial / others',font=font,fill='#20343f')
            target=output/'previews'/f'crop_{number:02d}.jpg';target.parent.mkdir(exist_ok=True);canvas.save(target,quality=88)
        previews.append({'source_image':parent['source_image'],'split':parent['split'],'path':target.relative_to(output).as_posix(),
                         'positive_targets':len(ann['positive_targets']),'ignore_regions':len(ann['ignore_regions'])})
    return previews


def write_html(output,summary):
    rows=[]
    for role,a in summary['splits'].items():
        rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [role,a['source_images'],a['crops'],
                     f"{a['mean_crops_per_source']:.2f}",a['positive_target_instances'],a['unique_positive_source_targets'],
                     a['supervision_states'].get('fully_ignored_no_positive',0),f"{a['image_and_annotation_gib']:.3f}"])+'</tr>')
    text='''<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>08 Crop veri denetimi</title><style>body{max-width:1100px;margin:30px auto;padding:0 20px;font:16px/1.6 system-ui;color:#20343f}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%;height:auto}code{overflow-wrap:anywhere}</style><h1>08 Crop veri denetimi</h1><p>Kaynak yalnızca resmi TRAIN’in sabit eğitim/geliştirme ayrımıdır. Resmi VAL kullanılmadı. Geometri yalnızca Stage-1 tahminlerinden gelir; görüntü pikselleri GT ile maskelenmedi/değiştirilmedi. Geliştirmedeki crop sayısı ve görüntü bazındaki koordinatlar seçilen 07 politikasıyla eşleşti.</p><div style="overflow:auto"><table><tr><th>Rol</th><th>Kaynak görüntü</th><th>Crop</th><th>Ort. crop</th><th>Tekrarlı pozitif örnek</th><th>Benzersiz kaynak GT</th><th>Tam ignore, pozitif yok</th><th>PNG+JSON GiB</th></tr>'''
    text+=''.join(rows)+'</table></div><p>Pozitif kutu: kaynak GT alanının en az yarısı crop içinde. Daha küçük kesişimler ve ham ignore/others kutuları ayrı ignore geometrisidir. Boş ve yalnız ignore içeren crop’lar korunur. Pozitif örnek toplamları örtüşen crop’larda tekrar sayılır; bağımsız nesne sayısı veya tespit recall’u değildir.</p><p><strong>Standart YOLO/Ultralytics loader bu veri sözleşmesini tek başına karşılamaz.</strong> Ignore geometrisini ve pozitif önceliğini işleyen eğitim kodu doğrulanmadan eğitim başlatılmamalıdır. Bu adım model yüklemez ve eğitim bütçesi seçmez. Gerçek crop sayısı ve sınıf dağılımı sonraki adımda örnekleme/bütçe kararına kaynak olur.</p>'
    for p in summary.get('previews',[]):text+='<figure><img src="'+html.escape(p['path'])+'" alt="Crop etiket denetimi"><figcaption>'+html.escape(p['source_image'])+'</figcaption></figure>'
    text+='<p>Protokol: <code>'+summary['protocol_sha256']+'</code></p></html>'
    (output/'dataset_summary.html').write_text(text,encoding='utf-8')


def make_bundle(output):
    bundle=output/'08_crop_veri_rapor.zip'
    allowed=['protocol.json','result.json','progress.json','environment.json']
    success=(output/'result.json').exists() and load_json(output/'result.json').get('status')=='COMPLETED'
    if success:allowed+=['dataset_summary.json','dataset_summary.html','class_counts.csv','dataset_contract.json']
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in allowed:
            if (output/name).exists():z.write(output/name,name)
        if success:
            for p in sorted((output/'previews').glob('crop_*.jpg')):z.write(p,p.relative_to(output).as_posix())
        if (output/'console.log').exists():
            with (output/'console.log').open('rb') as stream:
                stream.seek(max(0,(output/'console.log').stat().st_size-150000));z.writestr('console_tail.log',stream.read())
    return bundle


def run(stage1=None,source06=None,source07=None,output=None,decision_file=None):
    from .__main__ import _Tee
    stage1=Path(stage1 or ROOT/'experiments/stage1_v1_seed42').resolve()
    source06=Path(source06 or ROOT/'experiments/stage1_predictions_v1_seed42').resolve()
    source07=Path(source07 or ROOT/'experiments/crop_merge_v1_seed42').resolve()
    output=Path(output or ROOT/'experiments/crop_dataset_v1_seed42').resolve()
    decision_file=Path(decision_file or DEFAULT_DECISION).resolve()
    for source in [stage1,source06,source07]:
        if source==output or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('08 cikti klasoru mevcut kaynak klasorlerden ayri olmali')
    output.mkdir(parents=True,exist_ok=True)
    with training_lock(output/'.crop_data.lock'):
        with (output/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    with training_lock(source07/'.merge.lock'),training_lock(source06/'.prediction.lock'),training_lock(stage1/'.training.lock'):
                        print(f'\n08 GERCEK CROP VERISI: {utc_now()}\nPNG + hedef/ignore JSON; model yuklenmez.',flush=True)
                        started=time.perf_counter();sources=validate_inputs(stage1,source06,source07,decision_file)
                        configuration=configuration_for(sources);digest=fingerprint(configuration);protocol=output/'protocol.json'
                        if protocol.exists() and fingerprint(load_json(protocol))!=digest:
                            raise ValueError('08 veri protokolu/kodu/kaynagi degismis; mevcut crop verisiyle karistirilmadi')
                        write_json(protocol,configuration);write_json(output/'environment.json',system_info(include_torch=False))
                        files,work=process_sources(sources['manifest']['records'],stage1/'data',source06,source07,output,sources,configuration,digest)
                        summary=finalize_dataset(files,output,sources,configuration,digest)
                        summary['previews']=make_previews(files,output,configuration['preview_count_per_split'])
                        write_json(output/'dataset_summary.json',summary);write_html(output,summary)
                        result={'status':'COMPLETED','work':work,'crop_counts':{role:a['crops'] for role,a in summary['splits'].items()},
                                'development_geometry_matches_07':True,'requires_ignore_aware_training_loader':True,
                                'model_loaded':False,'training_started':False,'official_val_used':False,
                                'elapsed_this_invocation_seconds':time.perf_counter()-started}
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();result={'status':'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',
                                                  'error':str(exc),'error_type':type(exc).__name__,
                                                  'resume':'Rerun 08; committed source folders are verified and reused'}
                write_json(output/'result.json',result);write_json(output/'progress.json',result)
                print('SONUC:',result['status'],flush=True)
        print('GONDERILECEK RAPOR:',make_bundle(output),flush=True)
        return result['status']=='COMPLETED'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1-run',type=Path);parser.add_argument('--source06',type=Path)
    parser.add_argument('--source07',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--decision',type=Path)
    a=parser.parse_args()
    try:return 0 if run(a.stage1_run,a.source06,a.source07,a.output,a.decision) else 1
    except Exception:traceback.print_exc();return 1


if __name__=='__main__':raise SystemExit(main())
