"""Read immutable 08 source records, sample by source and load positive/ignore data."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path,PurePosixPath

from .common import sha256
from .stage1 import fingerprint
from .crop_labels import validate_annotation


SOURCE08_PROTOCOL='647bd53a65d7214bada37216070533596bbd6c388a3f8e354c69e30c75a51d7f'


def safe_path(root,name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Veri yolu gecersiz')
    path=(Path(root)/p).resolve()
    if not path.is_relative_to(Path(root).resolve()):raise ValueError('Veri yolu klasor disinda')
    return path


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load_catalog(root,expected=SOURCE08_PROTOCOL):
    root=Path(root);protocol=read_json(root/'protocol.json');result=read_json(root/'result.json')
    summary=read_json(root/'dataset_summary.json');contract=read_json(root/'dataset_contract.json')
    if (fingerprint(protocol)!=expected or summary['protocol_sha256']!=expected or
            contract['protocol_sha256']!=expected or result['status']!='COMPLETED' or
            result['official_val_used'] is not False or protocol['training_started'] is not False or
            protocol['code_hashes']!={n:sha256(Path(__file__).with_name(n)) for n in ['crop_data.py','crop_labels.py']}):
        raise ValueError('Tamamlanmis ve degismemis 08 verisi gerekli')
    index=read_json(root/'source_index.json')
    if index['protocol_sha256']!=expected:raise ValueError('08 kaynak indeksi protokolu farkli')
    groups={'train':[],'val':[]};seen=set();states={'train':Counter(),'val':Counter()};excluded=Counter()
    crops=Counter();positive=Counter();raw=Counter()
    for i,item in enumerate(index['records'],1):
        role=item['split'];identity=item['source_image']
        if role not in groups or identity in seen or not identity.startswith('images/train/'):
            raise ValueError('Kaynak tekrari veya resmi TRAIN disinda veri')
        seen.add(identity)
        path=safe_path(root,item['source_record'])
        if sha256(path)!=item['sha256']:raise ValueError('08 kaynak kaydi degismis: '+str(path))
        doc=read_json(path)
        if (doc['content_sha256']!=fingerprint({k:v for k,v in doc.items() if k!='content_sha256'}) or
                doc['protocol_sha256']!=expected or doc['source_image']!=identity or doc['split']!=role or
                doc['group']!=item['group'] or len(doc['crops'])!=doc['crop_count'] or doc['crop_count']!=item['crop_count']):
            raise ValueError('08 kaynak kaydi kimligi/sayaci farkli')
        candidates=[];raw[role]+=doc['raw_source_target_count'];crops[role]+=doc['crop_count']
        for crop in doc['crops']:
            states[role][crop['supervision_state']]+=1;positive[role]+=crop['positive_targets']
            if crop['supervision_state']=='fully_ignored_no_positive':excluded[role]+=1;continue
            entry={k:crop[k] for k in ['crop_index','image_sha256','annotation_sha256','width','height',
                                      'positive_targets','ignored_boxes_by_reason','supervision_state']}
            entry.update(source_image=identity,split=role,group=doc['group'],protocol_sha256=expected,
                         image=(path.parent/ crop['image']).relative_to(root).as_posix(),
                         annotation=(path.parent/ crop['annotation']).relative_to(root).as_posix())
            safe_path(root,entry['image']);safe_path(root,entry['annotation']);candidates.append(entry)
        groups[role].append({'source_image':identity,'crops':candidates})
        if i==1 or i%1000==0:print(f'08 indeks denetimi: {i}/{len(index["records"])}',flush=True)
    for role,label in [('train','training'),('val','development')]:
        a=summary['splits'][label]
        if (len(groups[role])!=a['source_images'] or crops[role]!=a['crops'] or
                positive[role]!=a['positive_target_instances'] or raw[role]!=a['raw_source_targets'] or
                dict(states[role])!=a['supervision_states']):raise ValueError('08 toplam sayaclari farkli')
    train_groups={item['group'] for item in index['records'] if item['split']=='train'}
    dev_groups={item['group'] for item in index['records'] if item['split']=='val'}
    if train_groups & dev_groups:raise ValueError('Train/dev grup sizintisi')
    info={'source08_protocol_sha256':expected,'source_index_sha256':sha256(root/'source_index.json'),
          'summary_sha256':sha256(root/'dataset_summary.json'),
          'total_crops':dict(crops),'fully_ignored_crops_excluded':dict(excluded),
          'source_images':{r:len(g) for r,g in groups.items()},
          'eligible_source_images':{r:sum(bool(x['crops']) for x in g) for r,g in groups.items()},
          'sample_read_integrity':'PNG/annotation hash, source/protocol identity and geometry verified on every read'}
    return groups,info


def source_round(groups,round_index,seed=42):
    """One eligible crop per parent; every eligible crop cycles before repetition."""
    ordered=sorted((g for g in groups if g['crops']),key=lambda g:g['source_image'])
    random.Random(seed+round_index).shuffle(ordered)
    result=[]
    for group in ordered:
        options=sorted(group['crops'],key=lambda c:c['crop_index'])
        token=f'{seed}:crop-cycle:{group["source_image"]}'.encode()
        random.Random(int.from_bytes(hashlib.sha256(token).digest()[:8],'big')).shuffle(options)
        result.append(options[round_index%len(options)])
    return result


def pilot_plan(groups,phases=3,representative=240,stress_count=16,development=32):
    if phases*representative>sum(bool(g['crops']) for g in groups['train']):
        raise ValueError('Pilot icin yeterli egitim kaynagi yok')
    base=source_round(groups['train'],0)
    all_crops=[c for g in groups['train'] for c in g['crops']]
    stress=[];seen=set()
    rankings=[lambda c:c['positive_targets'],lambda c:c['width']*c['height'],
              lambda c:sum(c['ignored_boxes_by_reason'].values())]
    # Fixed deterministic stress mix, independent of observed training loss.
    for rank,total in zip(rankings,[8,12,16]):
        for c in sorted(all_crops,key=lambda c:(-rank(c),c['source_image'],c['crop_index'])):
            key=(c['source_image'],c['crop_index'])
            if key not in seen:seen.add(key);stress.append(c)
            if len(stress)>=total:break
    stress=stress[:stress_count]
    if len(stress)!=stress_count:raise ValueError('Stress ornekleri eksik')
    dev=source_round(groups['val'],0)[:development]
    if len(dev)!=development:raise ValueError('Gelistirme ornekleri eksik')
    return [{'phase':p+1,'representative':base[p*representative:(p+1)*representative],
             'stress':stress,'development':dev} for p in range(phases)]


def load_sample(root,item,resolution,device):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    image=safe_path(root,item['image']);annotation=safe_path(root,item['annotation'])
    if sha256(image)!=item['image_sha256'] or sha256(annotation)!=item['annotation_sha256']:
        raise ValueError('Crop PNG veya annotation hash degeri farkli')
    a=read_json(annotation);validate_annotation(a)
    if (a['protocol_sha256']!=item['protocol_sha256'] or a['source_image']!=item['source_image'] or
            a['split']!=item['split'] or a['group']!=item['group'] or a['crop_index']!=item['crop_index'] or
            a['width']!=item['width'] or a['height']!=item['height'] or
            a['supervision_state']!=item['supervision_state'] or len(a['positive_targets'])!=item['positive_targets']):
        raise ValueError('Crop annotation/katalog kimligi farkli')
    with Image.open(image) as im:
        if im.size!=(a['width'],a['height']) or im.format!='PNG':raise ValueError('Dogal PNG boyutu farkli')
        pixels=np.array(im.convert('RGB'))
    pixels=cv2.resize(pixels,(resolution,resolution),interpolation=cv2.INTER_LINEAR)
    image_tensor=torch.from_numpy(pixels.transpose(2,0,1).copy()).unsqueeze(0).to(device=device,dtype=torch.float32)/255
    positives=a['positive_targets'];ignores=a['ignore_regions']
    ignore=torch.tensor([r['box_xyxy'] for r in ignores],dtype=torch.float32,device=device).reshape(-1,4)
    if len(ignore):ignore/=torch.tensor([a['width'],a['height'],a['width'],a['height']],device=device)
    return {'img':image_tensor,'cls':torch.tensor([r['class_id'] for r in positives],device=device,dtype=torch.long),
            'bboxes':torch.tensor([r['box_cxcywh_normalized'] for r in positives],device=device,dtype=torch.float32).reshape(-1,4),
            'batch_idx':torch.zeros(len(positives),device=device,dtype=torch.long),'ignore_boxes':[ignore]}
