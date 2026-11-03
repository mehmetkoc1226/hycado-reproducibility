"""Crop annotation semantics; preserves ignored/other/partial objects explicitly."""
from __future__ import annotations

import math

from .data import parse_annotation
from .stability import inspect_labels
from .proposal_geometry import union_area


ANNOTATION_POLICY = {
    'id':'hycado_crop_annotations_v1',
    'positives':'score=1 and raw category 1..10; clipped to source image; exact audited duplicates removed',
    'minimum_visible_source_box_fraction':.5,
    'positive_minimum_size_filter':None,
    'partial_policy':'positive clipped intersection at >=0.5 area fraction; smaller positive fragments become ignore boxes',
    'original_ignore_policy':'category 0 or score 0 becomes ignore geometry',
    'others_policy':'category 11 with score 1 becomes training ignore geometry, not an eleventh target class',
    'overlapping_positive_and_ignore':'retain valid positive; matched-positive supervision must take precedence in future loss',
    'images':'unaltered native-resolution RGB crop, lossless PNG; no mask, resize, rotation, or EXIF transpose',
    'empty_crops':'retain; distinguish clean background, ignore-only, and fully ignored without positive supervision',
    'requires_ignore_aware_training_loader':True,
    'generic_YOLO_yaml_exported':False,
    'official_evaluator_equivalence_claimed':False,
}


def canonical_source_targets(text,width,height,clean_label_text,source='<annotation>'):
    """Reproduce Stage-1 label audit, then retain original pixel geometry/IDs."""
    parsed=parse_annotation(text,width,height,source=source)
    expected=''.join(str(r[0])+' '+' '.join(f'{v:.10f}' for v in r[1:])+'\n' for r in parsed['labels'])
    cleaned,duplicates,count=inspect_labels(expected,source)
    if cleaned!=clean_label_text:
        raise ValueError('Ham annotation ile denetlenmis Stage-1 etiketi eslesmiyor: '+source)
    seen,targets,ignores=set(),[],[]
    for number,row in enumerate(parsed['raw_rows'],1):
        x,y,w,h,score,category,truncation,occlusion=row
        box=[max(0.,x),max(0.,y),min(float(width),x+w),min(float(height),y+h)]
        if w<=0 or h<=0 or box[2]<=box[0] or box[3]<=box[1]:continue
        common={'source_record_index':number,'box_xyxy':box,'raw_category':int(category),
                'raw_score':int(score),'truncation':truncation,'occlusion':occlusion}
        if score==0 or category==0:
            ignores.append({**common,'reason':'source_ignore'})
        elif category==11:
            ignores.append({**common,'reason':'source_others'})
        else:
            a,b,c,d=box
            key=(int(category)-1,*(float(f'{v:.10f}') for v in
                  [(a+c)/(2*width),(b+d)/(2*height),(c-a)/width,(d-b)/height]))
            if key in seen:continue
            seen.add(key);targets.append({**common,'class_id':int(category)-1})
    if len(targets)!=count:
        raise ValueError('Ham piksel kutulari / temiz etiket sayisi farkli')
    return {'targets':targets,'ignore_regions':ignores,'duplicate_target_rows_removed':len(duplicates),
            'invalid_geometry_rows_omitted':len(parsed['issues'])}


def intersection(a,b):
    r=[max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])]
    return r if r[2]>r[0] and r[3]>r[1] else None


def crop_annotations(source_targets,source_ignores,crop,min_fraction=.5):
    if (len(crop)!=4 or any(not isinstance(v,int) or v<0 for v in crop) or
            crop[2]<=crop[0] or crop[3]<=crop[1] or not 0<min_fraction<=1):
        raise ValueError('Crop koordinatlari / gorunurluk esigi gecersiz')
    left,top,right,bottom=crop;width,height=right-left,bottom-top
    positives,ignored=[],[]
    for target in source_targets:
        visible=intersection(target['box_xyxy'],crop)
        if visible is None:continue
        a,b,c,d=target['box_xyxy']
        fraction=(visible[2]-visible[0])*(visible[3]-visible[1])/((c-a)*(d-b))
        local=[visible[0]-left,visible[1]-top,visible[2]-left,visible[3]-top]
        common={'source_record_index':target['source_record_index'],'box_xyxy':local,
                'visible_source_fraction':fraction,'source_box_xyxy':target['box_xyxy']}
        if fraction>=min_fraction-1e-10:
            x1,y1,x2,y2=local
            positives.append({**common,'class_id':target['class_id'],
                              'box_cxcywh_normalized':[(x1+x2)/(2*width),(y1+y2)/(2*height),
                                                       (x2-x1)/width,(y2-y1)/height],
                              'new_crop_truncation':fraction<1-1e-10,
                              'source_truncation':target['truncation'],'source_occlusion':target['occlusion']})
        else:
            ignored.append({**common,'reason':'partial_target_below_50pct',
                            'source_target_class_id':target['class_id']})
    for region in source_ignores:
        visible=intersection(region['box_xyxy'],crop)
        if visible is not None:
            ignored.append({'source_record_index':region['source_record_index'],
                            'box_xyxy':[visible[0]-left,visible[1]-top,visible[2]-left,visible[3]-top],
                            'reason':region['reason'],'raw_category':region['raw_category'],'raw_score':region['raw_score']})
    area=union_area([r['box_xyxy'] for r in ignored])/(width*height)
    if not -1e-10<=area<=1+1e-10:raise ValueError('Ignore alan orani gecersiz')
    state=('positive' if positives else 'clean_background' if not ignored else
           'fully_ignored_no_positive' if area>=1-1e-10 else 'ignore_with_background')
    document={'schema':'hycado_crop_annotation_v1','width':width,'height':height,'source_crop_xyxy':list(crop),
              'coordinate_system':'local_continuous_xyxy_half_open; no inclusive +1',
              'positive_targets':positives,'ignore_regions':ignored,'ignore_union_fraction':min(1.,max(0.,area)),
              'supervision_state':state,'requires_ignore_aware_loader':True}
    validate_annotation(document)
    return document


def validate_annotation(document):
    w,h=document['width'],document['height']
    if w<=0 or h<=0 or document.get('requires_ignore_aware_loader') is not True:
        raise ValueError('Crop anotasyon sozlesmesi gecersiz')
    for key in ['positive_targets','ignore_regions']:
        for row in document[key]:
            box=row['box_xyxy']
            if (len(box)!=4 or not all(math.isfinite(v) for v in box) or not
                    (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h)):
                raise ValueError('Crop ici kutu siniri/boyutu gecersiz')
    for row in document['positive_targets']:
        if not isinstance(row['class_id'],int) or not 0<=row['class_id']<10:
            raise ValueError('Crop hedef sinifi gecersiz')
        x,y,bw,bh=row['box_cxcywh_normalized']
        if not all(math.isfinite(v) for v in [x,y,bw,bh]) or bw<=0 or bh<=0 or not (0<=x<=1 and 0<=y<=1):
            raise ValueError('Normalize crop kutusu gecersiz')
        decoded=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
        if any(abs(a-b)>1e-6 for a,b in zip(decoded,row['box_xyxy'])):
            raise ValueError('Normalize kutu ile piksel kutusu eslesmiyor')
