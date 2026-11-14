"""Exact original P2 SAHI factory/inference function bodies; no forced FP16 or backend change."""
EXPECTED_NAMES=['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

def sahi_factory(weight):
    from sahi import AutoDetectionModel
    model=AutoDetectionModel.from_pretrained(model_type='ultralytics',
            model_path=str(weight),confidence_threshold=.001,device='cuda:0',image_size=800)
    names=model.model.names
    names=[names[i] for i in range(len(names))] if isinstance(names,dict) else list(names)
    if names!=EXPECTED_NAMES:raise ValueError('SAHI model class names changed')
    return model

def sahi_predict(model,image,slice_size):
    from PIL import Image
    import numpy as np
    import torch
    from sahi.predict import get_sliced_prediction
    with Image.open(image) as im:
        w,h=im.size;rgb=np.array(im.convert('RGB'))
    if rgb.shape!=(h,w,3):raise ValueError('Decoded image size mismatch')
    torch.cuda.synchronize()
    pred=get_sliced_prediction(rgb,model,slice_height=slice_size,slice_width=slice_size,
              overlap_height_ratio=.2,overlap_width_ratio=.2,perform_standard_pred=True,
              postprocess_type='NMS',postprocess_match_metric='IOU',
              postprocess_match_threshold=.5,postprocess_class_agnostic=False,verbose=0)
    torch.cuda.synchronize()
    rows=[]
    for item in pred.object_prediction_list:
        xy=item.bbox.to_xyxy();rows.append([*map(float,xy),float(item.score.value),int(item.category.id)])
    return w,h,rows,0
