"""Real pinned YOLOv9-C CPU inference; synthetic inputs and checkpoint only."""
import copy
import json
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from ultralytics.nn.tasks import DetectionModel
from hycado_start import proposals
from hycado_start.data import NAMES
from hycado_start.common import sha256, write_json

torch.set_num_threads(2)
torch.manual_seed(42)
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    model = DetectionModel('yolov9c.yaml', nc=10, verbose=False)
    model.names = dict(enumerate(NAMES))
    with torch.no_grad():
        for branch in model.model[-1].cv3:
            branch[-1].bias.fill_(0.)
    train_model = copy.deepcopy(model)
    # Poison the train-model field: inference must use the EMA field, not train model.
    with torch.no_grad():
        next(train_model.parameters()).fill_(float('nan'))
    checkpoint = root/'synthetic_checkpoint.pt'
    torch.save({'model':train_model, 'ema':model, 'protocol_sha256':'synthetic',
                'epoch':79,'history':[{'epoch':i} for i in range(1,81)]},checkpoint)
    del model, train_model
    ema = proposals.load_ema(checkpoint,'synthetic')
    assert torch.isfinite(next(ema.parameters())).all()
    output = root/'output';output.mkdir()
    config = {'inference': {'imgsz':800,'batch':1,'half':True,'rect':True,'conf':.001,
              'iou':.7,'max_det':1000,'agnostic_nms':False,'augment':False,'device':'0',
              'cpu_threads':2,'opencv_threads':1,'seed':42},
              'coverage':{'thresholds':[.05,.15,.3],'top_k':[10,30,60],'context_scales':[1.5,2.,3.]}}
    records=[]
    for split,shape in [('train',(53,97)),('val',(79,61))]:
        image = root/'data/images'/split/'test.jpg';image.parent.mkdir(parents=True)
        cv2.imwrite(str(image), np.random.default_rng(2).integers(0,256,(*shape,3),dtype=np.uint8))
        label = root/'data/labels'/split/'test.txt';label.parent.mkdir(parents=True)
        label.write_text('4 .5 .5 .2 .2\n')
        records.append({'image':f'images/{split}/test.jpg','source_image':f'images/train/{split}.jpg',
                        'split':split,'image_sha256':sha256(image),'gt_count':1})
    def factory():return proposals.make_predictor(ema,config,output,_cpu_test=True)
    prediction = proposals.cache_predictions(records,root/'data',output,'test',factory)
    assert prediction['new_this_invocation']==2
    nonempty_count = 0
    for record, shape in zip(records,[(53,97),(79,61)]):
        c=proposals.validate_cache(proposals.load_json(proposals.prediction_path(output,record)),record,'test')
        assert (c['height'],c['width'])==shape
        assert len(c['detections']) > 0
        nonempty_count += len(c['detections'])
        for box in c['detections']:
            assert 0 <= box[0] < box[2] <= shape[1]
            assert 0 <= box[1] < box[3] <= shape[0]
    def must_not_load():raise AssertionError('Cache-only restart must not reload model')
    resumed=proposals.cache_predictions(records,root/'data',output,'test',must_not_load)
    assert resumed['reused_this_invocation']==2
    coverage=proposals.measure_coverage(records,root/'data',output,'test',config)
    assert coverage['development_images']==1 and coverage['configurations']==27
    print('CPU_INTEGRATION_PASS',json.dumps({'model':'actual YOLOv9-C 10 classes, synthetic random weights',
          'torch':torch.__version__,'images':2,'nonempty_detection_count':nonempty_count,'native_shapes_verified':True,'EMA_field_selected':True,
          'cache_only_restart':True,'coverage_configurations':27,'CUDA_FP16_tested':False}))
