"""Ignore-aware classification for the pinned Ultralytics 8.3.203 RT-DETR loss.

Only unmatched queries with >=0.5 predicted-box IoA against any ignore rectangle
lose classification supervision. Hungarian matching and matched box losses are
unchanged. This is an explicit training rule, not the official VisDrone metric.
"""
from __future__ import annotations

from collections import Counter
import torch
import torch.nn.functional as F
from ultralytics.models.utils.loss import RTDETRDetectionLoss


IGNORE_RULE = {
    'id':'hycado_rtdetr_ignore_query_v1',
    'threshold':.5,
    'measure':'intersection / predicted query box area against any one ignore rectangle',
    'box_coordinates':'normalized continuous xyxy; prediction cxcywh converted without clipping',
    'mask_gradient':'detached geometry; no gradient through query selection',
    'positive_priority':'every matched query retains all class channels and bbox/GIoU supervision',
    'scope':'unmatched classification in final, encoder, decoder auxiliary and denoising branches',
    'normalization':'original query-count and matched-GT denominators; ignored terms set to zero, no renormalization',
    'official_evaluator_equivalence_claimed':False,
}


def ignored_queries(boxes,regions,threshold=.5):
    """Return B x Q boolean mask; bounded chunks avoid Q x all-regions peaks."""
    boxes=boxes.detach().float()
    result=torch.zeros(boxes.shape[:2],device=boxes.device,dtype=torch.bool)
    if len(regions)!=len(boxes):raise ValueError('Ignore batch boyutu farkli')
    for b,rects in enumerate(regions):
        rects=rects.to(device=boxes.device,dtype=torch.float32).reshape(-1,4)
        if not rects.numel():continue
        xy,wh=boxes[b,:,:2],boxes[b,:,2:]
        lower,upper=xy-wh/2,xy+wh/2
        area=wh.clamp(min=0).prod(-1).clamp(min=1e-12)
        for chunk in rects.split(128):
            lo=torch.maximum(lower[:,None,:],chunk[None,:,:2])
            hi=torch.minimum(upper[:,None,:],chunk[None,:,2:])
            overlap=(hi-lo).clamp(min=0).prod(-1)/area[:,None]
            result[b]|=(overlap>=threshold).any(-1)
    return result


class IgnoreAwareRTDETRLoss(RTDETRDetectionLoss):
    def __init__(self,nc=10):
        super().__init__(nc=nc,use_vfl=True)
        self.audit=Counter();self._regions=None;self._query_boxes=None

    def forward(self,preds,batch,**kwargs):
        self._regions=batch['ignore_boxes'];self.audit=Counter()
        try:return super().forward(preds,batch,**kwargs)
        finally:self._regions=None;self._query_boxes=None

    def _get_loss(self,pred_bboxes,pred_scores,*args,**kwargs):
        self._query_boxes=pred_bboxes.detach()
        return super()._get_loss(pred_bboxes,pred_scores,*args,**kwargs)

    def _get_loss_class(self,pred_scores,targets,gt_scores,num_gts,postfix=''):
        raw=ignored_queries(self._query_boxes,self._regions,IGNORE_RULE['threshold'])
        matched=targets!=self.nc
        blocked=raw & ~matched
        self.audit['branch_calls']+=1
        self.audit['blocked_query_branch_instances']+=int(blocked.sum().item())
        self.audit['positive_query_branch_instances_retained']+=int(matched.sum().item())
        self.audit['positive_over_ignore_branch_instances_retained']+=int((raw & matched).sum().item())
        if postfix:self.audit['denoising_branch_calls']+=1
        if not blocked.any():
            return super()._get_loss_class(pred_scores,targets,gt_scores,num_gts,postfix)
        bs,nq=pred_scores.shape[:2]
        one_hot=torch.zeros((bs,nq,self.nc+1),device=targets.device,dtype=torch.int64)
        one_hot.scatter_(2,targets.unsqueeze(-1),1);one_hot=one_hot[...,:-1]
        if num_gts:
            quality=gt_scores.view(bs,nq,1)*one_hot
            weight=self.vfl.alpha*pred_scores.sigmoid().pow(self.vfl.gamma)*(1-one_hot)+quality*one_hot
            with torch.autocast(device_type=pred_scores.device.type,enabled=False):
                elements=F.binary_cross_entropy_with_logits(pred_scores.float(),quality.float(),reduction='none')*weight
        else:
            label=one_hot.float()
            elements=F.binary_cross_entropy_with_logits(pred_scores,label,reduction='none')
            probability=pred_scores.sigmoid()
            elements=elements*(1-(label*probability+(1-label)*(1-probability)))**self.fl.gamma
            self.fl.alpha=self.fl.alpha.to(device=pred_scores.device,dtype=pred_scores.dtype)
            elements=elements*(label*self.fl.alpha+(1-label)*(1-self.fl.alpha))
        value=(elements*(~blocked).unsqueeze(-1)).mean(1).sum()
        value/=max(num_gts,1)/nq
        return {'loss_class'+postfix:value.squeeze()*self.loss_gain['class']}


def forward_loss(model,batch,criterion):
    """RT-DETR native forward/denoising split, keeping the extra ignore field."""
    img=batch['img'];idx=batch['batch_idx'].long().view(-1)
    targets={'cls':batch['cls'].long().view(-1),'bboxes':batch['bboxes'],
             'batch_idx':idx,'gt_groups':[(idx==i).sum().item() for i in range(img.shape[0])],
             'ignore_boxes':batch['ignore_boxes']}
    outputs=model.predict(img,batch=targets)
    dec_boxes,dec_scores,enc_boxes,enc_scores,meta=outputs if model.training else outputs[1]
    if meta is None:dn_boxes=dn_scores=None
    else:
        dn_boxes,dec_boxes=torch.split(dec_boxes,meta['dn_num_split'],dim=2)
        dn_scores,dec_scores=torch.split(dec_scores,meta['dn_num_split'],dim=2)
    boxes=torch.cat([enc_boxes.unsqueeze(0),dec_boxes])
    scores=torch.cat([enc_scores.unsqueeze(0),dec_scores])
    losses=criterion((boxes,scores),targets,dn_bboxes=dn_boxes,dn_scores=dn_scores,dn_meta=meta)
    return sum(losses.values()),losses


def criterion_self_check(device='cpu'):
    """Gradient check on the actual execution device, including auxiliary/DN."""
    boxes=torch.tensor([.5,.5,.2,.2],device=device).repeat(2,1,3,1)
    logits=torch.zeros((2,1,3,10),device=device,requires_grad=True)
    dn_boxes=boxes[:,:,:2].clone();dn_scores=torch.zeros((2,1,2,10),device=device,requires_grad=True)
    batch={'cls':torch.tensor([3],device=device),'bboxes':boxes[0,0,:1].clone(),'gt_groups':[1],
           'ignore_boxes':[torch.tensor([[0.,0.,1.,1.]],device=device)]}
    criterion=IgnoreAwareRTDETRLoss(10)
    losses=criterion((boxes,logits),batch,dn_bboxes=dn_boxes,dn_scores=dn_scores,
                     dn_meta={'dn_pos_idx':[torch.tensor([0],device=device)],'dn_num_group':1})
    sum(losses.values()).backward()
    # The matcher may select any one of the tied regular queries. Each layer
    # must have exactly one query with nonzero score gradient; every DN layer
    # must retain its predetermined positive and suppress its negative.
    active=logits.grad.abs().sum(-1)>0
    if not torch.equal(active.sum(-1),torch.ones((2,1),device=device,dtype=torch.long)):
        raise AssertionError('Ana/aux ignore gradyan denetimi basarisiz')
    if not (dn_scores.grad[:,:,0].abs().sum()>0 and dn_scores.grad[:,:,1].abs().sum()==0):
        raise AssertionError('Denoising pozitif/ignore gradyan denetimi basarisiz')
    if criterion.audit['positive_over_ignore_branch_instances_retained']!=4:
        raise AssertionError('Pozitif onceligi dal sayisi farkli')
    return {'status':'PASSED','device':str(device),'main_aux_and_denoising_checked':True,
            'ignored_unmatched_score_gradient_zero':True,'matched_positive_gradient_retained':True,
            'audit':dict(criterion.audit)}
