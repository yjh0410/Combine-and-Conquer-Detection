import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_ops import *
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='none'):
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(input=logits, 
                                                    target=targets, 
                                                    reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()

    elif reduction == "sum":
        loss = loss.sum()

    return loss


class Criterion(object):
    def __init__(self, 
                 cfg, 
                 device, 
                 loss_hmp_weight=1.0, 
                 loss_reg_weight=1.0,
                 loss_iou_weight=1.0,
                 num_classes=80):
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.loss_hmp_weight = loss_hmp_weight
        self.loss_reg_weight = loss_reg_weight
        self.loss_iou_weight = loss_iou_weight


    def loss_heatmap(self, pred, target, num_bboxes):
        loss = sigmoid_focal_loss(pred, target, reduction='none')
        loss = loss.sum() / num_bboxes

        return loss


    def loss_bboxes(self, pred_box, tgt_box, num_bboxes):
        ious = get_ious(pred_box,
                         tgt_box,
                         box_mode="xyxy",
                         iou_type='giou')

        loss = (1.0 - ious).sum() / num_bboxes

        return loss, ious


    def loss_ious(self, pred_iou, tgt_iou, num_bboxes):
        loss = F.binary_cross_entropy_with_logits(
            pred_iou, tgt_iou, reduction='none'
            )

        loss = loss.sum() / num_bboxes

        return loss


    def __call__(self, outputs, targets):
        """
            outputs['pred_hmp']: (Tensor) [B, M, C]
            outputs['pred_box']: (Tensor) [B, M, 4]
            outputs['pred_iou']: (Tensor) [B, M, 1]
            outputs['stride']: (Int) stride of the model output
            targets: (Tensor) [B, H, W, C+4+1]
        """
        device = outputs['pred_hmp'].device
        # targets
        gt_heatmaps = targets[..., :self.num_classes]
        gt_bboxes = targets[..., self.num_classes:self.num_classes+4]
        gt_fg_mask = targets[..., self.num_classes+4:]

        # [B, M, C] -> [BM, C]
        pred_hmp = outputs['pred_hmp'].view(-1, self.num_classes)
        pred_box = outputs['pred_box'].view(-1, 4)
        pred_iou = outputs['pred_iou'].view(-1)
        
        gt_heatmaps = gt_heatmaps.view(-1, self.num_classes).to(device)
        gt_bboxes = gt_bboxes.view(-1, 4).to(device)
        gt_fg_mask = gt_fg_mask.view(-1).to(device)
        foreground_idxs = (gt_fg_mask > 0)

        num_foreground = foreground_idxs.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        # heatmap loss
        loss_hmp = self.loss_heatmap(
            pred_hmp,
            gt_heatmaps,
            num_foreground
            )

        # bboxes loss
        matched_pred_delta = pred_box[foreground_idxs]
        matched_tgt_delta = gt_bboxes[foreground_idxs]
        loss_bboxes, ious = self.loss_bboxes(
            matched_pred_delta, 
            matched_tgt_delta,
            num_foreground
            )

        # iou loss
        matched_pred_iou = pred_iou[foreground_idxs]
        matched_tgt_iou = ious.clone().detach().clamp(0.)
        loss_ious = self.loss_ious(
            matched_pred_iou, 
            matched_tgt_iou, 
            num_foreground
            )

        # total loss
        losses = self.loss_hmp_weight * loss_hmp + \
                 self.loss_reg_weight * loss_bboxes + \
                 self.loss_iou_weight * loss_ious

        loss_dict = dict(
                loss_hmp = loss_hmp,
                loss_bboxes = loss_bboxes,
                loss_ious = loss_ious,
                losses = losses
        )

        return loss_dict
    

if __name__ == "__main__":
    pass