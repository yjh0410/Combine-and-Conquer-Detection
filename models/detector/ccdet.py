import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..backbone import build_backbone
from ..neck import build_fpn, build_neck
from ..head import build_head

from .loss import Criterion


DEFAULT_SCALE_CLAMP = np.log(1000.)


# Combine-and-Conquer Detector
class CCDet(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 img_size=640,
                 num_classes=20,
                 topk=100,
                 conf_thresh = 0.01,
                 nms_thresh = 0.6,
                 trainable=False):
        super(CCDet, self).__init__()
        # config
        self.cfg = cfg
        # init
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.stride = self.cfg['stride']
        self.fpn_idx = cfg['fpn_idx']

        # generate anchors
        self.anchors = self.generate_anchors(img_size)

        # backbone
        self.backbone, bk_dims = build_backbone(
            bk_name=cfg['backbone'],
            pretrained=cfg['pretrained']
            )
        bk_dims = [bk_dims[layer_idx] for layer_idx in self.fpn_idx]

        # neck
        self.neck = build_neck(
            neck_name=cfg['neck_name'],
            neck_cfg=cfg,
            in_dim=bk_dims[-1],
            out_dim=cfg['fpn_dims'][-1]
            )

        # feat aggregation
        bk_dims[-1] = cfg['fpn_dims'][-1]
        self.fpn = build_fpn(
            fpn_name=cfg['fpn_name'],
            fpn_cfg=cfg,
            in_dims=bk_dims,
            out_dims=cfg['fpn_dims']
            )

        # detection head
        self.head = build_head(
            head_name=cfg['head_name'],
            head_cfg=cfg,
            in_dim=cfg['fpn_dims'][0], 
            out_dim=cfg['head_dim']
            )

        # pred
        self.hmp_pred = nn.Conv2d(cfg['head_dim'], self.num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(cfg['head_dim'], 4, kernel_size=1)
        self.iou_pred = nn.Conv2d(cfg['head_dim'], 1, kernel_size=1)

        if trainable:
            # init bias
            self.init_bias()

        # criterion
        if trainable:
            self.criterion = Criterion(
                cfg=cfg,
                device=device,
                loss_hmp_weight=cfg['loss_hmp_weight'],
                loss_reg_weight=cfg['loss_reg_weight'],
                loss_iou_weight=cfg['loss_iou_weight'],
                num_classes=num_classes
                )


    @torch.no_grad()
    def inference_single_image(self, x):
        """
            x: Tensor -> [1, C, H, W]
        """
        # backbone
        bk_feats = self.backbone(x)
        pyramid_feats = [bk_feats[layer_idx] for layer_idx in self.fpn_idx]

        # neck
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])
        
        # fpn
        pyramid_feats = self.fpn(pyramid_feats)

        # head
        top_feat = pyramid_feats[0]
        cls_feat, reg_feat = self.head(top_feat)

        # pred
        hmp_pred = self.hmp_pred(cls_feat)[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)
        reg_pred = self.reg_pred(reg_feat)[0].permute(1, 2, 0).contiguous().view(-1, 4)
        iou_pred = self.iou_pred(reg_feat)[0].permute(1, 2, 0).contiguous().view(-1, 1)

        # scores
        scores, labels = torch.max(torch.sqrt(hmp_pred.sigmoid() * iou_pred.sigmoid()), dim=-1)

        # [M, 2]
        anchors = self.anchors

        # topk
        if scores.shape[0] > self.topk:
            scores, indices = torch.topk(scores, self.topk)
            labels = labels[indices]
            reg_pred = reg_pred[indices]
            anchors = anchors[indices]

        # decode box: [M, 4]
        bboxes = self.decode_boxes(anchors, reg_pred) / self.img_size
        bboxes = bboxes.clamp(0., 1.)

        # to cpu
        scores = scores.cpu().numpy()    # [N,]
        labels = labels.cpu().numpy()    # [N,]
        bboxes = bboxes.cpu().numpy()    # [N, 4]

        # nms
        if self.cfg['use_nms']:
            keep = np.zeros(len(bboxes), dtype=np.int)
            for i in range(self.num_classes):
                inds = np.where(labels == i)[0]
                if len(inds) == 0:
                    continue
                c_bboxes = bboxes[inds]
                c_scores = scores[inds]
                c_keep = self.nms(c_bboxes, c_scores)
                keep[inds[c_keep]] = 1

            keep = np.where(keep > 0)
            bboxes = bboxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        return scores, labels, bboxes


    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # batch size
            B = x.size(0)

            # backbone
            bk_feats = self.backbone(x)
            pyramid_feats = [bk_feats[layer_idx] for layer_idx in self.fpn_idx]

            # neck
            pyramid_feats[-1] = self.neck(pyramid_feats[-1])
            
            # feat aggregation
            pyramid_feats = self.fpn(pyramid_feats)

            # detection head
            top_feat = pyramid_feats[0]
            cls_feat, reg_feat = self.head(top_feat)

            # pred
            hmp_pred = self.hmp_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)
            iou_pred = self.iou_pred(reg_feat)
        
            # [B, C, H, W] -> [B, H, W, C]
            hmp_pred = hmp_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            # [B, 4, H, W] -> [B, H, W, 4]
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            box_pred = self.decode_boxes(self.anchors[None], reg_pred)
            # [B, 1, H, W] -> [B, H, W, 1]
            iou_pred = iou_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)

            # output dict
            outputs = {"pred_hmp": hmp_pred,        # [B, M, C]
                       "pred_box": box_pred,        # [B, M, 4]
                       "pred_iou": iou_pred,        # [B, M, 1]
                       'stride': self.stride}       # Int

            # loss
            loss_dict = self.criterion(
                outputs = outputs, targets = targets)

            return loss_dict 


    def init_bias(self):  
        # Init head
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # init hmp pred
        nn.init.constant_(self.hmp_pred.bias, bias_value)


    def generate_anchors(self, img_size):
        img_h = img_w = img_size
        # generate grid cells
        fmp_w, fmp_h = img_w // self.stride, img_h // self.stride
        anchors_y, anchors_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        anchors = torch.stack([anchors_x, anchors_y], dim=-1).float()
        # [H, W, 2] -> [M, 2], M=HW
        anchors = anchors.to(self.device)
        anchors = anchors.view(-1, 2)
        
        return anchors


    def decode_boxes(self, anchors, reg_pred):
        """
        input box :  [wl, ht, wr, hb]
        output box : [x1, y1, x2, y2]
        """
        reg_pred = reg_pred.clamp(max=DEFAULT_SCALE_CLAMP).exp()
        output = torch.zeros_like(reg_pred)
        # x1 = x - wl
        # y1 = y - ht
        output[..., :2] = anchors - reg_pred[..., :2]
        # x2 = x + wr
        # y2 = y + hb
        output[..., 2:] = anchors + reg_pred[..., 2:]
        
        # rescale
        output = output * self.stride

        return output


    def nms(self, dets, scores):
        """"Pure Python NMS CCDet."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

