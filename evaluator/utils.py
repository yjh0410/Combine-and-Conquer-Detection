import numpy as np
import torch
import torch.nn.functional as F


# Test Time Augmentation (TTA)
class TestTimeAugmentation(object):
    def __init__(self, num_classes=80, nms_thresh=0.4, scale_range=[512, 1536, 128]):
        self.num_classes = num_classes
        self.nms_thresh = nms_thresh
        self.scales = np.arange(scale_range[0], scale_range[1]+1, scale_range[2])


    def nms(self, dets, scores, nms_thresh=0.4):
        """"Pure Python NMS baseline."""
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
            inds = np.where(ovr <= nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def __call__(self, x, model):
        # x: Tensor -> [B, C, H, W]
        bboxes_list = []
        scores_list = []
        labels_list = []

        # multi scale
        for s in self.scales:
            if x.size(-1) == s and x.size(-2) == s:
                x_scale = x
            else:
                x_scale = F.interpolate(
                    input=x, 
                    size=(s, s), 
                    mode='bilinear', 
                    align_corners=False
                    )
            # reset anchors
            model.reset_anchors(s)
            # inference
            scores, labels, bboxes = model(x_scale)

            bboxes_list.append(bboxes)
            scores_list.append(scores)
            labels_list.append(labels)

            # Flip
            x_flip = torch.flip(x_scale, [-1])
            # inference
            scores, labels, bboxes = model(x_flip)
            # reflip bboxes
            bboxes = bboxes.copy()
            bboxes[:, 0::2] = 1.0 - bboxes[:, 2::-2]
            
            bboxes_list.append(bboxes)
            scores_list.append(scores)
            labels_list.append(labels)

        bboxes = np.concatenate(bboxes_list)
        scores = np.concatenate(scores_list)
        labels = np.concatenate(labels_list)

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores, self.nms_thresh)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return scores, labels, bboxes
