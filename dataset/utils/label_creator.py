import numpy as np
import torch


class HMPCreator(object):
    """ Creator for Heatmap"""
    def __init__(self, num_classes, stride):
        self.num_classes = num_classes
        self.stride = stride


    def __call__(self, img_size, targets):
        img_h = img_w = img_size
        fmp_h = fmp_w = img_size // self.stride
        # prepare
        gt_heatmaps = np.zeros([fmp_h, fmp_w, self.num_classes])
        gt_bboxes = np.zeros([fmp_h, fmp_w, 4])
        gt_bboxes_weights = np.full([fmp_h, fmp_w, 1], -1.0)

        bboxes = targets['boxes']
        labels = targets['labels']

        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox.tolist()
            label = int(label)

            # compute center, width and height
            xc = (x2 + x1) * 0.5
            yc = (y2 + y1) * 0.5
            bw = x2 - x1
            bh = y2 - y1

            # check bbox
            if bw < 1. or bh < 1.:
                continue 

            # scale bbox
            bw_s = bw / self.stride
            bh_s = bh / self.stride

            # gaussian radius
            rw = max(int(bw_s / 6), 1)
            rh = max(int(bh_s / 6), 1)

            # scale box to feature size
            x1s = int(x1 / img_w * fmp_w)
            y1s = int(y1 / img_h * fmp_h)
            x2s = int(x2 / img_w * fmp_w)
            y2s = int(y2 / img_h * fmp_h)
            xc_s = xc / self.stride
            yc_s = yc / self.stride
            grid_x = int(xc_s)
            grid_y = int(yc_s)    

            img_area = (img_h * img_w)
            box_area = (x2 - x1) * (y2 - y1)
            # assign the target to center anchor
            gt_heatmaps[grid_y, grid_x, label] = 1.0
            gt_bboxes[grid_y, grid_x] = np.array([x1, y1, x2, y2])
            gt_bboxes_weights[grid_y, grid_x] = 2.0 - box_area / img_area

            # create a Gauss Heatmap for the target
            prev_hmp = gt_heatmaps[y1s:y2s, x1s:x2s, label]
            grid_x_mat, grid_y_mat = np.meshgrid(np.arange(x1s, x2s), np.arange(y1s, y2s))
            M = -(grid_x_mat - grid_x)**2 / (2*(rw)**2) \
                -(grid_y_mat - grid_y)**2 / (2*(rh)**2)
            cur_hmp = np.exp(M)

            gt_heatmaps[y1s:y2s, x1s:x2s, label] = np.maximum(cur_hmp, prev_hmp)

            # multi positive samples
            for i in range(grid_x - 1, grid_x + 2):
                for j in range(grid_y - 1, grid_y + 2):
                    if (j >=y1s and j < y2s) and (i >=x1s and i < x2s):
                        gt_bboxes[j, i] = np.array([x1, y1, x2, y2])
                        gt_bboxes_weights[j, i] = 2.0 - box_area / img_area
        
        targets = np.concatenate([
            gt_heatmaps, 
            gt_bboxes, 
            gt_bboxes_weights], axis=-1)

        return torch.from_numpy(targets).float() 
