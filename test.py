import os
import cv2
import time
import argparse
import numpy as np
from numpy import random
import torch
import torch.backends.cudnn as cudnn

from dataset.voc import VOCDetection, VOC_CLASSES
from dataset.coco import COCODataset, coco_class_index, coco_class_labels
from dataset.widerface import WIDERFaceDetection
from dataset.crowdhuman import CrowdHumanDetection
from dataset.transforms import ValTransforms
from utils.misc import TestTimeAugmentation

from models.build import build_model


parser = argparse.ArgumentParser(description='Combine-and-Conquer Object Detection')
# basic
parser.add_argument('-size', '--img_size', default=640, type=int,
                    help='img_size')
parser.add_argument('--show', action='store_true', default=False,
                    help='show the visulization results.')
parser.add_argument('-vs', '--visual_threshold', default=0.5, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')
parser.add_argument('--save_folder', default='det_results/', type=str,
                    help='Dir to save results')
parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                help='distributed training')
parser.add_argument('--local_rank', type=int, default=0, 
                    help='local_rank')

# model
parser.add_argument('-v', '--version', default='ccdet',
                    help='ccdet')
parser.add_argument('--stride', type=int, default=4, 
                    help='output stride')
parser.add_argument('-bk', '--backbone', default='r18',
                    help='r18, r50, r101')
parser.add_argument('--weight', default='weight/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--nms_thresh', default=0.45, type=float,
                    help='NMS threshold')
parser.add_argument('-nms', '--use_nms', action='store_true', default=False,
                    help='use nms.')
parser.add_argument('--topk', default=100, type=int,
                    help='topk prediction')
                    
# dataset
parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                    help='data root')
parser.add_argument('-d', '--dataset', default='coco',
                    help='coco.')
# TTA
parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                    help='use test time augmentation.')

args = parser.parse_args()



def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, 
              bboxes, 
              scores, 
              cls_inds, 
              vis_thresh, 
              class_colors, 
              class_names, 
              class_indexs=None, 
              dataset='voc'):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            if dataset == 'coco-val' or dataset == 'coco-test':
                cls_color = class_colors[int(cls_inds[i])]
                cls_id = class_indexs[int(cls_inds[i])]
            else:
                cls_id = int(cls_inds[i])
                cls_color = class_colors[cls_id]
                
            if len(class_names) > 1:
                mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img
        

def test(net, 
         device, 
         testset,
         transform, 
         vis_thresh, 
         class_colors=None, 
         class_names=None, 
         class_indexs=None, 
         show=False,
         test_aug=None, 
         dataset='voc'):
    num_images = len(testset)
    save_path = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(save_path, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        img, _ = testset.pull_image(index)
        h, w, _ = img.shape

        # to tensor
        x = transform(img)[0]
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # forward
        # test augmentation:
        if test_aug is not None:
            bboxes, scores, cls_inds = test_aug(x, net)
        else:
            # inference
            bboxes, scores, cls_inds = net(x)
        print("detection time used ", time.time() - t0, "s")
        
        # scale each detection back up to the image
        scale = np.array([[w, h, w, h]])
        # map the boxes to origin image scale
        bboxes *= scale

        # vis detection
        img_processed = visualize(img=img,
                            bboxes=bboxes,
                            scores=scores,
                            cls_inds=cls_inds,
                            vis_thresh=vis_thresh,
                            class_colors=class_colors,
                            class_names=class_names,
                            class_indexs=class_indexs,
                            dataset=dataset
                            )
        if show:
            cv2.imshow('detection', img_processed)
            cv2.waitKey(0)
        # save result
        cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img_processed)


if __name__ == '__main__':
    random.seed(0)
    # get device
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    if args.dataset == 'voc':
        print('test on voc ...')
        data_dir = os.path.join(args.root, 'VOCdevkit')
        dataset = VOCDetection(root=data_dir, 
                               image_sets=[('2007', 'test')], 
                               transform=None)
        class_names = VOC_CLASSES
        class_indexs = None
        num_classes = 20
        # test augmentation
        ta_nms = 0.4
        scale_range = [320, 1280, 128]

    elif args.dataset == 'coco':
        print('test on coco-val ...')
        data_dir = os.path.join(args.root, 'COCO')
        dataset = COCODataset(
                    data_dir=data_dir,
                    image_set='val2017',
                    img_size=args.img_size)
        class_names = coco_class_labels
        class_indexs = coco_class_index
        num_classes = 80
        # test augmentation
        ta_nms = 0.4
        scale_range = [320, 1280, 128]

    elif args.dataset == 'widerface':
        print('test on widerface ...')
        data_dir = os.path.join(args.root, 'WiderFace')
        dataset = WIDERFaceDetection(root=data_dir, 
                                     train=False,
                                     image_sets='val', 
                                     transform=None)
        class_names = ['face']
        class_indexs = None
        num_classes = 1
        # test augmentation
        ta_nms = 0.3
        scale_range = [512, 1536, 128]

    elif args.dataset == 'crowdhuman':
        data_dir = os.path.join(args.root, 'CrowdHuman')
        dataset = CrowdHumanDetection(root=data_dir, 
                                     train=False,
                                     transform=None)
        class_names = ['person']
        class_indexs = None
        num_classes = 1

    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # load model
    model = build_model(args, device, num_classes, local_rank=0, train=False)
                         
    model.load_state_dict(torch.load(args.weight, map_location=device), strict=False)
    model = model.to(device).eval()
    print('Finished loading model!')

    test_aug = TestTimeAugmentation(num_classes=num_classes,
                                    nms_thresh=ta_nms,
                                    scale_range=scale_range) if args.test_aug else None

    # test
    test(net=model, 
        device=device, 
        testset=dataset,
        transform=ValTransforms(args.img_size),
        vis_thresh=args.visual_threshold,
        class_colors=class_colors,
        class_names=class_names,
        class_indexs=class_indexs,
        show=args.show,
        test_aug=test_aug,
        dataset=args.dataset)
