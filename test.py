import os
import cv2
import time
import argparse
import numpy as np
import torch

from evaluator.utils import TestTimeAugmentation

from dataset.voc import VOCDetection, VOC_CLASSES
from dataset.coco import COCODataset, coco_class_index, coco_class_labels
from dataset.widerface import WIDERFaceDetection
from dataset.crowdhuman import CrowdHumanDetection
from dataset.utils.transforms import ValTransforms
from utils.misc import load_weight

from config import build_config
from models.build import build_model

def parse_args():
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

    # dataset
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc, coco.')

    # model
    parser.add_argument('-v', '--version', default='ccdet_r18', type=str,
                        help='build ccdet')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')
                        
    # TTA
    parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                        help='use test time augmentation.')

    return parser.parse_args()



def plot_bbox_labels(image, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(
            image, 
            (x1, y1-t_size[1]), 
            (int(x1 + t_size[0] * text_scale), y1), 
            cls_color, -1
            )
        # put the test on the title bbox
        cv2.putText(
            image, label, 
            (int(x1), int(y1 - 5)), 0, 
            text_scale, (0, 0, 0), 1, 
            lineType=cv2.LINE_AA
            )

    return image


def visualize(image, 
              bboxes, 
              scores, 
              labels, 
              vis_thresh, 
              class_colors, 
              class_names, 
              class_indexs=None, 
              dataset='voc'):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            if dataset == 'coco' or dataset == 'coco-test':
                cls_color = class_colors[int(labels[i])]
                cls_id = class_indexs[int(labels[i])]
            else:
                cls_id = int(labels[i])
                cls_color = class_colors[cls_id]
                
            if len(class_names) > 1:
                mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
            image = plot_bbox_labels(
                image, bbox, mess, 
                cls_color, text_scale=ts
                )

    return image
        

@torch.no_grad()
def test(args, model, device, testset, transform, 
         class_colors=None, class_names=None, class_indexs=None, 
         show=False, vis_thresh=0.5, test_aug=None, dataset='voc'
         ):
    num_images = len(testset)
    save_path = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(save_path, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        image, _ = testset.pull_image(index)
        orig_h, orig_w, _ = image.shape
        orig_size = np.array([[orig_w, orig_h, orig_w, orig_h]])

        # to tensor
        x = transform(image)[0]
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # inference
        if test_aug is not None:
            # test augmentation:
            scores, labels, bboxes = test_aug(x, model)
        else:
            scores, labels, bboxes = model(x)
        print("Infer: {:.6f} s".format(time.time() - t0))
        
        # rescale
        bboxes *= orig_size

        # vis detection
        img_processed = visualize(
            image=image,
            bboxes=bboxes,
            scores=scores,
            labels=labels,
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
        cv2.imwrite(
            os.path.join(save_path, str(index).zfill(6) +'.jpg'), 
            img_processed
            )


if __name__ == '__main__':
    args = parse_args()

    # get device
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    if args.dataset == 'voc':
        # config
        d_cfg, m_cfg = build_config('voc', args.version)
        # dataset
        print('test on voc ...')
        dataset = VOCDetection(
            img_size=d_cfg['test_size'],
            data_root=d_cfg['data_root'],
            image_sets=[('2007', 'test')],
            is_train=False)
        
        class_names = VOC_CLASSES
        class_indexs = None
        num_classes = 20
        # test augmentation
        ta_nms = 0.4
        scale_range = [320, 800, 32]

    elif args.dataset == 'coco':
        # config
        d_cfg, m_cfg = build_config('coco', args.version)
        # dataset
        print('test on coco-val ...')
        dataset = COCODataset(
                    img_size=d_cfg['test_size'],
                    data_root=d_cfg['data_root'],
                    image_set='val2017')
        class_names = coco_class_labels
        class_indexs = coco_class_index
        num_classes = 80
        # test augmentation
        ta_nms = 0.4
        scale_range = [320, 800, 32]

    elif args.dataset == 'coco-test':
        # config
        d_cfg, m_cfg = build_config('coco', args.version)
        # dataset
        print('test on coco-test ...')
        dataset = COCODataset(
                    img_size=d_cfg['test_size'],
                    data_root=d_cfg['data_root'],
                    image_set='test2017')
        class_names = coco_class_labels
        class_indexs = coco_class_index
        num_classes = 80
        # test augmentation
        ta_nms = 0.4
        scale_range = [320, 800, 32]

    elif args.dataset == 'widerface':
        # config
        d_cfg, m_cfg = build_config('widerface', args.version)
        # dataset
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

    np.random.seed(0)
    class_colors = [
        (np.random.randint(255),
        np.random.randint(255),
        np.random.randint(255)
        ) for _ in range(num_classes)]

    # build model
    model = build_model(
        model_cfg=m_cfg,
        device=device,
        img_size=args.img_size,
        num_classes=num_classes,
        is_train=False,
        eval_mode=False
        )

    # load trained weight
    model = load_weight(
        device=device, 
        model=model, 
        path_to_ckpt=args.weight
        )

    # to eval
    model = model.to(device).eval()

    # transform
    transform = ValTransforms(
        img_size=args.img_size,
        format=d_cfg['format'],
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std']
        )

    # TTA
    test_aug = TestTimeAugmentation(
        num_classes=num_classes,
        nms_thresh=ta_nms,
        scale_range=scale_range
        ) if args.test_aug else None

    # test
    test(args=args,
         model=model, 
         device=device, 
         testset=dataset,
         transform=transform,
         vis_thresh=args.visual_threshold,
         class_colors=class_colors,
         class_names=class_names,
         class_indexs=class_indexs,
         show=args.show,
         test_aug=test_aug,
         dataset=args.dataset)
