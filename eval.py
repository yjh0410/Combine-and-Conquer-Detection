import os
import argparse
import torch

from evaluator.voc_evaluator import VOCEvaluator
from evaluator.coco_evaluator import COCOEvaluator
from evaluator.widerface_evaluator import WiderFaceEvaluator

from dataset.utils.transforms import ValTransforms
from utils.misc import load_weight

from config import build_config
from models.build import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Combine-and-Conquer Object Detection')
    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='img_size')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')

    # model
    parser.add_argument('-v', '--version', default='ccdet',
                        help='ccdet')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco-val',
                        help='coco-val, coco-test, widerface-val, widerface-test.')
    # TTA
    parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                        help='use test time augmentation.')

    return parser.parse_args()


def voc_test(d_cfg, model, device, transform):
    evaluator = VOCEvaluator(data_root=d_cfg['data_root'],
                             device=device,
                             transform=transform,
                             display=True)

    # VOC evaluation
    evaluator.evaluate(model)


def coco_test(d_cfg, model, device, transform, test=False, test_aug=False):
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOEvaluator(
            data_root=d_cfg['data_root'],
            device=device,
            testset=True,
            transform=transform,
            test_aug=test_aug
            )

    else:
        # eval
        evaluator = COCOEvaluator(
            data_root=d_cfg['data_root'],
            device=device,
            testset=False,
            transform=transform,
            test_aug=test_aug
            )

    # COCO evaluation
    evaluator.evaluate(model)


def widerface_test(d_cfg, model, device, transform, test=False, test_aug=False):
    evaluator = WiderFaceEvaluator(
        data_root=d_cfg['data_root'],
        device=device,
        transform=transform,
        image_set='test' if test else 'val',
        test_aug=test_aug
        )

    # VOC evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg, m_cfg = build_config(args.dataset, args.version)

    # build model
    model = build_model(
        cfg=m_cfg,
        device=device,
        img_size=args.img_size,
        num_classes=d_cfg['num_classes'],
        is_train=False
        )

    # load weight
    model = load_weight(
        device=device, 
        model=model, 
        path_to_ckpt=args.weight
        )
    
    # transform
    transform = ValTransforms(
        img_size=args.img_size,
        format=d_cfg['format'],
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std']
        )

    # evaluation
    with torch.no_grad():
        if args.dataset == 'voc':
            voc_test(d_cfg, model, device, transform)
        elif args.dataset == 'coco':
            coco_test(d_cfg, model, device, transform, test=False, test_aug=args.test_aug)
        elif args.dataset == 'coco-test':
            coco_test(d_cfg, model, device, transform, test=True, test_aug=args.test_aug)
        elif args.dataset == 'widerface-val':
            widerface_test(d_cfg, model, device, transform, test=False, test_aug=args.test_aug)
        elif args.dataset == 'widerface-test':
            widerface_test(d_cfg, model, device, transform, test=True, test_aug=args.test_aug)
