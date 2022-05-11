import os
import argparse
import torch

from dataset.transforms import ValTransforms
from evaluator.voc_evaluator import VOCEvaluator
from evaluator.coco_evaluator import COCOEvaluator
from evaluator.widerface_evaluator import WiderFaceEvaluator

from models.build import build_model


parser = argparse.ArgumentParser(description='Combine-and-Conquer Object Detection')
# basic
parser.add_argument('-size', '--img_size', default=640, type=int,
                    help='img_size')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use cuda')
parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                help='distributed training')
parser.add_argument('--local_rank', type=int, default=0, 
                    help='local_rank')
# model
parser.add_argument('-v', '--version', default='ccdet',
                    help='ccdet')
parser.add_argument('-bk', '--backbone', default='r18',
                    help='r18, r50, r101')
parser.add_argument('--stride', type=int, default=4, 
                    help='output stride')
parser.add_argument('--weight', default='weight/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--nms_thresh', default=0.45, type=float,
                    help='NMS threshold')
parser.add_argument('-nms', '--use_nms', action='store_true', default=False,
                    help='use nms.')
parser.add_argument('--topk', default=300, type=int,
                    help='topk prediction')
# dataset
parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                    help='data root')
parser.add_argument('-d', '--dataset', default='coco-val',
                    help='coco-val, coco-test, widerface-val, widerface-test.')
# TTA
parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                    help='use test time augmentation.')

args = parser.parse_args()


def voc_test(model, device, img_size):
    data_dir = os.path.join(args.root, 'VOCdevkit')
    evaluator = VOCEvaluator(data_root=data_dir,
                                img_size=img_size,
                                device=device,
                                transform=ValTransforms(img_size),
                                display=True)

    # VOC evaluation
    evaluator.evaluate(model)


def coco_test(model, device, img_size, test=False, test_aug=False):
    data_dir = os.path.join(args.root, 'COCO')
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOEvaluator(
                        data_dir=data_dir,
                        img_size=img_size,
                        device=device,
                        testset=True,
                        transform=ValTransforms(img_size),
                        test_aug=test_aug
                        )

    else:
        # eval
        evaluator = COCOEvaluator(
                        data_dir=data_dir,
                        img_size=img_size,
                        device=device,
                        testset=False,
                        transform=ValTransforms(img_size),
                        test_aug=test_aug
                        )

    # COCO evaluation
    evaluator.evaluate(model)


def widerface_test(model, device, img_size, test=False, test_aug=False):
    data_dir = os.path.join(args.root, 'WiderFace')
    evaluator = WiderFaceEvaluator(data_root=data_dir,
                                device=device,
                                transform=ValTransforms(img_size),
                                image_set='test' if test else 'val',
                                test_aug=test_aug
                                )

    # VOC evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    # dataset
    if args.dataset == 'voc':
        print('eval on voc ...')
        num_classes = 20
    elif args.dataset == 'coco-val':
        print('eval on coco-val ...')
        num_classes = 80
    elif args.dataset == 'coco-test':
        print('eval on coco-test-dev ...')
        num_classes = 80
    elif args.dataset == 'widerface-val':
        print('eval on widerface-val ...')
        num_classes = 1
    elif args.dataset == 'widerface-test':
        print('eval on widerface-test ...')
        num_classes = 1
    else:
        print('unknow dataset !! we only support voc, coco-val, coco-test, widerface-val, widerface-test !!!')
        exit(0)

    # cuda
    if args.cuda:
        print('use cuda')
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load model
    model = build_model(args, device, num_classes, local_rank=0, train=False)

    model.load_state_dict(torch.load(args.weight, map_location=device), strict=False)
    model = model.to(device).eval()
    print('Finished loading model!')
    
    # evaluation
    with torch.no_grad():
        if args.dataset == 'voc':
            voc_test(model, device, args.img_size)
        elif args.dataset == 'coco-val':
            coco_test(model, device, args.img_size, test=False, test_aug=args.test_aug)
        elif args.dataset == 'coco-test':
            coco_test(model, device, args.img_size, test=True, test_aug=args.test_aug)
        elif args.dataset == 'widerface-val':
            widerface_test(model, device, args.img_size, test=False, test_aug=args.test_aug)
        elif args.dataset == 'widerface-test':
            widerface_test(model, device, args.img_size, test=True, test_aug=args.test_aug)
