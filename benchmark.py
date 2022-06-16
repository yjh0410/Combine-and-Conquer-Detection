import argparse
import numpy as np
import time
import os
import torch

from dataset.coco import COCODataset, coco_class_index, coco_class_labels
from dataset.utils.transforms import ValTransforms
from utils.misc import load_weight
from utils.com_paras_flops import FLOPs_and_Params
from utils.fuse_conv_bn import fuse_conv_bn

from config import build_config
from models.build import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Combine-and-Conquer Object Detection')
    # Model
    parser.add_argument('-v', '--version', default='ccdet', type=str,
                        help='build ccdet')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse conv and bn')

    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the min size of input image')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')

    return parser.parse_args()


def test(model, device, img_size, testset, transform):
    # Step-1: Compute FLOPs and Params
    FLOPs_and_Params(model=model, 
                     img_size=img_size, 
                     device=device)

    # Step-2: Compute FPS
    num_images = 2002
    total_time = 0
    count = 0
    with torch.no_grad():
        for index in range(num_images):
            if index % 500 == 0:
                print('Testing image {:d}/{:d}....'.format(index+1, num_images))
            image, _ = testset.pull_image(index)

            orig_h, orig_w, _ = image.shape
            orig_size = np.array([[orig_w, orig_h, orig_w, orig_h]])

            # pre-process
            x = transform(image)[0]
            x = x.unsqueeze(0).to(device)

            # star time
            torch.cuda.synchronize()
            start_time = time.perf_counter()    

            # inference
            scores, labels, bboxes = model(x)
            
            # rescale
            bboxes *= orig_size

            # end time
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            # print("detection time used ", elapsed, "s")
            if index > 1:
                total_time += elapsed
                count += 1
            
        print('- FPS :', 1.0 / (total_time / count))



if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg, m_cfg = build_config('coco', args.version)

    # dataset
    print('test on coco-val ...')
    data_root = os.path.join(d_cfg['data_root'])
    class_names = coco_class_labels
    class_indexs = coco_class_index
    num_classes = 80
    dataset = COCODataset(
                data_root=data_root,
                image_set='val2017',
                is_train=False)

    # build model
    model = build_model(
        model_cfg=m_cfg,
        device=device,
        img_size=args.img_size,
        num_classes=num_classes,
        is_train=False,
        eval_mode=False
        )

    # # load trained weight
    # model = load_weight(
    #     device=device, 
    #     model=model, 
    #     path_to_ckpt=args.weight
    #     )

    # transform
    transform = ValTransforms(
        img_size=args.img_size,
        format=d_cfg['format'],
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std']
        )


    # fuse conv bn
    if args.fuse_conv_bn:
        print('fuse conv and bn ...')
        model = fuse_conv_bn(model)

    # run
    test(
        model=model, 
        img_size=args.img_size,
        device=device, 
        testset=dataset,
        transform=transform
        )
