import math
import os
import cv2
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluator.coco_evaluator import COCOEvaluator
from evaluator.voc_evaluator import VOCEvaluator
from evaluator.widerface_evaluator import WiderFaceEvaluator

from dataset.voc import VOCDetection
from dataset.coco import COCODataset
from dataset.widerface import WIDERFaceDetection
from dataset.crowdhuman import CrowdHumanDetection

from dataset.utils.transforms import BaseTransforms, TrainTransforms, ValTransforms


def build_dataset(d_cfg, m_cfg, args, device):
    # transform
    trans_config = d_cfg['transforms']
    print('==============================')
    print('TrainTransforms: {}'.format(trans_config))
    color_augment = BaseTransforms(
        img_size=d_cfg['train_size'],
        random_size=d_cfg['random_size'],
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std'],
        format=d_cfg['format']
    )
    train_transform = TrainTransforms(
        trans_config=trans_config,
        img_size=d_cfg['train_size'],
        random_size=d_cfg['random_size'],
        format=d_cfg['format'],
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std']
        )
    val_transform = ValTransforms(
        img_size=d_cfg['test_size'],
        format=d_cfg['format'],
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std']
        )
    # dataset
    if args.dataset == 'voc':
        num_classes = 20
        # dataset
        dataset = VOCDetection(
            img_size=d_cfg['train_size'],
            data_root=d_cfg['data_root'],
            stride=m_cfg['stride'],
            transform=train_transform,
            color_augment=color_augment,
            mosaic_prob=d_cfg['mosaic_prob'],
            mixup_prob=d_cfg['mixup_prob'],
            is_train=True)
        # evaluator
        evaluator = VOCEvaluator(
            data_root=d_cfg['data_root'],
            device=device,
            transform=val_transform)

    elif args.dataset == 'coco':
        num_classes = 80
        # dataset
        dataset = COCODataset(
            img_size=d_cfg['train_size'],
            stride=m_cfg['stride'],
            data_root=d_cfg['data_root'],
            image_set='train2017',
            transform=train_transform,
            color_augment=color_augment,
            mosaic_prob=d_cfg['mosaic_prob'],
            mixup_prob=d_cfg['mixup_prob'],
            is_train=True)
        # evaluator
        evaluator = COCOEvaluator(
            data_root=d_cfg['data_root'],
            device=device,
            transform=val_transform)

    elif args.dataset == 'widerface':
        num_classes = 1
        # dataset
        dataset = WIDERFaceDetection(
            data_root=d_cfg['data_root'],
            img_size=d_cfg['train_size'],
            stride=m_cfg['stride'],
            image_set='train',
            transform=train_transform,
            color_augment=color_augment,
            mosaic_prob=d_cfg['mosaic_prob'],
            mixup_prob=d_cfg['mixup_prob'],
            is_train=True)
        # evaluator
        evaluator = WiderFaceEvaluator(
            data_root=d_cfg['data_root'],
            device=device,
            transform=val_transform)
    

    else:
        print('unknow dataset !!')
        exit(0)

    print('==============================')
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))

    return dataset, evaluator


def build_dataloader(args, dataset, batch_size, collate_fn=None):
    # distributed
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler, 
                                                        batch_size, 
                                                        drop_last=True)

    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_sampler=batch_sampler_train,
                                             collate_fn=collate_fn, 
                                             num_workers=args.num_workers,
                                             pin_memory=True)
    
    return dataloader
    

def load_weight(device, model, path_to_ckpt):
    checkpoint = torch.load(path_to_ckpt, map_location='cpu')
    # checkpoint state dict
    checkpoint_state_dict = checkpoint.pop("model")
    # model state dict
    model_state_dict = model.state_dict()
    # check
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint_state_dict.pop(k)
        else:
            checkpoint_state_dict.pop(k)
            print(k)

    model.load_state_dict(checkpoint_state_dict)
    model = model.to(device).eval()
    print('Finished loading model!')

    return model


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class CollateFunc(object):
    def __call__(self, batch):
        targets = []
        images = []

        for sample in batch:
            image = sample[0]
            target = sample[1]

            images.append(image)
            targets.append(target)

        images = torch.stack(images, 0)   # [B, C, H, W]
        targets = torch.stack(targets, 0)  # [B, H, W, C]

        return images, targets


# Model EMA
class ModelEMA(object):
    def __init__(self, model, decay=0.9999, updates=0):
        # create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000.))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()
