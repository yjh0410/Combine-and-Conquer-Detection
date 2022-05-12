from __future__ import division

import os
import math
import argparse
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.cuda.amp as amp

from utils import distributed_utils
from utils.com_paras_flops import FLOPs_and_Params
from utils.misc import ModelEMA, CollateFunc, build_dataset, build_dataloader
from utils.solver.optimizer import build_optimizer
from utils.solver.warmup_schedule import build_warmup

from engine import train_with_warmup, train_one_epoch, val_one_epoch

from config import build_config
from models.build import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='CC-Det')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='path to save weight')
    parser.add_argument('--eval_epoch', default=10, type=int, 
                        help='after eval epoch, the model is evaluated on val dataset.')
    parser.add_argument('--fp16', dest="fp16", action="store_true", default=False,
                        help="Adopting mix precision training.")

    # model
    parser.add_argument('-v', '--version', default='ccdet_r18', type=str,
                        help='build CC-Det')
    parser.add_argument('-p', '--coco_pretrained', default=None, type=str,
                        help='coco pretrained weight')

    # dataset
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, widerface, crowdhuman')
    
    # train trick
    parser.add_argument('--ema', action='store_true', default=False,
                        help='Model EMA')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # dist
    print('World size: {}'.format(distributed_utils.get_world_size()))
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # amp
    scaler = amp.GradScaler(enabled=args.fp16)

    # d_cfg: dataset config
    # m_cfg: model config
    d_cfg, m_cfg = build_config(args.dataset, args.version)

    # dataset and evaluator
    dataset, evaluator = build_dataset(d_cfg, m_cfg, args, device)

    # dataloader
    batch_size = m_cfg['batch_size'] * distributed_utils.get_world_size()
    dataloader = build_dataloader(args, dataset, batch_size, CollateFunc())

    # build model
    model = build_model(
        cfg=m_cfg,
        device=device,
        img_size=d_cfg['train_size'],
        num_classes=d_cfg['num_classes'],
        is_train=True,
        coco_pretrained=args.coco_pretrained
        )
    model = model.to(device).train()

    # SyncBatchNorm
    if args.sybn and args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # compute FLOPs and Params
    if distributed_utils.is_main_process:
        model_copy = deepcopy(model_without_ddp)
        model_copy.trainable = False
        model_copy.eval()
        FLOPs_and_Params(
            model=model_copy,
            img_size=d_cfg['test_size'], 
            device=device)
        model_copy.trainable = True
        model_copy.train()
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()

    # EMA
    ema = ModelEMA(model) if args.ema else None

    # optimizer
    base_lr = m_cfg['base_lr'] * batch_size
    min_lr = base_lr * m_cfg['min_lr_ratio']
    optimizer = build_optimizer(
        model=model_without_ddp,
        base_lr=base_lr,
        name=m_cfg['optimizer'],
        momentum=m_cfg['momentum'],
        weight_decay=m_cfg['weight_decay']
        )
    
    # warmup scheduler
    wp_iter = len(dataloader) * m_cfg['wp_epoch']
    warmup_scheduler = build_warmup(
        name=m_cfg['warmup'],
        base_lr=base_lr,
        wp_iter=wp_iter,
        warmup_factor=m_cfg['warmup_factor']
        )


    # start training loop
    best_map = -1.0
    lr_schedule=True
    total_epochs = m_cfg['wp_epoch'] + m_cfg['max_epoch']
    for epoch in range(total_epochs):
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)            

        # train one epoch
        if epoch < m_cfg['wp_epoch']:
            # warmup training loop
            train_with_warmup(
                epoch=epoch,
                total_epochs=total_epochs,
                args=args, 
                device=device, 
                ema=ema,
                model=model, 
                img_size=d_cfg['train_size'], 
                dataloader=dataloader, 
                optimizer=optimizer, 
                warmup_scheduler=warmup_scheduler,
                scaler=scaler)

        else:
            if epoch == m_cfg['wp_epoch']:
                print('Warmup is Over !!!')
                warmup_scheduler.set_lr(optimizer, base_lr)
                
            # use cos lr decay
            T_max = total_epochs - m_cfg['no_aug_epoch']
            if epoch > T_max:
                print('Cosine annealing is over !!')
                lr_schedule = False
                for param_group in optimizer.param_groups:
                    param_group['lr'] = min_lr

            if lr_schedule:
                tmp_lr = min_lr + 0.5*(base_lr - min_lr)*(1 + math.cos(math.pi*epoch / T_max))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = tmp_lr

            # train one epoch
            train_one_epoch(
                epoch=epoch,
                total_epochs=total_epochs,
                args=args, 
                device=device,
                ema=ema, 
                model=model, 
                img_size=d_cfg['train_size'], 
                dataloader=dataloader, 
                optimizer=optimizer,
                scaler=scaler)
        
        # evaluation
        if (epoch % args.eval_epoch) == 0 or (epoch == total_epochs - 1):
            best_map = val_one_epoch(
                args=args, 
                model=ema.ema if args.ema else model_without_ddp, 
                evaluator=evaluator,
                optimizer=optimizer,
                epoch=epoch,
                best_map=best_map,
                path_to_save=path_to_save)

        # close mosaic augmentation
        if d_cfg['mosaic_prob'] > 0. and \
            total_epochs - epoch == m_cfg['no_aug_epoch']:
            print('close Mosaic Augmentation ...')
            dataloader.dataset.mosaic_prob = 0.
        # close mixup augmentation
        if d_cfg['mixup_prob'] > 0. and \
            total_epochs - epoch == m_cfg['no_aug_epoch']:
            print('close Mixup Augmentation ...')
            dataloader.dataset.mixup_prob = 0.


if __name__ == '__main__':
    train()
