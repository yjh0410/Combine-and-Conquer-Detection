import torch
import torch.distributed as dist

import time
import os
import numpy as np

from utils import distributed_utils


def train_with_warmup(epoch,
                      total_epochs,
                      args, 
                      device, 
                      ema,
                      model, 
                      img_size, 
                      dataloader, 
                      optimizer, 
                      warmup_scheduler,
                      scaler):
    epoch_size = len(dataloader)
    t0 = time.time()
    # train one epoch
    for iter_i, (images, targets) in enumerate(dataloader):
        ni = iter_i + epoch * epoch_size
        # warmup
        warmup_scheduler.warmup(ni, optimizer)

        # to device
        images = images.to(device)
        targets = targets.to(device)

        # inference
        with torch.cuda.amp.autocast(enabled=args.fp16):
            loss_dict = model(images, targets=targets)
            losses = loss_dict['losses']

        # reduce            
        loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

        # check loss
        if torch.isnan(losses):
            print('loss is NAN !!')
            continue

        # Backward and Optimize
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        # ema
        if args.ema:
            ema.update(model)

        # display
        if distributed_utils.is_main_process() and iter_i % 10 == 0:
            t1 = time.time()
            cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
            # basic infor
            log =  '[Epoch: {}/{}]'.format(epoch+1, total_epochs)
            log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
            log += '[lr: {:.6f}]'.format(cur_lr[0])
            # loss infor
            for k in loss_dict_reduced.keys():
                log += '[{}: {:.2f}]'.format(k, loss_dict[k])

            # other infor
            log += '[time: {:.2f}]'.format(t1 - t0)
            log += '[size: {}]'.format(img_size)

            # print log infor
            print(log, flush=True)
            
            t0 = time.time()


def train_one_epoch(epoch,
                    total_epochs,
                    args, 
                    device, 
                    ema,
                    model, 
                    img_size, 
                    dataloader, 
                    optimizer,
                    scaler):
    epoch_size = len(dataloader)
    t0 = time.time()
    # train one epoch
    for iter_i, (images, targets) in enumerate(dataloader):
        ni = iter_i + epoch * epoch_size
        # to device
        images = images.to(device)
        targets = targets.to(device)

        # inference
        with torch.cuda.amp.autocast(enabled=args.fp16):
            loss_dict = model(images, targets=targets)
            losses = loss_dict['losses']

        # reduce            
        loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

        # check loss
        if torch.isnan(losses):
            print('loss is NAN !!')
            continue

        # Backward and Optimize
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        # ema
        if args.ema:
            ema.update(model)

        # display
        if distributed_utils.is_main_process() and iter_i % 10 == 0:
            t1 = time.time()
            cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
            # basic infor
            log =  '[Epoch: {}/{}]'.format(epoch+1, total_epochs)
            log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
            log += '[lr: {:.6f}]'.format(cur_lr[0])
            # loss infor
            for k in loss_dict_reduced.keys():
                log += '[{}: {:.2f}]'.format(k, loss_dict[k])

            # other infor
            log += '[time: {:.2f}]'.format(t1 - t0)
            log += '[size: {}]'.format(img_size)

            # print log infor
            print(log, flush=True)
            
            t0 = time.time()


def val_one_epoch(args, 
                  model, 
                  evaluator,
                  optimizer,
                  epoch,
                  best_map,
                  path_to_save):
    # check evaluator
    if distributed_utils.is_main_process():
        if evaluator is None:
            print('No evaluator ... save model and go on training.')
            print('Saving state, epoch: {}'.format(epoch + 1))
            weight_name = '{}_epoch_{}.pth'.format(args.version, epoch + 1)
            checkpoint_path = os.path.join(path_to_save, weight_name)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args}, 
                        checkpoint_path)                      
            
        else:
            print('eval ...')
            # set eval mode
            model.trainable = False
            model.eval()

            # evaluate
            evaluator.evaluate(model)

            cur_map = evaluator.map
            if cur_map > best_map:
                # update best-map
                best_map = cur_map
                # save model
                print('Saving state, epoch:', epoch + 1)
                weight_name = '{}_epoch_{}_{:.2f}.pth'.format(args.version, epoch + 1, best_map*100)
                checkpoint_path = os.path.join(path_to_save, weight_name)
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'args': args}, 
                            checkpoint_path)                      

            # set train mode.
            model.trainable = True
            model.train()

    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()

    return best_map
