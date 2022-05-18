# Model Configuration


m_config = {
    # CC-Det
    'ccdet_r18': {
        # backbone
        'backbone': 'resnet18',
        'pretrained': True,
        'stride': 4,
        'depthwise': False,
        # neck
        'neck_name': 'DE',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        # fpn
        'fpn_name': 'basicfpn',
        'fpn_dims': [64, 128, 256, 512],
        'fpn_idx': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_act': 'relu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        # head
        'head': 'decoupled_head',
        'head_k': 3,
        'head_dim': 64,
        'head_act': 'relu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 1,
        'num_reg_layers': 1,
        # post-process
        'nms_kernel': 5,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'test_topk': 100,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_iou_weight': 1.0,
        # training configuration
        'max_epoch': 200,
        'no_aug_epoch': 15,
        'batch_size': 32,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.01,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
    },
    
    'ccdet_r50': {
        # backbone
        'backbone': 'resnet50',
        'pretrained': True,
        'stride': 4,
        'depthwise': False,
        # neck
        'neck_name': 'DE',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        # fpn
        'fpn_name': 'basicfpn',
        'fpn_dims': [64, 128, 256, 512],
        'fpn_idx': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_act': 'relu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        # head
        'head': 'decoupled_head',
        'head_k': 3,
        'head_dim': 64,
        'head_act': 'relu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 1,
        'num_reg_layers': 1,
        # post-process
        'nms_kernel': 5,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'test_topk': 100,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_iou_weight': 1.0,
        # training configuration
        'max_epoch': 200,
        'no_aug_epoch': 15,
        'batch_size': 32,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.01,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
    },
    
    'ccdet_r101': {
        # backbone
        'backbone': 'resnet101',
        'pretrained': True,
        'stride': 4,
        'depthwise': False,
        # neck
        'neck_name': 'DE',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        # fpn
        'fpn_name': 'basicfpn',
        'fpn_dims': [64, 128, 256, 512],
        'fpn_idx': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_act': 'relu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        # head
        'head': 'decoupled_head',
        'head_k': 3,
        'head_dim': 64,
        'head_act': 'relu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 1,
        'num_reg_layers': 1,
        # post-process
        'nms_kernel': 5,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'test_topk': 100,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_iou_weight': 1.0,
        # training configuration
        'max_epoch': 200,
        'no_aug_epoch': 15,
        'batch_size': 16,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.01,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
    },
    
    'ccdet_d19': {
        # backbone
        'backbone': 'darknet19',
        'pretrained': True,
        'stride': 4,
        'depthwise': False,
        # neck
        'neck_name': 'DE',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        # fpn
        'fpn_name': 'basicfpn',
        'fpn_dims': [64, 128, 256, 512],
        'fpn_idx': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_act': 'lrelu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        # head
        'head': 'decoupled_head',
        'head_k': 3,
        'head_dim': 64,
        'head_act': 'lrelu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 1,
        'num_reg_layers': 1,
        # post-process
        'nms_kernel': 5,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'test_topk': 100,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_iou_weight': 1.0,
        # training configuration
        'max_epoch': 200,
        'no_aug_epoch': 15,
        'batch_size': 32,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.01,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
    },
    
    'ccdet_d53': {
        # backbone
        'backbone': 'darknet53',
        'pretrained': True,
        'stride': 4,
        'depthwise': False,
        # neck
        'neck_name': 'DE',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        # fpn
        'fpn_name': 'basicfpn',
        'fpn_dims': [64, 128, 256, 512],
        'fpn_idx': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_act': 'lrelu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        # head
        'head': 'decoupled_head',
        'head_k': 3,
        'head_dim': 64,
        'head_act': 'lrelu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 1,
        'num_reg_layers': 1,
        # post-process
        'nms_kernel': 5,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'test_topk': 100,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_iou_weight': 1.0,
        # training configuration
        'max_epoch': 200,
        'no_aug_epoch': 15,
        'batch_size': 16,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.01,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
    },
    
    'ccdet_vgg16': {
        # backbone
        'backbone': 'vgg16',
        'pretrained': True,
        'stride': 4,
        'depthwise': False,
        # neck
        'neck_name': 'DE',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        # fpn
        'fpn_name': 'basicfpn',
        'fpn_dims': [64, 128, 256, 512],
        'fpn_idx': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_act': 'relu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        # head
        'head': 'decoupled_head',
        'head_k': 3,
        'head_dim': 64,
        'head_act': 'relu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 1,
        'num_reg_layers': 1,
        # post-process
        'nms_kernel': 5,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'test_topk': 100,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_iou_weight': 1.0,
        # training configuration
        'max_epoch': 200,
        'no_aug_epoch': 15,
        'batch_size': 16,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.01,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
    },
    
    # CCDet with Modified CSPDarkNet
    'ccdet_s': {
        # backbone
        'backbone': 'yolox_backbone',
        'pretrained': True,
        'bk_act': 'silu',
        'stride': 4,
        'depthwise': False,
        'width': 0.5,
        'depth': 0.33,
        # neck
        'neck_name': 'DE',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        # fpn
        'fpn_name': 'basicfpn',
        'fpn_dims': [64, 128, 256, 512],
        'fpn_idx': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        # head
        'head': 'decoupled_head',
        'head_k': 3,
        'head_dim': 32,
        'head_act': 'silu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 1,
        'num_reg_layers': 1,
        # post-process
        'nms_kernel': 5,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'test_topk': 100,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_iou_weight': 1.0,
        # training configuration
        'max_epoch': 200,
        'no_aug_epoch': 15,
        'batch_size': 32,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.01,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
    },
    
    'ccdet_m': {
        # backbone
        'backbone': 'yolox_backbone',
        'pretrained': True,
        'bk_act': 'silu',
        'stride': 4,
        'depthwise': False,
        'width': 0.75,
        'depth': 0.67,
        # neck
        'neck_name': 'DE',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        # fpn
        'fpn_name': 'basicfpn',
        'fpn_dims': [64, 128, 256, 512],
        'fpn_idx': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        # head
        'head': 'decoupled_head',
        'head_k': 3,
        'head_dim': 48,
        'head_act': 'silu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 1,
        'num_reg_layers': 1,
        # post-process
        'nms_kernel': 5,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'test_topk': 100,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_iou_weight': 1.0,
        # training configuration
        'max_epoch': 200,
        'no_aug_epoch': 15,
        'batch_size': 32,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.01,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
    },
    
    'ccdet_l': {
        # backbone
        'backbone': 'yolox_backbone',
        'pretrained': True,
        'bk_act': 'silu',
        'stride': 4,
        'depthwise': False,
        'width': 1.0,
        'depth': 1.0,
        # neck
        'neck_name': 'DE',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        # fpn
        'fpn_name': 'basicfpn',
        'fpn_dims': [64, 128, 256, 512],
        'fpn_idx': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        # head
        'head': 'decoupled_head',
        'head_k': 3,
        'head_dim': 64,
        'head_act': 'silu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 1,
        'num_reg_layers': 1,
        # post-process
        'nms_kernel': 5,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'test_topk': 100,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_iou_weight': 1.0,
        # training configuration
        'max_epoch': 200,
        'no_aug_epoch': 15,
        'batch_size': 32,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.01,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
    },
    
    'ccdet_x': {
        # backbone
        'backbone': 'yolox_backbone',
        'pretrained': True,
        'bk_act': 'silu',
        'stride': 4,
        'depthwise': False,
        'width': 1.25,
        'depth': 1.33,
        # neck
        'neck_name': 'DE',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        # fpn
        'fpn_name': 'basicfpn',
        'fpn_dims': [64, 128, 256, 512],
        'fpn_idx': ['layer1', 'layer2', 'layer3', 'layer4'],
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        # head
        'head': 'decoupled_head',
        'head_k': 3,
        'head_dim': 64,
        'head_act': 'silu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 1,
        'num_reg_layers': 1,
        # post-process
        'nms_kernel': 5,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'test_topk': 100,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_iou_weight': 1.0,
        # training configuration
        'max_epoch': 200,
        'no_aug_epoch': 15,
        'batch_size': 16,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.01,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
    },
    
    'ccdet_t': {
        # backbone
        'backbone': 'yolox_backbone',
        'pretrained': True,
        'bk_act': 'silu',
        'stride': 8,
        'depthwise': False,
        'width': 0.375,
        'depth': 0.33,
        # neck
        'neck_name': 'DE',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        # fpn
        'fpn_name': 'basicfpn',
        'fpn_dims': [64, 128, 256],
        'fpn_idx': ['layer2', 'layer3', 'layer4'],
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        # head
        'head': 'decoupled_head',
        'head_k': 3,
        'head_dim': 64,
        'head_act': 'silu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 1,
        'num_reg_layers': 1,
        # post-process
        'nms_kernel': 5,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'test_topk': 100,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_iou_weight': 1.0,
        # training configuration
        'max_epoch': 200,
        'no_aug_epoch': 15,
        'batch_size': 32,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.01,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
    },
    
    'ccdet_n': {
        # backbone
        'backbone': 'yolox_backbone',
        'pretrained': True,
        'bk_act': 'silu',
        'stride': 8,
        'depthwise': True,
        'width': 0.25,
        'depth': 0.33,
        # neck
        'neck_name': None,
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'silu',
        'neck_norm': 'BN',
        # fpn
        'fpn_name': 'basicfpn',
        'fpn_dims': [48, 96, 192],
        'fpn_idx': ['layer2', 'layer3', 'layer4'],
        'fpn_act': 'silu',
        'fpn_norm': 'BN',
        'fpn_dw': True,
        # head
        'head': 'decoupled_head',
        'head_k': 3,
        'head_dim': 48,
        'head_act': 'silu',
        'head_norm': 'BN',
        'head_dw': True,
        'num_cls_layers': 1,
        'num_reg_layers': 1,
        # post-process
        'nms_kernel': 5,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'test_topk': 100,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_iou_weight': 1.0,
        # training configuration
        'max_epoch': 200,
        'no_aug_epoch': 15,
        'batch_size': 64,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.01,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
    },
    
    # CC-Det-E
    'ccdet_e_r18': {
        # backbone
        'backbone': 'resnet18',
        'pretrained': True,
        'stride': 8,
        'depthwise': False,
        # neck
        'neck_name': 'DE',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        # fpn
        'fpn_name': 'basicfpn',
        'fpn_dims': [128, 256, 256],
        'fpn_idx': ['layer2', 'layer3', 'layer4'],
        'fpn_act': 'relu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        # head
        'head': 'decoupled_head',
        'head_k': 3,
        'head_dim': 64,
        'head_act': 'relu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 1,
        'num_reg_layers': 1,
        # post-process
        'nms_kernel': 5,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'test_topk': 100,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_iou_weight': 1.0,
        # training configuration
        'max_epoch': 200,
        'no_aug_epoch': 15,
        'batch_size': 32,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.01,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
    },
    
    'ccdet_e_r50': {
        # backbone
        'backbone': 'resnet50',
        'pretrained': True,
        'stride': 8,
        'depthwise': False,
        # neck
        'neck_name': 'DE',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        # fpn
        'fpn_name': 'basicfpn',
        'fpn_dims': [128, 256, 256],
        'fpn_idx': ['layer2', 'layer3', 'layer4'],
        'fpn_act': 'relu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        # head
        'head': 'decoupled_head',
        'head_k': 3,
        'head_dim': 64,
        'head_act': 'relu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 1,
        'num_reg_layers': 1,
        # post-process
        'nms_kernel': 5,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'test_topk': 100,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 1.0,
        'loss_iou_weight': 1.0,
        # training configuration
        'max_epoch': 200,
        'no_aug_epoch': 15,
        'batch_size': 32,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.01,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
    }
}