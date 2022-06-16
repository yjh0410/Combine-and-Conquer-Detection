# Model Configuration


m_config = {
    # CC-Det
    'ccdet_r18_fpn': {
        # Backbone
        'backbone': 'resnet18',
        'pretrained': True,
        'stride': 8,
        # Neck
        'neck_name': 'dilated_encoder',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        # Feat Aggregation
        'fpn_name': 'basicfpn',
        'fpn_dims': [128, 256, 512],
        'fpn_idx': ['layer2', 'layer3', 'layer4'],
        'fpn_act': 'relu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        # Detection Head
        'head_name': 'decoupled_head',
        'head_k': 3,
        'head_dim': 128,
        'head_act': 'relu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 4,
        'num_reg_layers': 4,
        # post-process
        'use_nms': True,
        'conf_thresh': 0.01,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'inference_topk': 100,
        'eval_topk': 1000,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 2.0,
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
    
    'ccdet_r18': {
        # Backbone
        'backbone': 'resnet18',
        'pretrained': True,
        'stride': 8,
        # Neck
        'neck_name': 'dilated_encoder',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        # Feat Aggregation
        'fpn_name': 'yolopafpn',
        'fpn_dims': [128, 256, 512],
        'fpn_idx': ['layer2', 'layer3', 'layer4'],
        'fpn_act': 'relu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        'depth': 1,
        # Detection Head
        'head_name': 'decoupled_head',
        'head_k': 3,
        'head_dim': 128,
        'head_act': 'relu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 4,
        'num_reg_layers': 4,
        # post-process
        'use_nms': True,
        'conf_thresh': 0.01,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'inference_topk': 100,
        'eval_topk': 1000,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 2.0,
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
        # Backbone
        'backbone': 'resnet50',
        'pretrained': True,
        'stride': 8,
        # Neck
        'neck_name': 'dilated_encoder',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        # Feat Aggregation
        'fpn_name': 'yolopafpn',
        'fpn_dims': [128, 256, 512],
        'fpn_idx': ['layer2', 'layer3', 'layer4'],
        'fpn_act': 'relu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        'depth': 1,
        # Detection Head
        'head_name': 'decoupled_head',
        'head_k': 3,
        'head_dim': 128,
        'head_act': 'relu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 4,
        'num_reg_layers': 4,
        # post-process
        'use_nms': True,
        'conf_thresh': 0.01,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'inference_topk': 100,
        'eval_topk': 1000,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 2.0,
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
        # Backbone
        'backbone': 'resnet101',
        'pretrained': True,
        'stride': 8,
        # Neck
        'neck_name': 'dilated_encoder',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        # Feat Aggregation
        'fpn_name': 'yolopafpn',
        'fpn_dims': [128, 256, 512],
        'fpn_idx': ['layer2', 'layer3', 'layer4'],
        'fpn_act': 'relu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        'depth': 3,
        # Detection Head
        'head_name': 'decoupled_head',
        'head_k': 3,
        'head_dim': 128,
        'head_act': 'relu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 4,
        'num_reg_layers': 4,
        # post-process
        'use_nms': True,
        'conf_thresh': 0.01,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'inference_topk': 100,
        'eval_topk': 1000,
        # loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 2.0,
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

    'ccdet_cd53': {
        # Backbone
        'backbone': 'cspdarknet53',
        'pretrained': True,
        'stride': 8,
        # Neck
        'neck_name': 'dilated_encoder',
        'dilations': [2, 4, 6, 8],
        'expand_ratio': 0.5,
        'neck_act': 'relu',
        'neck_norm': 'BN',
        # Feat Aggregation
        'fpn_name': 'yolopafpn',
        'fpn_dims': [128, 256, 512],
        'fpn_idx': ['layer2', 'layer3', 'layer4'],
        'fpn_act': 'relu',
        'fpn_norm': 'BN',
        'fpn_dw': False,
        'depth': 1,
        # Detection Head
        'head_name': 'decoupled_head',
        'head_k': 3,
        'head_dim': 128,
        'head_act': 'relu',
        'head_norm': 'BN',
        'head_dw': False,
        'num_cls_layers': 4,
        'num_reg_layers': 4,
        # Post-process
        'use_nms': True,
        'conf_thresh': 0.01,
        'nms_thresh': 0.5,
        'train_topk': 1000,
        'inference_topk': 100,
        'eval_topk': 1000,
        # Loss
        'loss_hmp_weight': 1.0,
        'loss_reg_weight': 2.0,
        'loss_iou_weight': 1.0,
        # Training configuration
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

}