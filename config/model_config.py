# Model Configuration


m_config = {
    # CC-Det
    # backbone
    'stride': 8,
    'backbone': {'resnet18': {'pretrained': True},
                 'resnet50': {'pretrained': True},
                 'resnet101': {'pretrained': True},
                 'cspdarknet53': {'pretrained': True}
                                },
    # neck
    'neck': {'dilated_encoder': {'dilations': [2, 4, 6, 8],
                                 'expand_ratio': 0.5,
                                 'neck_act': 'relu',
                                 'neck_norm': 'BN'},
             'spp': {'kernel_sizes': [5, 9, 13],
                     'expand_ratio': 0.5,
                     'neck_act': 'relu',
                     'neck_norm': 'BN'},
                     },
    # feature aggregation
    'feat_aggr':{'basicfpn': {'fpn_dims': [128, 256, 512],
                                'fpn_idx': ['layer2', 'layer3', 'layer4'],
                                'fpn_act': 'relu',
                                'fpn_norm': 'BN',
                                'fpn_dw': False},
                 'yolopafpn': {'fpn_dims': [128, 256, 512],
                               'fpn_idx': ['layer2', 'layer3', 'layer4'],
                               'fpn_act': 'lrelu',
                               'fpn_norm': 'BN',
                               'fpn_dw': False,
                               'depth': 3},
                               },
    # head
    'head': {'decoupled_head': {'head_k': 3,
                                'head_dim': 128,
                                'head_act': 'relu',
                                'head_norm': 'BN',
                                'head_dw': False,
                                'num_cls_layers': 2,
                                'num_reg_layers': 2},
                                },
    # post-process
    'use_nms': True,
    'conf_thresh': 0.01,
    'nms_thresh': 0.5,
    'train_topk': 1000,
    'inference_topk': 100,
    'eval_topk': 1000,
    # loss
    'loss_hmp_weight': 1.0,
    'loss_reg_weight': 5.0,
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