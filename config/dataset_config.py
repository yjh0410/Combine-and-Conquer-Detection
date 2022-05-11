# Dataset Configuration

d_config = {
    'voc':{
        'data_root': '/mnt/share/ssd2/dataset/VOCdevkit',
        # 'data_root': 'D:\\python_work\\object-detection\\dataset\\VOCdevkit',
        'num_classes': 20,
        'train_size': 640,
        'test_size': 640,
        'random_size': [320, 352, 384, 416,
                        448, 480, 512, 544,
                        576, 608, 640],
        'pixel_mean': [123.675, 116.28, 103.53],
        'pixel_std': [58.395, 57.12, 57.375],
        'format': 'RGB',
        'mosaic_prob': 0.5,
        'mixup_prob': 0.5,
        'transforms': [{'name': 'DistortTransform',
                        'hue': 0.1,
                        'saturation': 1.5,
                        'exposure': 1.5},
                        {'name': 'RandomHorizontalFlip'},
                        {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                        {'name': 'ToTensor'},
                        {'name': 'Resize'},
                        {'name': 'Normalize'},
                        {'name': 'PadImage'}],
    },
    'coco':{
        'data_root': None,
        'num_classes': 80,
    },
    'widerface':{
        'data_root': None,
        'num_classes': 1,
    },
    'crowdhuman':{
        'data_root': None,
        'num_classes': 1,
    }
}