import torch

from .detector.ccdet import CCDet


def build_model(cfg, 
                device, 
                img_size, 
                num_classes, 
                is_train=False,
                coco_pretrained=None,
                use_nms=False):
    # build CC-Det    
    model = CCDet(
        cfg=cfg,
        device=device,
        img_size=img_size,
        num_classes=num_classes,
        topk=cfg['train_topk'] if is_train else cfg['test_topk'],
        nms_thresh=cfg['nms_thresh'],
        trainable=is_train,
        use_nms=use_nms) 

    # Load COCO pretrained weight
    if coco_pretrained is not None:
        print('Loading COCO pretrained weight ...')
        checkpoint = torch.load(coco_pretrained, map_location='cpu')
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
                print(k)

        model.load_state_dict(checkpoint_state_dict, strict=False)

    return model
