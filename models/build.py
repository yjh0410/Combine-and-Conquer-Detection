import torch

from .detector.ccdet import CCDet


def build_model(model_cfg, 
                device, 
                img_size, 
                num_classes, 
                is_train=False,
                coco_pretrained=None,
                resume=None,
                eval_mode=False):
    # topk candidate number
    if is_train:
        topk = model_cfg['train_topk']
    else:
        if eval_mode:
            topk = model_cfg['eval_topk']
        else:
            topk = model_cfg['inference_topk']

    # build CC-Det    
    model = CCDet(
        cfg=model_cfg,
        device=device,
        img_size=img_size,
        num_classes=num_classes,
        topk=topk,
        conf_thresh=model_cfg['conf_thresh'],
        nms_thresh=model_cfg['nms_thresh'],
        trainable=is_train) 

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

    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)
                        
    return model
