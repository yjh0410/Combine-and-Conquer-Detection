from .darknet import build_darknet
from .resnet import build_resnet
from .vggnet import build_vgg
from .yolox_backbone import build_cspdarknet


# Build Backbone
def build_backbone(cfg, model_name='resnet18', pretrained=False):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))

    # backbone
    if 'resnet' in model_name:
        backbone, bk_feats = build_resnet(
            model_name=model_name, pretrained=pretrained)
    
    elif 'darknet' in model_name:
        backbone, bk_feats = build_darknet(
            model_name=model_name, pretrained=pretrained)

    elif 'cspdarknet' in model_name:
        backbone, bk_feats = build_cspdarknet(
            depth=cfg['depth'],
            width=cfg['width'],
            depthwise=cfg['depthwise'],
            act_type=cfg['bk_act'],
            pretrained=pretrained)

    elif 'vgg' in model_name:
        backbone, bk_feats = build_vgg(
            model_name=model_name, pretrained=pretrained)

    else:
        print("Unknown Backbone !!")
        exit()

    return backbone, bk_feats
