from .darknet import build_darknet
from .resnet import build_resnet
from .cspdarknet import build_cspdarknet


# Build Backbone
def build_backbone(cfg, model_name='resnet18', pretrained=False):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))

    # backbone
    if 'resnet' in model_name:
        backbone, bk_feats = build_resnet(
            model_name=model_name, pretrained=pretrained)
    
    elif model_name == 'cspdarknet53':
        backbone, bk_feats = build_cspdarknet(pretrained=pretrained)

    elif model_name == 'darknet19' or model_name == 'darknet53':
        backbone, bk_feats = build_darknet(
            model_name=model_name, pretrained=pretrained)

    else:
        print("Unknown Backbone !!")
        exit()

    return backbone, bk_feats
