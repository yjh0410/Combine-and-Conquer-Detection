from .resnet import build_resnet
from .cspdarknet import build_cspdarknet


# Build Backbone
def build_backbone(bk_name, bk_cfg):
    print('==============================')
    print('Backbone: {}'.format(bk_name.upper()))

    # backbone
    if 'resnet' in bk_name:
        backbone, bk_feats = build_resnet(
            model_name=bk_name, pretrained=bk_cfg['pretrained'])
    
    elif bk_name == 'cspdarknet53':
        backbone, bk_feats = build_cspdarknet(pretrained=bk_cfg['pretrained'])

    else:
        print("Unknown Backbone !!")
        exit()

    return backbone, bk_feats
