from .resnet import build_resnet


# Build Backbone
def build_backbone(bk_name, pretrained=False):
    print('==============================')
    print('Backbone: {}'.format(bk_name.upper()))

    # backbone
    if 'resnet' in bk_name:
        backbone, bk_feats = build_resnet(
            model_name=bk_name, pretrained=pretrained)
    
    else:
        print("Unknown Backbone !!")
        exit()

    return backbone, bk_feats
