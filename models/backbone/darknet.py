import torch
import torch.nn as nn
import numpy as np
import os


model_urls = {
    "darknet_19": "https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/darknet19.pth",
    "darknet_53": "https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/darknet53.pth",
}


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_dim, out_dim, k, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, k, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class ResBlock(nn.Module):
    def __init__(self, in_dim, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(in_dim, in_dim//2, 1),
                Conv_BN_LeakyReLU(in_dim//2, in_dim, 3, padding=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x


class DarkNet_19(nn.Module):
    def __init__(self):
        print("Initializing the darknet19 network ......")
        
        super(DarkNet_19, self).__init__()
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(in_dim=3, out_dim=32, k=3, padding=1),
            nn.MaxPool2d((2,2), 2),
        )

        # output : stride = 2, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(in_dim=32, out_dim=64, k=3, padding=1)
        )

        # output : stride = 4, c = 128
        self.maxpool_2 = nn.MaxPool2d((2, 2), stride=2)
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(in_dim=64, out_dim=128, k=3, padding=1),
            Conv_BN_LeakyReLU(in_dim=128, out_dim=64, k=1),
            Conv_BN_LeakyReLU(in_dim=64, out_dim=128, k=3, padding=1)
        )

        # output : stride = 8, c = 256
        self.maxpool_3 = nn.MaxPool2d((2,2), 2)
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(in_dim=128, out_dim=256, k=3, padding=1),
            Conv_BN_LeakyReLU(in_dim=256, out_dim=128, k=1),
            Conv_BN_LeakyReLU(in_dim=128, out_dim=256, k=3, padding=1)
        )

        # output : stride = 16, c = 512
        self.maxpool_4 = nn.MaxPool2d((2, 2), stride=2)
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(in_dim=256, out_dim=512, k=3, padding=1),
            Conv_BN_LeakyReLU(in_dim=512, out_dim=256, k=1),
            Conv_BN_LeakyReLU(in_dim=256, out_dim=512, k=3, padding=1),
            Conv_BN_LeakyReLU(in_dim=512, out_dim=256, k=1),
            Conv_BN_LeakyReLU(in_dim=256, out_dim=512, k=3, padding=1)
        )
        
        # output : stride = 32, c = 1024
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(in_dim=512, out_dim=1024, k=3, padding=1),
            Conv_BN_LeakyReLU(in_dim=1024, out_dim=512, k=1),
            Conv_BN_LeakyReLU(in_dim=512, out_dim=1024, k=3, padding=1),
            Conv_BN_LeakyReLU(in_dim=1024, out_dim=512, k=1),
            Conv_BN_LeakyReLU(in_dim=512, out_dim=1024, k=3, padding=1)
        )


    def forward(self, x):
        c1 = self.conv_1(x)
        c1 = self.conv_2(c1)
        c2 = self.conv_3(self.maxpool_2(c1))
        c3 = self.conv_4(self.maxpool_2(c2))
        c4 = self.conv_5(self.maxpool_4(c3))
        c5 = self.conv_6(self.maxpool_5(c4))

        outputs = {
            'layer1': c2,
            'layer2': c3,
            'layer3': c4,
            'layer4': c5
        }
        return outputs


class DarkNet_53(nn.Module):
    """
    DarkNet-53.
    """
    def __init__(self):
        super(DarkNet_53, self).__init__()
        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU(in_dim=3, out_dim=32, k=3, padding=1),
            Conv_BN_LeakyReLU(in_dim=32, out_dim=64, k=3, padding=1, stride=2),
            ResBlock(in_dim=64, nblocks=1)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(in_dim=64, out_dim=128, k=3, padding=1, stride=2),
            ResBlock(in_dim=128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(in_dim=128, out_dim=256, k=3, padding=1, stride=2),
            ResBlock(in_dim=256, nblocks=8)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(in_dim=256, out_dim=512, k=3, padding=1, stride=2),
            ResBlock(in_dim=512, nblocks=8)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(in_dim=512, out_dim=1024, k=3, padding=1, stride=2),
            ResBlock(in_dim=1024, nblocks=4)
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = {
            'layer1': c2,
            'layer2': c3,
            'layer3': c4,
            'layer4': c5
        }
        return outputs


# Build DarkNet
def build_darknet(model_name='darknet_19', pretrained=False):
    # build backbone
    if model_name == 'darknet_19':
        backbone = DarkNet_19()
        feat_dims = [128, 256, 512, 1024]
    elif model_name == 'darknet_53':
        backbone = DarkNet_53()
        feat_dims = [128, 256, 512, 1024]

    # load weight
    if pretrained:
        print('Loading pretrained {} ...'.format(model_name))
        url = model_urls[model_name]
        checkpoint_state_dict = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)

        # model state dict
        model_state_dict = backbone.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
                    print(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        backbone.load_state_dict(checkpoint_state_dict)

    return backbone, feat_dims


if __name__ == '__main__':
    import time
    model, feat_dims = build_darknet(model_name='darknet_53', pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for k in outputs.keys():
        print(outputs[k].shape)
