import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic.conv import Conv
from ..basic.upsample import ResizeConv
from ..basic.bottleneck_csp import BottleneckCSP


# Basic FPN
class BasicFPN(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 1024, 2048],  # [C2, C3, C4, C5]
                 out_dims=[64, 128, 256, 512],    # [P2, P3, P4, P5]
                 act_type='relu',
                 norm_type='BN',
                 depthwise=False):
        super(BasicFPN, self).__init__()
        self.proj_layers = nn.ModuleList()
        self.smooth_layers = nn.ModuleList()
        self.deconv_layers = nn.ModuleList()

        assert len(in_dims) == len(out_dims)

        # input project layers
        for i in range(len(in_dims)):
            # input project
            self.proj_layers.append(
                Conv(in_dims[i], out_dims[i], k=1, 
                    act_type=None, norm_type=norm_type)
                    )

        # upsample layers
        out_dims_ = out_dims[::-1]
        for i in range(1, len(out_dims)):
            # deconv layer
            self.deconv_layers.append(
                ResizeConv(
                    out_dims_[i-1], out_dims_[i],
                    act_type=act_type,
                    norm_type=norm_type,
                    scale_factor=2,
                    mode='nearest')
                    )

        # smooth layers
        for i in range(1, len(out_dims)):
            # input project
            self.smooth_layers.append(
                Conv(out_dims_[i], out_dims_[i], k=3, p=1,
                    act_type=None, norm_type=norm_type, 
                    depthwise=depthwise)
                    )


    def forward(self, features):
        """
            features: List(Tensor)[..., C3, C4, C5, ...]
        """
        # input project
        inputs = []
        for feat, layer in zip(features, self.proj_layers):
            inputs.append(layer(feat))

        # feature pyramid
        pymaid_feats = []
        # [..., C4, C5] -> [C5, C4, ...]
        inputs = inputs[::-1]
        top_level_feat = inputs[0]
        prev_feat = top_level_feat
        pymaid_feats.append(prev_feat)

        for feat, deconv, smooth in zip(inputs[1:], self.deconv_layers, self.smooth_layers):
            top_down_feat = deconv(prev_feat)
            prev_feat = smooth(feat + top_down_feat)
            pymaid_feats.insert(0, prev_feat)

        return pymaid_feats


# YoloPaFPN
class YoloPaFPN(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 1024], # [c3, c4, c5]
                 out_dims=[128, 256, 512], # [p3, p4, p5]
                 depth=3, 
                 norm_type='BN',
                 act_type='relu',
                 depthwise=False):
        super(YoloPaFPN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        nblocks = int(depth)

        self.proj_layers = nn.ModuleList()

        # input project layers
        for i in range(len(in_dims)):
            # input project
            self.proj_layers.append(
                Conv(in_dims[i], out_dims[i], k=1, 
                    act_type=None, norm_type=norm_type)
                    )

        self.head_conv_0 = Conv(out_dims[-1], out_dims[-1]//2, k=1,
                                norm_type=norm_type, act_type=act_type)  # 10
        self.head_csp_0 = BottleneckCSP(out_dims[-2] + out_dims[-1]//2,
                                        out_dims[-2], n=nblocks,
                                        shortcut=False, depthwise=depthwise,
                                        norm_type=norm_type, act_type=act_type)

        # P3/8-small
        self.head_conv_1 = Conv(out_dims[-2], out_dims[-2]//2, k=1,
                                norm_type=norm_type, act_type=act_type)  # 14
        self.head_csp_1 = BottleneckCSP(out_dims[-3] + out_dims[-2]//2, out_dims[-3], n=nblocks,
                                        shortcut=False, depthwise=depthwise,
                                        norm_type=norm_type, act_type=act_type)

        # P4/16-medium
        self.head_conv_2 = Conv(out_dims[-3], out_dims[-3], k=3, p=1, s=2,
                                depthwise=depthwise, norm_type=norm_type, act_type=act_type)
        self.head_csp_2 = BottleneckCSP(out_dims[-3] + out_dims[-2]//2, out_dims[-2], n=nblocks,
                                        shortcut=False, depthwise=depthwise,
                                        norm_type=norm_type, act_type=act_type)

        # P8/32-large
        self.head_conv_3 = Conv(out_dims[-2], out_dims[-2], k=3, p=1, s=2,
                                depthwise=depthwise, norm_type=norm_type, act_type=act_type)
        self.head_csp_3 = BottleneckCSP(out_dims[-2] + out_dims[-1]//2, out_dims[-1], n=nblocks,
                                        shortcut=False, depthwise=depthwise,
                                        norm_type=norm_type, act_type=act_type)

        # top-down
        self.top_down_fpn = BasicFPN(
            out_dims, out_dims,
            act_type=act_type,
            norm_type=norm_type,
            depthwise=depthwise
            )


    def forward(self, features):
        # input project
        inputs = []
        for feat, layer in zip(features, self.proj_layers):
            inputs.append(layer(feat))

        c3, c4, c5 = inputs

        c6 = self.head_conv_0(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)   # s32->s16
        c8 = torch.cat([c7, c4], dim=1)
        c9 = self.head_csp_0(c8)
        # P3/8
        c10 = self.head_conv_1(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)   # s16->s8
        c12 = torch.cat([c11, c3], dim=1)
        c13 = self.head_csp_1(c12)  # to det
        # p4/16
        c14 = self.head_conv_2(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.head_csp_2(c15)  # to det
        # p5/32
        c17 = self.head_conv_3(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.head_csp_3(c18)  # to det

        out_feats = [c13, c16, c19] # [P3, P4, P5]
        out_feats = self.top_down_fpn(out_feats)

        return out_feats
