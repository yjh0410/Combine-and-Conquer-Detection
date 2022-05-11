import torch
import torch.nn as nn
from .conv import Conv


class UpSample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(UpSample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corner = align_corner

    def forward(self, x):
        return torch.nn.functional.interpolate(
            input=x, size=self.size, scale_factor=self.scale_factor, 
            mode=self.mode, align_corners=self.align_corner
            )


# Upsample layer with interpolate operation and 1x1 conv layer
class ResizeConv(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='relu', norm_type='BN', 
                 size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(ResizeConv, self).__init__()
        self.upsample = UpSample(
            size=size, scale_factor=scale_factor, 
            mode=mode, align_corner=align_corner
            )
        self.conv = Conv(
            in_dim, out_dim, k=1, 
            act_type=act_type, norm_type=norm_type
            )

    def forward(self, x):
        x = self.conv(self.upsample(x))
        return x

