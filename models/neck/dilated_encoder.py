import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic.conv import Conv


# Standard bottleneck
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_dim, dilation=1, 
                 expand_ratio=0.5, 
                 act_type='relu', norm_type='BN'):
        super(Bottleneck, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.branch = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, 
                 p=dilation, d=dilation, 
                 act_type=act_type, 
                 norm_type=norm_type),
            Conv(inter_dim, in_dim, k=1, act_type=act_type, norm_type=norm_type)
        )

    def forward(self, x):
        return x + self.branch(x)


class DilateEncoder(nn.Module):
    """ DilateEncoder """
    def __init__(self, in_dim, out_dim, 
                 expand_ratio=0.5, 
                 dilations=[2, 4, 6, 8], 
                 act_type='relu',
                 norm_type='BN'):
        super(DilateEncoder, self).__init__()
        self.projector = nn.Sequential(
            Conv(in_dim, out_dim, k=1, act_type=None),
            Conv(out_dim, out_dim, k=3, p=1, act_type=None)
        )
        encoders = [Bottleneck(in_dim=out_dim, dilation=d, 
                               expand_ratio=expand_ratio,
                               act_type=act_type,
                               norm_type=norm_type) for d in dilations]
        self.encoders = nn.Sequential(*encoders)

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)

        return x
