import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basic.conv import Conv


# Spatial Pyramid Pooling
class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self, 
                 in_dim, out_dim, 
                 expand_ratio=0.5,
                 kernel_sizes = [5, 9, 13],
                 act_type='relu', 
                 norm_type='BN'):
        super(SPP, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.kernel_sizes = kernel_sizes
        self.cv1 = Conv(
            in_dim, inter_dim, k=1, 
            act_type=act_type, norm_type=norm_type
            )
        self.cv2 = Conv(
            inter_dim*(len(kernel_sizes) + 1), out_dim, k=1, 
            act_type=act_type, norm_type=norm_type
            )

    def forward(self, x):
        x = self.cv1(x)
        outputs = [x]
        for k in self.kernel_sizes:
            outputs.append(
                F.max_pool2d(x, k, stride=1, padding=k//2)
            )
        x = torch.cat(outputs, dim=1)
        x = self.cv2(x)

        return x

