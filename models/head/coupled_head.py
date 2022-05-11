import torch.nn as nn

from ..basic.conv import Conv


class CoupledHead(nn.Module):
    def __init__(self,
                 cfg,
                 in_dim=1024,
                 out_dim=256):
        super().__init__()
        head_dim = head_dim

        # input project
        self.input_proj = Conv(
            in_dim, out_dim, k=1, 
            act_type=cfg['head_act'], 
            norm_type=cfg['head_norm'])

        # head
        self.feat = nn.Sequential(
                    *[Conv(head_dim, head_dim, 
                           k=cfg['head_k'], p=cfg['head_k']//2, 
                           act_type=cfg['head_act'], 
                           norm_type=cfg['head_norm'],
                           depthwise=cfg['head_dw']) for _ in range(cfg['num_head_layers'])]
                           )


    def forward(self, x):
        feat = self.input_proj(x)
        feat = self.feat(feat)

        return feat, feat
