import torch.nn as nn

from ..basic.conv import Conv


class DecoupledHead(nn.Module):
    def __init__(self,
                 cfg,
                 in_dim=1024,
                 out_dim=256):
        super().__init__()

        # input project
        self.input_proj_cls = Conv(
            in_dim, out_dim, k=1, 
            act_type=cfg['head_act'], 
            norm_type=cfg['head_norm'])
        self.input_proj_reg = Conv(
            in_dim, out_dim, k=1, 
            act_type=cfg['head_act'], 
            norm_type=cfg['head_norm'])

        # head
        self.cls_feat = nn.Sequential(
                    *[Conv(out_dim, out_dim, 
                           k=cfg['head_k'], p=cfg['head_k']//2, 
                           act_type=cfg['head_act'], 
                           norm_type=cfg['head_norm'],
                           depthwise=cfg['head_dw']) for _ in range(cfg['num_cls_layers'])]
                           )
        self.reg_feat = nn.Sequential(
                    *[Conv(out_dim, out_dim, 
                           k=cfg['head_k'], p=cfg['head_k']//2, 
                           act_type=cfg['head_act'], 
                           norm_type=cfg['head_norm'],
                           depthwise=cfg['head_dw']) for _ in range(cfg['num_reg_layers'])]
                           )


    def forward(self, x):
        cls_feat = self.input_proj_cls(x)
        reg_feat = self.input_proj_reg(x)

        cls_feat = self.cls_feat(cls_feat)
        reg_feat = self.reg_feat(reg_feat)

        return cls_feat, reg_feat
