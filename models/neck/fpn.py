import torch.nn as nn
from ..basic.conv import Conv
from ..basic.upsample import ResizeConv
from .spp import SPP
from .dilated_encoder import DilateEncoder


# Basic FPN
class BasicFPN(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 1024, 2048],  # [C2, C3, C4, C5]
                 out_dims=[64, 128, 256, 512],    # [P2, P3, P4, P5]
                 act_type='relu',
                 norm_type='BN',
                 depthwise=False,
                 extra_module_cfg=None):
        super(BasicFPN, self).__init__()
        self.proj_layers = nn.ModuleList()
        self.smooth_layers = nn.ModuleList()
        self.deconv_layers = nn.ModuleList()

        print(in_dims, out_dims)
        assert len(in_dims) == len(out_dims)

        # input project layers
        for i in range(len(in_dims)):
            # input project
            if i < len(in_dims) - 1:
                self.proj_layers.append(
                    Conv(in_dims[i], out_dims[i], k=1, 
                        act_type=None, norm_type=norm_type)
                        )
            # extra module for C5
            else:
                print('==============================')
                print('Extra Module: {}'.format(extra_module_cfg['neck_name'].upper()))
                
                if extra_module_cfg['neck_name'] == 'SPP':
                    self.proj_layers.append(
                        SPP(in_dims[i], out_dims[i],
                            expand_ratio=extra_module_cfg['expand_ratio'], 
                            kernel_sizes=extra_module_cfg['kernel_sizes'],
                            act_type=extra_module_cfg['neck_act'],
                            norm_type=extra_module_cfg['neck_norm']
                            )
                    )
                elif extra_module_cfg['neck_name'] == 'DE':
                    self.proj_layers.append(
                        DilateEncoder(
                            in_dims[i], out_dims[i],
                            expand_ratio=extra_module_cfg['expand_ratio'],
                            dilations=extra_module_cfg['dilations'],
                            act_type=extra_module_cfg['neck_act'],
                            norm_type=extra_module_cfg['neck_norm']
                            )
                    )
                else:
                    self.proj_layers.append(
                        Conv(in_dims[i], out_dims[i], k=1, 
                            act_type=None, norm_type=norm_type)
                            )

        # upsample layers
        out_dims_ = out_dims[::-1]
        for i in range(1, len(out_dims)):
            # deconv layer
            self.deconv_layers.append(
                ResizeConv(out_dims_[i-1], out_dims_[i], 
                           act_type=act_type, norm_type=norm_type,
                           scale_factor=2, mode='nearest'))

        # smooth layers
        for i in range(len(out_dims)):
            # input project
            self.smooth_layers.append(
                Conv(out_dims[i], out_dims[i], k=3, p=1,
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

        for feat, deconv in zip(inputs[1:], self.deconv_layers):
            top_down_feat = deconv(prev_feat)
            prev_feat = feat + top_down_feat
            pymaid_feats.insert(0, prev_feat)

        # smooth layers
        outputs = []
        for feat, layer in zip(pymaid_feats, self.smooth_layers):
            outputs.append(layer(feat))

        return outputs
            