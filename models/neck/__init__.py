from .fpn import BasicFPN, PaFPN
from .spp import SPP
from .dilated_encoder import DilateEncoder


# build feature pyramid network
def build_fpn(fpn_name,
              fpn_cfg, 
              in_dims=[128, 256, 512, 1024], 
              out_dims=[64, 128, 256, 512]):
    print('==============================')
    print('FPN: {}'.format(fpn_name.upper()))

    if fpn_name == 'basicfpn':
        return BasicFPN(
            in_dims=in_dims, out_dims=out_dims, 
            act_type=fpn_cfg['fpn_act'], 
            norm_type=fpn_cfg['fpn_norm'], 
            depthwise=fpn_cfg['fpn_dw']
            )
    elif fpn_name == 'pafpn':
        return PaFPN(
            in_dims=in_dims,
            out_dims=out_dims,
            depth=fpn_cfg['depth'],
            act_type=fpn_cfg['fpn_act'], 
            norm_type=fpn_cfg['fpn_norm'], 
            depthwise=fpn_cfg['fpn_dw']
            )
            
  
    else:
        print("Unknown FPN version ...")
        exit()


def build_neck(neck_name, neck_cfg, in_dim, out_dim):
    print('==============================')
    print('Neck: {}'.format(neck_name))
    # build neck
    if neck_name == 'spp':
        neck = SPP(
            in_dim=in_dim,
            out_dim=out_dim,
            expand_ratio=neck_cfg['expand_ratio'],
            kernel_sizes=neck_cfg['kernel_sizes'],
            act_type=neck_cfg['neck_act'],
            norm_type=neck_cfg['neck_norm']
            )
    elif neck_name == 'dilated_encoder':
        neck = DilateEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            expand_ratio=neck_cfg['expand_ratio'],
            dilations=neck_cfg['dilations'],
            act_type=neck_cfg['neck_act'],
            norm_type=neck_cfg['neck_norm']
            )

    return neck
