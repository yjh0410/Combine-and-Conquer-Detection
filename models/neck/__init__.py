from .fpn import BasicFPN


# build feature pyramid network
def build_fpn(cfg=None, 
              in_dims=[128, 256, 512, 1024], 
              out_dims=[64, 128, 256, 512]):
    print('==============================')
    print('FPN: {}'.format(cfg['fpn_name'].upper()))

    if cfg['fpn_name'] == 'basicfpn':
        return BasicFPN(
            in_dims=in_dims, out_dims=out_dims, 
            act_type=cfg['fpn_act'], 
            norm_type=cfg['fpn_norm'], 
            depthwise=cfg['fpn_dw'], 
            extra_module_cfg=cfg)
            
    else:
        print("Unknown FPN version ...")
        exit()


