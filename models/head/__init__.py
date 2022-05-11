from .decoupled_head import DecoupledHead
from .coupled_head import CoupledHead


# Build detection head
def build_head(cfg, in_dim, out_dim):
    head_name = cfg['head']
    print('==============================')
    print('Head: {}'.format(cfg['head'].upper()))
    
    if head_name == 'decoupled_head':
        head = DecoupledHead(
            cfg=cfg,
            in_dim=in_dim,
            out_dim=out_dim,
        )

    elif head_name == 'coupled_head':
        head = CoupledHead(
            cfg=cfg,
            in_dim=in_dim,
            out_dim=out_dim,
        )

    return head
