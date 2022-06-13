from .decoupled_head import DecoupledHead
from .coupled_head import CoupledHead


# Build detection head
def build_head(head_name, head_cfg, in_dim, out_dim):
    print('==============================')
    print('Head: {}'.format(head_name.upper()))
    
    if head_name == 'decoupled_head':
        head = DecoupledHead(
            cfg=head_cfg,
            in_dim=in_dim,
            out_dim=out_dim,
        )

    elif head_name == 'coupled_head':
        head = CoupledHead(
            cfg=head_cfg,
            in_dim=in_dim,
            out_dim=out_dim,
        )

    return head
