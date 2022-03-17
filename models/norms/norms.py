
import torch
import torch.nn as nn

from models.norms.graph_norm import GraphNorm
from models.norms.local_bn1d import LocalBN1d
from models.norms.instance_norm import InstanceNorm
from models.norms.xxx_norm import XXX_Norm
from models.norms.xxx_norm2 import XXX_Norm2
from models.norms.xxx_norm3 import XXX_Norm3

class Norms(nn.Module):

    def __init__(self, norm_type = 'bn', embed_dim=300, print_info=None):
        super(Norms, self).__init__()
        assert norm_type in ['bn', 'gn', 'in', 'xn', 'xn2', 'xn3', 'None']
        self.norm_type = norm_type
        self.norm = None
        if norm_type == 'bn':
            self.norm = LocalBN1d(embed_dim)
        elif norm_type == 'gn':
            self.norm = GraphNorm(embed_dim)
        elif norm_type == 'in':
            self.norm = InstanceNorm(embed_dim)
        elif norm_type == 'xn':
            self.norm = XXX_Norm(embed_dim)
        elif norm_type == 'xn2':
            self.norm = XXX_Norm2(embed_dim)      
        elif norm_type == 'xn3':
            self.norm = XXX_Norm3(embed_dim)

    def forward(self, graphs, tensor):

        if self.norm_type == 'None':
            tensor = tensor
        else: 
            tensor = self.norm(graphs, tensor)

        norm_tensor = tensor
        return norm_tensor