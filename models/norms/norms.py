
import torch
import torch.nn as nn

from models.norms.graph_norm import GraphNorm
from models.norms.local_bn1d import LocalBN1d
from models.norms.instance_norm import InstanceNorm
from models.norms.lp_norm import LP_Norm
from models.norms.lp2_norm import LP2_Norm
from models.norms.xxx_norm import XXX_Norm

class Norms(nn.Module):

    def __init__(self, norm_type = 'bn', embed_dim=300, print_info=None):
        super(Norms, self).__init__()
        assert norm_type in ['bn', 'gn', 'in', 'ln', 'ln2', 'xn', 'None']
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
        elif norm_type == 'ln':
            self.norm = nn.ModuleList()
            self.norm.append(LP_Norm(embed_dim))
            self.norm.append(LocalBN1d(embed_dim))    
        elif norm_type == 'ln2':
            self.norm = nn.ModuleList()
            self.norm.append(LP2_Norm(embed_dim))
            self.norm.append(LocalBN1d(embed_dim))        
               

    def forward(self, graphs, tensor):

        if self.norm_type == 'None':
            tensor = tensor
        elif self.norm_type == 'ln':
            tensor = self.norm[0](graphs, tensor)
            tensor = self.norm[1](graphs, tensor)
        elif self.norm_type == 'ln2':
            tensor = self.norm[0](graphs, tensor)
            tensor = self.norm[1](graphs, tensor)
        else: 
            tensor = self.norm(graphs, tensor)

        norm_tensor = tensor
        return norm_tensor