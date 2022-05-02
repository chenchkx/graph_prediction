
import torch
import torch.nn as nn

from models.norm.graph_norm import GraphNorm
from models.norm.local_bn1d import LocalBN1d
from models.norm.local_bn1d_manu import LocalBN1d_Manu
from models.norm.local_bn1d_falseaffine import LocalBN1d_FalseAffine
from models.norm.instance_norm import InstanceNorm
from models.norm.xxx_norm import XXX_Norm
from models.norm.xxx_norm1 import XXX_Norm1
from models.norm.xxx_norm2 import XXX_Norm2
from models.norm.xxx_norm3 import XXX_Norm3
from models.norm.xxx_norm4 import XXX_Norm4
from models.norm.xxx_norm5 import XXX_Norm5
from models.norm.xxx_norm6 import XXX_Norm6
from models.norm.xxx_norm7 import XXX_Norm7
from models.norm.xxx_norm8 import XXX_Norm8
from models.norm.xxx_norm9 import XXX_Norm9
from models.norm.xxx_norm10 import XXX_Norm10

class GNN_Norm(nn.Module):
    def __init__(self, norm_type = 'bn', embed_dim=300, affine=True):
        super(GNN_Norm, self).__init__()

        self.norm_type = norm_type
        self.norm = None
        if norm_type == 'bn':
            self.norm = LocalBN1d(embed_dim, affine=affine)
        elif norm_type == 'gn':
            self.norm = GraphNorm(embed_dim, affine=affine)
        elif norm_type == 'in':
            self.norm = InstanceNorm(embed_dim, affine=affine)
        elif norm_type == 'xn':
            self.norm = XXX_Norm(embed_dim, affine=affine)
        elif norm_type == 'xn1':
            self.norm = XXX_Norm1(embed_dim, affine=affine)
        elif norm_type == 'xn2':
            self.norm = XXX_Norm2(embed_dim, affine=affine)      
        elif norm_type == 'xn3':
            self.norm = XXX_Norm3(embed_dim, affine=affine)
        elif norm_type == 'xn4':
            self.norm = XXX_Norm4(embed_dim, affine=affine)
        elif norm_type == 'xn5':
            self.norm = XXX_Norm5(embed_dim, affine=affine)
        elif norm_type == 'xn6':
            self.norm = XXX_Norm6(embed_dim, affine=affine)
        elif norm_type == 'xn7':
            self.norm = XXX_Norm7(embed_dim)
        elif norm_type == 'xn8':
            self.norm = XXX_Norm8(embed_dim)
        elif norm_type == 'xn9':
            self.norm = XXX_Norm9(embed_dim)    
        elif norm_type == 'xn10':
            self.norm = XXX_Norm10(embed_dim)   

    def forward(self, graphs, tensor):

        if self.norm_type == 'None':
            tensor = tensor
        else: 
            tensor = self.norm(graphs, tensor)

        norm_tensor = tensor
        return norm_tensor