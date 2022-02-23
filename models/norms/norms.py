
import torch
import torch.nn as nn

from models.norms.graph_norm import GraphNorm
from models.norms.local_bn1d import LocalBN1d

class Norms(nn.Module):

    def __init__(self, norm_type = 'bn', embed_dim=300, print_info=None):
        super(Norms, self).__init__()
        assert norm_type in ['bn', 'gn', 'mn', 'None']
        self.norm_type = norm_type
        self.norm = None
        if norm_type == 'bn':
            self.norm = LocalBN1d(embed_dim)
        elif norm_type == 'gn':
            self.norm = GraphNorm(embed_dim)
        elif norm_type == 'mn':
            self.norm = nn.ModuleList()
            self.norm.append(GraphNorm(embed_dim))
            self.norm.append(LocalBN1d(embed_dim))

    def forward(self, graphs, tensor, print_=False):

        if self.norm_type == 'None':
            tensor = tensor
        elif self.norm_type == 'mn':
            tensor = self.norm[0](graphs, tensor)
            tensor = self.norm[1](graphs, tensor)
        else: 
            tensor = self.norm(graphs, tensor)

        norm_tensor = tensor
        return norm_tensor