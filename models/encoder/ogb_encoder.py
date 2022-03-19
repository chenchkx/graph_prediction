

import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class OGB_NodeEncoder(nn.Module):
    def __init__(self, ogb_dataset, embed_dim):
        super(OGB_NodeEncoder, self).__init__()

        self.embed_dim = embed_dim
        if 'mol' in ogb_dataset:
            self.node_encoder = AtomEncoder(embed_dim)
        elif 'ppa' in ogb_dataset:
            self.node_encoder = nn.Embedding(1, embed_dim)
    
    def forward(self, tensor):

        return self.node_encoder(tensor)


class OGB_EdgeEncoder(nn.Module):
    def __init__(self, ogb_dataset, embed_dim=300):
        super(OGB_EdgeEncoder, self).__init__()

        self.embed_dim = embed_dim
        if 'mol' in ogb_dataset:
            self.edge_encoder = BondEncoder(embed_dim)
        elif 'ppa' in ogb_dataset:
            self.edge_encoder = nn.Linear(7, embed_dim)
        elif 'code' in ogb_dataset:
            self.edge_encoder = nn.Linear(2, embed_dim)
    def forward(self, tensor):

        return self.edge_encoder(tensor)
