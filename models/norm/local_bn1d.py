
import torch.nn as nn

class LocalBN1d(nn.Module):
    def __init__(self, embed_dim=300):
        super(LocalBN1d, self).__init__()

        self.embed_dim = embed_dim
        self.norm = nn.BatchNorm1d(embed_dim)
    
    def forward(self, graphs, tensor):

        return self.norm(tensor)

