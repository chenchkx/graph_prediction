

import torch
import torch.nn as nn

from dgl.ops import segment

class InstanceNorm(nn.Module):

    def __init__(self, embed_dim=300, affine=True):
        super(InstanceNorm, self).__init__()

        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.affine = affine

    def repeat(self, tensor, batch_list):
        batch_size = len(batch_list)
        batch_indx = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        return tensor[batch_indx]

    def forward(self, graph, tensor):
        batch_list = graph.batch_num_nodes()      
        mean = segment.segment_reduce(batch_list, tensor, reducer='mean')
        mean = self.repeat(mean, batch_list)
        sub = tensor - mean
        std = segment.segment_reduce(batch_list, sub.pow(2), reducer='mean')
        std = (std + 1e-6).sqrt()
        std = self.repeat(std, batch_list)
        rst = sub / std

        if self.affine:
            rst = self.weight*rst + self.bias
        else:
            rst = rst
        return rst

        
