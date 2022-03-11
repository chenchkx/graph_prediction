

import torch
import torch.nn as nn

from dgl.ops import segment

class LP2_Norm(nn.Module):
    ### Graph norm implemented by dgl toolkit
    ### more details please refer to the paper: Graphnorm: A principled approach to accelerating graph neural network training

    def __init__(self, embed_dim=300):
        super(LP2_Norm, self).__init__()

        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.mean_scale = nn.Parameter(torch.ones(embed_dim))

    def repeat_with_blockDiagMatrix(self, tensor, batch_list):
        num_node = batch_list.sum()
        batch_size = len(batch_list)
        adj_indx_row = torch.arange(num_node).to(tensor.device)
        adj_indx_col = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        adj_indx = torch.stack([adj_indx_row, adj_indx_col], 0)
        adj_elem = torch.ones(num_node).to(tensor.device) 
        adj_matrix = torch.sparse.FloatTensor(adj_indx, adj_elem, torch.Size([num_node, batch_size]))
        return torch.spmm(adj_matrix, tensor)

    def repeat(self, tensor, batch_list):
        batch_size = len(batch_list)
        batch_indx = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        return tensor[batch_indx]

    def forward(self, graph, tensor):
        batch_list = graph.batch_num_nodes()    
        tensor_abs = torch.abs(tensor)
        tensor_max = segment.segment_reduce(batch_list, tensor_abs, reducer='max')
        tensor_max[tensor_max<=1e-12] = 1e-12
        tensor_max = self.repeat(tensor_max, batch_list)

        return tensor/tensor_max

        ### Function torch.repeat_interleave() is faster. But it makes seed fail, the result is not reproducible.
        # mean = segment.segment_reduce(batch_list, tensor, reducer='mean')
        # mean = torch.repeat_interleave(mean, batch_list, dim=0, output_size = tensor.shape[0])
        # sub = tensor - mean * self.mean_scale
        # std = segment.segment_reduce(batch_list, sub.pow(2), reducer='mean')
        # std = (std + 1e-6).sqrt()
        # std = torch.repeat_interleave(std, batch_list, dim=0, output_size = tensor.shape[0])
        # return self.weight * sub / std + self.bias

        
