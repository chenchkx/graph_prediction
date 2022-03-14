

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import segment
from utils.utils_practice import repeat_tensor_interleave

class XXX_Norm(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(XXX_Norm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else: 
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.mean_scale_weight = nn.Parameter(torch.zeros(num_features))
        self.var_scale_weight = nn.Parameter(torch.ones(num_features))
        self.var_scale_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, graph, tensor):
        batch_num_nodes = graph.batch_num_nodes() 

        bn_mean = tensor.mean(0, keepdim=False)
        bn_var = tensor.var(0, keepdim=False)
        bn_output = (tensor-bn_mean)/torch.sqrt(bn_var+self.eps)

        tensor_mean = segment.segment_reduce(batch_num_nodes, tensor, reducer='mean')
        tensor_mean = bn_mean - tensor_mean
        tensor_mean = repeat_tensor_interleave(tensor_mean, batch_num_nodes)
        tensor = tensor + self.mean_scale_weight*tensor_mean

        tensor_var = segment.segment_reduce(batch_num_nodes, tensor.pow(2), reducer='mean')
        tensor_var = repeat_tensor_interleave(tensor_var, batch_num_nodes)
        
        var_scale = self.var_scale_weight*torch.sqrt((bn_var+self.eps)/(tensor_var+self.eps))
        var_scale = torch.sigmoid(var_scale)
        # var_scale = torch.tanh(self.var_scale_weight*self.running_var/(tensor_var+self.eps)-1) + 1
        if self.affine:
            results = self.weight*var_scale*bn_output + self.bias
        else:
            results = var_scale*bn_output

        return results

   
