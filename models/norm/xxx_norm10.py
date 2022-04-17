

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import segment
from utils.utils_practice import repeat_tensor_interleave

class XXX_Norm10(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(XXX_Norm10, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else: 
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.fea_scale_weight = nn.Parameter(torch.zeros(num_features))
        self.var_scale_weight = nn.Parameter(torch.ones(num_features))
        self.var_scale_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, graph, tensor):  
      
        tensor = tensor*(graph.ndata['degrees_normed']*graph.ndata['batch_nodes']).unsqueeze(1)
        graph_mean = segment.segment_reduce(graph.batch_num_nodes(), tensor, reducer='mean')
        tensor = tensor + self.fea_scale_weight*repeat_tensor_interleave(graph_mean, graph.batch_num_nodes())

        exponential_average_factor = 0.0 if self.momentum is None else self.momentum
        bn_training = True if self.training else ((self.running_mean is None) and (self.running_var is None))
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None: 
                self.num_batches_tracked = self.num_batches_tracked + 1  
                if self.momentum is None:  
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else: 
                    exponential_average_factor = self.momentum
            batch_mean = tensor.mean(0, keepdim=False)
            batch_var = tensor.var(0, keepdim=False)
        else:  
            batch_mean = torch.autograd.Variable(self.running_mean)
            batch_var = torch.autograd.Variable(self.running_var)         
        results = F.batch_norm(
                    tensor, self.running_mean, self.running_var, None, None,
                    bn_training, exponential_average_factor, self.eps)

        var_scale = segment.segment_reduce(graph.batch_num_nodes(), torch.pow(results,2), reducer='mean')
        var_scale = torch.sigmoid(self.var_scale_weight*batch_var/(var_scale+self.eps)+self.var_scale_bias)
        results = results*repeat_tensor_interleave(var_scale, graph.batch_num_nodes()) 

        # if self.affine:
        #     results = results + self.bias*batch_mean
        # else:
        #     results = results  

        return results

   