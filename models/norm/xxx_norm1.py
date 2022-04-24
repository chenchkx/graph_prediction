

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops.segment import segment_reduce
from utils.utils_practice import repeat_tensor_interleave as segment_repeat

class XXX_Norm1(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(XXX_Norm1, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else: 
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)


    def forward(self, graph, tensor):  
        
        batch_num_nodes = graph.batch_num_nodes()
        fea_scale = (graph.ndata['degrees_normed']*graph.ndata['batch_nodes']).unsqueeze(1)

        tensor = tensor*fea_scale

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
            batch_mean = self.running_mean
            batch_var = self.running_var
        results = F.batch_norm(
                    tensor, self.running_mean, self.running_var, None, None,
                    bn_training, exponential_average_factor, self.eps)
    
        var_scale = segment_repeat(batch_var/(segment_reduce(batch_num_nodes,torch.pow(results,2),reducer='mean')+self.eps), batch_num_nodes)   
        var_scale = torch.sigmoid(var_scale*graph.ndata['degrees_normed'].unsqueeze(1))
        
        if self.affine:
            results = self.weight*var_scale*results + self.bias*batch_mean    
        else:
            results = var_scale*results
     
        return results

   