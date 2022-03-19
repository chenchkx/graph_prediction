

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import segment
from utils.utils_practice import repeat_tensor_interleave

class XXX_Norm3(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(XXX_Norm3, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else: 
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.center_weight = nn.Parameter(torch.zeros(num_features))

    def forward(self, graph, tensor):
        # self.denegative_parameter()
        batch_num_nodes = graph.batch_num_nodes()      
        # tensor_max = segment.segment_reduce(batch_num_nodes, tensor, reducer='max')
        # tensor_min = segment.segment_reduce(batch_num_nodes, tensor, reducer='min')
        # tensor_max = repeat_tensor_interleave(tensor_max, batch_num_nodes)
        # tensor_min = repeat_tensor_interleave(tensor_min, batch_num_nodes)

        tensor_max = segment.segment_reduce(batch_num_nodes, tensor.abs(), reducer='max')
        tensor_max = repeat_tensor_interleave(tensor_max, batch_num_nodes)

        tensor = tensor/(tensor_max+self.eps)      
        # tensor = tensor - self.center_weight*tensor.mean(0, keepdim=False)
        # tensor = torch.sign(tensor)*torch.pow(tensor.abs()+self.eps, 0.25)

        exponential_average_factor = 0.0 if self.momentum is None else self.momentum
        bn_training = True if self.training else (self.running_mean is None) and (self.running_var is None)
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None: 
                self.num_batches_tracked = self.num_batches_tracked + 1  
                if self.momentum is None:  
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else: 
                    exponential_average_factor = self.momentum
        results = F.batch_norm(
                    tensor, self.running_mean, self.running_var, None, None,
                    bn_training, exponential_average_factor, self.eps)
        # results_abs = results.abs()
        # results_abs[results_abs>1] = torch.log(results_abs[results_abs>1])
        # results = torch.sign(results)*results_abs

        if self.affine:
            results = self.weight*results + self.bias
        else:
            results = results
        
        return results

   
