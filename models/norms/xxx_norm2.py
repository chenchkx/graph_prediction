

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import segment
from utils.utils_practice import repeat_tensor_interleave

class XXX_Norm2(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(XXX_Norm2, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else: 
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.space_mapping_weight = nn.Parameter(torch.ones(num_features))
        self.space_mapping_bias = nn.Linear(1, num_features)
        self.space_retrace_weight = nn.Parameter(torch.ones(num_features))
        self.space_retrace_bias = nn.Embedding(1, num_features)

        self.register_buffer('bn_running_mean', torch.zeros(num_features))
        self.register_buffer('bn_num_batches_tracked', torch.tensor(0))
    
    def denegative_parameter(self):
        self.space_mapping_weight.data[self.space_mapping_weight.data<0]=0
        self.space_retrace_weight.data[self.space_retrace_weight.data<0]=0

    def forward(self, graph, tensor):
        # self.denegative_parameter()
        batch_num_nodes = graph.batch_num_nodes() 

        tensor_mean = segment.segment_reduce(batch_num_nodes, tensor, reducer='mean')
        tensor_mean = repeat_tensor_interleave(tensor_mean, batch_num_nodes)
        tensor_diff = tensor - (tensor_mean - (self.running_mean + self.space_mapping_bias.bias))
        tensor_var = segment.segment_reduce(batch_num_nodes, tensor_diff.pow(2), reducer='mean')
        tensor_var = repeat_tensor_interleave(tensor_var, batch_num_nodes)
        tensor_ =  tensor_diff/(self.space_mapping_weight*tensor_var+self.eps)

        exponential_average_factor = 0.0 if self.momentum is None else self.momentum
        bn_training = True if self.training else (self.running_mean is None) and (self.running_var is None)
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None: 
                self.num_batches_tracked = self.num_batches_tracked + 1  
                if self.momentum is None:  
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else: 
                    exponential_average_factor = self.momentum
        bn_output = F.batch_norm(
                    tensor_, self.running_mean, self.running_var, None, None,
                    bn_training, exponential_average_factor, self.eps)
        
        var_scale = torch.sigmoid(self.space_retrace_weight*self.running_var/(tensor_var+self.eps))
        results = var_scale*bn_output
        results = results + torch.sigmoid(self.space_retrace_bias.weight)*(tensor_mean-(self.running_mean+self.space_mapping_bias.bias))

        if self.affine:
            results = self.weight*results + self.bias
        else:
            results = results
        
        return results

   
