

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import segment
from utils.utils_practice import repeat_tensor_interleave

class XXX_Norm7(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(XXX_Norm7, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else: 
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.center_weight = nn.Parameter(torch.zeros(num_features))
        self.latent_energy = nn.Parameter(torch.zeros(num_features))
        self.graph_scale_weight = nn.Parameter(torch.ones(num_features))
        self.graph_scale_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, graph, tensor):
        # self.denegative_parameter()
        
        tensor = tensor + self.center_weight*tensor.mean(0, keepdim=False)
        tensor = tensor*torch.sigmoid(self.latent_energy)/(tensor.std(0, keepdim=False)+self.eps)

        exponential_average_factor = 0.0 if self.momentum is None else self.momentum
        bn_training = True if self.training else ((self.running_mean is None) and (self.running_var is None))
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

        batch_var = segment.segment_reduce(graph.batch_num_nodes(), torch.pow(results-results.mean(0, keepdim=False),2), reducer='mean')
        scale_weight = repeat_tensor_interleave(torch.sigmoid(self.graph_scale_weight*batch_var+self.graph_scale_bias), graph.batch_num_nodes())
        
        if self.affine:
            results = self.weight*scale_weight*results + self.bias
        else:
            results = scale_weight*results
        
        return results

   
