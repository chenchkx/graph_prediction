

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

        self.fea_scale_weight = nn.Parameter(torch.zeros(num_features))
        self.var_scale_weight = nn.Parameter(torch.ones(num_features))
        self.var_scale_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, graph, tensor):  
        
        fea_scale = (graph.ndata['degrees_normed']*graph.ndata['batch_nodes']).unsqueeze(1)
        tensor = tensor*fea_scale
        graph_mean = repeat_tensor_interleave(segment.segment_reduce(graph.batch_num_nodes(), tensor, reducer='mean'), graph.batch_num_nodes())
        tensor = tensor + self.fea_scale_weight*graph_mean

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

        result_mean = repeat_tensor_interleave(segment.segment_reduce(graph.batch_num_nodes(), results, reducer='mean'), graph.batch_num_nodes())
        var_scale = torch.sigmoid(self.var_scale_weight*result_mean*graph.ndata['degrees_normed'].unsqueeze(1)+self.var_scale_bias)
        results = results*var_scale

        # if self.affine:
        #     results = self.weight*results + self.bias
        # else:
        #     results = results
     
        return results