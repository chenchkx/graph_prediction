

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops.segment import segment_reduce
from utils.utils_practice import repeat_tensor_interleave as segment_repeat

class XXX_Norm4(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(XXX_Norm4, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else: 
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.lambda_weight = nn.Parameter(torch.zeros(num_features))
        self.npower_weight = nn.Parameter(torch.zeros(1))
        self.mean_weight = nn.Parameter(torch.zeros(num_features))

    def forward(self, graph, tensor):  
        
        batch_num_nodes = graph.batch_num_nodes()
        fea_calibrate = graph.ndata['node_weight_g_normed']*graph.ndata['batch_nodes']
        weight_scales = graph.ndata['node_weight_g_normed']*torch.pow(graph.ndata['sg_nodes'], self.npower_weight)
        tensor = tensor*fea_calibrate
        mean_t = segment_reduce(batch_num_nodes, tensor, reducer='mean')
        tensor = tensor + self.mean_weight*segment_repeat(mean_t,batch_num_nodes)

        exponential_average_factor = 0.0 if self.momentum is None else self.momentum
        bn_training = True if self.training else ((self.running_mean is None) and (self.running_var is None))
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None: 
                self.num_batches_tracked = self.num_batches_tracked + 1  
                if self.momentum is None:  
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else: 
                    exponential_average_factor = self.momentum
            batch_var = tensor.var(0, keepdim=False)
        else:
            batch_var = self.running_var
        results = F.batch_norm(
                    tensor, self.running_mean, self.running_var, None, None,
                    bn_training, exponential_average_factor, self.eps)

        # fea_scale = segment_repeat((batch_var/(segment_reduce(batch_num_nodes, torch.pow(results,2),reducer='mean')+self.eps)).sqrt(), batch_num_nodes)      
        sacle_factor = torch.sigmoid(self.lambda_weight*weight_scales.repeat(1,self.num_features))
     
        if self.affine:
            results = self.weight*sacle_factor*results + self.bias    
        else:
            results = sacle_factor*results
     
        return results
   