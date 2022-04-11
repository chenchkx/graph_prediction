

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import segment
from utils.utils_practice import repeat_tensor_interleave

class XXX_Norm5(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(XXX_Norm5, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else: 
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.fea_scale_weight = nn.Parameter(torch.zeros(num_features))
        self.var_scale_weight = nn.Parameter(torch.zeros(num_features))
        self.var_scale_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, graph, tensor):  
      
        tensor = tensor*(graph.ndata['degrees_normed']*graph.ndata['batch_nodes']).unsqueeze(1)

        if self.training: 
            mean_bn = tensor.mean(0, keepdim=False) #相当于x.mean(0, keepdim=False)
            var_bn = tensor.var(0, keepdim=False) #相当于x.var(0, keepdim=False)
            if self.momentum is not None:
                self.running_mean.mul_(1 - self.momentum)
                self.running_mean.add_((self.momentum) * mean_bn.data)
                self.running_var.mul_(1 - self.momentum)
                self.running_var.add_((self.momentum) * var_bn.data)
            else:  #直接取平均,以下是公式变形，即 m_new = (m_old*n + new_value)/(n+1)
                self.running_mean = self.running_mean+(mean_bn.data-self.running_mean)/(self.num_batches_tracked+1)
                self.running_var = self.running_var+(var_bn.data-self.running_var)/(self.num_batches_tracked+1)
            self.num_batches_tracked += 1
        else: #eval模式
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)       
        results = (tensor - mean_bn) / torch.sqrt(var_bn + self.eps)

        # if self.affine:
        #     results = self.weight*results + repeat_tensor_interleave(self.bias*graph_mean, graph.batch_num_nodes())
        # else:
        #     results = results
     
        return results
