

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import segment
from utils.utils_practice import repeat_tensor_interleave

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

        self.fea_scale_weight = nn.Parameter(torch.zeros(num_features))
        self.var_scale_weight = nn.Parameter(torch.zeros(num_features))
        self.var_scale_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, graph, tensor):  

        graph_mean = segment.segment_reduce(graph.batch_num_nodes(), tensor, reducer='mean')
        tensor = tensor + repeat_tensor_interleave(self.fea_scale_weight*graph_mean, graph.batch_num_nodes())    
        # tensor = tensor + self.fea_scale_weight*tensor.mean(0, keepdim=False)

        if self.training: #训练模型
            #数据是二维的情况下，可以这么处理，其他维的时候不是这样的，但原理都一样。
            mean_bn = tensor.mean(0, keepdim=True).squeeze(0) #相当于x.mean(0, keepdim=False)
            tensor_diff = torch.abs(tensor - mean_bn)
            # tensor_diff[tensor_diff<1] = torch.pow(tensor_diff[tensor_diff<1],2)
            var_bn = tensor_diff.mean(0, keepdim=True).squeeze(0)
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
        results = (tensor - mean_bn)/(var_bn + self.eps)

        if self.affine:
            results = self.weight*results + self.bias
        else:
            results = results

        return results