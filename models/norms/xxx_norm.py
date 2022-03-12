

import torch
import torch.nn as nn

from dgl.ops import segment

class XXX_Norm(nn.Module):
    ### Graph norm implemented by dgl toolkit
    ### more details please refer to the paper: Graphnorm: A principled approach to accelerating graph neural network training

    def __init__(self, embed_dim=300, momentum=0.9):
        super(XXX_Norm, self).__init__()
        self.momentum = momentum

        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.weight = nn.Parameter(torch.ones(embed_dim))

        self.register_buffer('running_mean',torch.zeros(embed_dim))
        self.register_buffer('running_var',torch.zeros(embed_dim))
        self.register_buffer('num_batches_tracked',torch.tensor(0))

    def repeat(self, tensor, batch_list):
        batch_size = len(batch_list)
        batch_indx = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        return tensor[batch_indx]

    def forward(self, graph, tensor):
        batch_list = graph.batch_num_nodes()    
        tensor_abs = torch.abs(tensor)
        tensor_max = segment.segment_reduce(batch_list, tensor_abs, reducer='max')
        tensor_max[tensor_max<=1e-12] = 1e-12
        tensor_max = tensor_max.max(1)[0].unsqueeze(1).repeat(1,tensor_max.shape[1])
        tensor_max = self.repeat(tensor_max.sqrt(), batch_list)
        tensor = tensor/tensor_max

        if self.training:
            mean_bn = tensor.mean(0, keepdim=True).squeeze(0) #相当于x.mean(0, keepdim=False)
            var_bn = tensor.var(0, keepdim=True).squeeze(0) #相当于x.var(0, keepdim=False)     
            if self.momentum is not None:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:  #直接取平均,以下是公式变形，即 m_new = (m_old*n + new_value)/(n+1)
                self.running_mean = self.running_mean+(mean_bn.data-self.running_mean)/(self.num_batches_tracked+1)
                self.running_var = self.running_var+(var_bn.data-self.running_var)/(self.num_batches_tracked+1)
            self.num_batches_tracked += 1
        else: #eval 
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)                  

        eps = 1e-5
        tensor_normalized = (tensor - mean_bn) / torch.sqrt(var_bn + eps)
        
        results = self.weight * tensor_normalized + self.bias

        return results

   
