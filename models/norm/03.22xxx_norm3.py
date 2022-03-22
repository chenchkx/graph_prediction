

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
        self.latent_energy = nn.Parameter(torch.zeros(num_features))
        self.graph_project_weight = nn.Parameter(torch.zeros(num_features))
        self.graph_project_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, graph, tensor):
        # self.denegative_parameter()
        
        tensor = tensor + self.center_weight*tensor.mean(0, keepdim=False)
        results = torch.sigmoid(self.latent_energy)*tensor/(tensor.std(0, keepdim=False)+self.eps)

        if self.affine:
            results = self.weight*results + self.bias
        else:
            results = results
        
        return results

   
