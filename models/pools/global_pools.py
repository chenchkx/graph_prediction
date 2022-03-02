
import torch
import torch.nn as nn
from dgl.nn.pytorch.glob import AvgPooling, SumPooling, MaxPooling
from models.pools.dkepool import DKEPooling

class Global_Pooling(nn.Module):
    def __init__(self, pooling_type):
        super(Global_Pooling, self).__init__()

        self.pooling_type = pooling_type
        if pooling_type == "sum":
            self.pooling = SumPooling()
        elif pooling_type == "mean":
            self.pooling = AvgPooling()
        elif pooling_type == "max":
            self.pooling = MaxPooling()
        elif pooling_type == "dke":
            self.pooling = DKEPooling()

    def forward(self, graph, feat):

        if self.pooling_type == 'dke':
            batch_list = graph.batch_num_nodes()
            representation = self.pooling(batch_list, feat)
        else:
            representation = self.pooling(graph, feat)
        return representation
