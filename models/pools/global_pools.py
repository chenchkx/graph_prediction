from ast import Global
from turtle import forward


import torch
import torch.nn as nn
from dgl.nn.pytorch.glob import AvgPooling, SumPooling, MaxPooling

class Global_Pooling(nn.Module):

    def __init__(self, pooling_type) -> None:
        super(Global_Pooling).__init__()

        if pooling_type == "sum":
            self.pooling = SumPooling()
        elif pooling_type == "mean":
            self.pooling = AvgPooling()
        elif pooling_type == "max":
            self.pooling = MaxPooling()

    def forward(self, graph, feat):


        return self.pooling(graph, feat)
