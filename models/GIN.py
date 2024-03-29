
import torch 
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from models.encoder.ogb_encoder import OGB_NodeEncoder, OGB_EdgeEncoder
from dgl.nn.pytorch.glob import AvgPooling, SumPooling, MaxPooling

from models.norm.gnn_norm import GNN_Norm

class GINConvLayer(nn.Module):
    def __init__(self, embed_dim, aggregator_type='sum', norm_type='bn',
                 learn_eps=False, init_eps = 0):
        super(GINConvLayer, self).__init__()

        # MLP for node updating in GIN convolution
        self.mlp_project_in = nn.Linear(embed_dim, 2 * embed_dim)
        self.mlp_hidden_norm = GNN_Norm(norm_type, 2 * embed_dim)
        self.mlp_project_out = nn.Linear(2 * embed_dim, embed_dim)

        if aggregator_type == 'sum':
            self.reduce = fn.sum
        elif aggregator_type == 'mean':
            self.reduce = fn.mean
        elif aggregator_type == 'max':
            self.reduce = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))

        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

    def forward(self, graphs, nfeat, efeat):
        graphs = graphs.local_var()
        # aggragation
        graphs.ndata['feat'] = nfeat
        graphs.apply_edges(fn.copy_u('feat', 'e'))
        graphs.edata['e'] = F.relu(efeat + graphs.edata['e'])
        # graphs.edata['e'] = efeat + graphs.edata['e']
        graphs.update_all(fn.copy_e('e','m'), self.reduce('m', 'feat'))
        # node feature updating 
        rst = self.mlp_project_in(graphs.ndata['feat'] + (1 + self.eps) * nfeat)
        rst = self.mlp_hidden_norm(graphs, rst)
        rst = self.mlp_project_out(F.relu(rst))
        return rst


class GIN(nn.Module):
    def __init__(self, dataset_name, embed_dim, output_dim, num_layer,
                       norm_type='bn', aggregator_type='sum', pooling_type="mean", 
                       activation=F.relu, dropout=0.5):
        super(GIN, self).__init__()
        self.num_layer = num_layer
        # input layer
        self.atom_encoder = OGB_NodeEncoder(dataset_name, embed_dim)
        # convolutional layer & bond layer 
        self.bond_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() 
        for i in range(num_layer-1):
            self.bond_layers.append(OGB_EdgeEncoder(dataset_name, embed_dim))
            self.conv_layers.append(GINConvLayer(embed_dim, norm_type=norm_type))
            self.norm_layers.append(GNN_Norm(norm_type, embed_dim))
        # output layer
        self.predict = nn.Linear(embed_dim, output_dim)   

        # pooling modules in GNN
        if pooling_type == "sum":
            self.pooling = SumPooling()
        elif pooling_type == "mean":
            self.pooling = AvgPooling()
        elif pooling_type == "max":
            self.pooling = MaxPooling()
        else:
            raise KeyError('Pooling type {} not recognized.'.format(pooling_type))

        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, graphs, nfeat, efeat):
        # initializing node features h_n
        h_n = self.atom_encoder(nfeat)

        self.conv_feature = []
        self.norm_feature = []

        for layer in range(self.num_layer-1):
            x = h_n
            # initializing edge features h_e & graph convolution for node features
            h_e = self.bond_layers[layer](efeat)
            h_n = self.conv_layers[layer](graphs, h_n, h_e)
            self.conv_feature.append(h_n)
            # graphs norm in batch graphs
            h_n = self.norm_layers[layer](graphs, h_n)
            self.norm_feature.append(h_n)
            # activation & residual 
            if layer != self.num_layer - 2:
                h_n = self.activation(h_n)
            h_n = self.dropout(h_n)
            h_n = h_n + x                
        # pooling & prediction
        g_n = self.pooling(graphs, h_n)
        pre = self.predict(g_n)
        return pre
