
import torch 
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from models.encoders.encoders import OGB_NodeEncoder, OGB_EdgeEncoder
from models.norms.norms import Norms
from models.pools.global_pools import Global_Pooling

class GCNConvLayer_SparseAdj(nn.Module):
    def __init__(self, embed_dim, aggregator_type='mean', self_loops=True):
        super(GCNConvLayer_SparseAdj, self).__init__()

        self.aggregator_type = aggregator_type
        self.self_loops = self_loops

        self.update_feat = nn.Linear(embed_dim, embed_dim)

    def aggregate(self, graphs, nfeat, efeat, aggregator_type, self_loops):
        num_node = nfeat.shape[0]
        degrees = graphs.in_degrees()
        ### adj matrix
        adj_indx = torch.stack(graphs.edges(), 0)
        adj_elem = torch.ones(efeat.shape[0]).to(adj_indx.device) 
        adj_neibor = torch.sparse.FloatTensor(adj_indx, adj_elem, torch.Size([num_node, num_node]))
        adj_matrix = adj_neibor     
        if self_loops:
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)]).to(adj_indx.device)
            self_elem = torch.ones(num_node).to(adj_indx.device)
            adj_self = torch.sparse.FloatTensor(self_loop_edge, self_elem, torch.Size([num_node, num_node]))  
            adj_matrix = adj_matrix + adj_self
            degrees = degrees + 1
        ### feature aggregate
        rst = torch.spmm(adj_matrix, nfeat)
        if aggregator_type == 'mean':
            rst = rst/degrees.unsqueeze(1)
        return rst

    def forward(self, graphs, nfeat, efeat):
        graphs = graphs.local_var()

        rst = self.aggregate(graphs, nfeat, efeat, self.aggregator_type, self.self_loops)
        # node feature updating 
        rst = self.update_feat(rst)
        return rst


class GCNConvLayer(nn.Module):
    def __init__(self, embed_dim, aggregator_type='mean'):
        super(GCNConvLayer, self).__init__()

        self.update_feat = nn.Linear(embed_dim, embed_dim)

        if aggregator_type == 'sum':
            self.reduce = fn.sum
        elif aggregator_type == 'mean':
            self.reduce = fn.mean
        elif aggregator_type == 'max':
            self.reduce = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))

    def forward(self, graphs, nfeat, efeat):
        graphs = graphs.local_var()
        # aggragation
        # graphs.ndata['h_n'] = nfeat
        # graphs.edata['h_e'] = efeat
        # graphs.update_all(fn.u_add_e('h_n', 'h_e', 'm'),
        #                   self.reduce('m', 'neigh'))
        # rst = self.update_feat(graphs.ndata['neigh'] +  nfeat)
        graphs.ndata['feat'] = nfeat
        graphs.apply_edges(fn.copy_u('feat', 'e'))
        graphs.edata['e'] = F.relu(efeat + graphs.edata['e'])
        graphs.update_all(fn.copy_e('e','m'), self.reduce('m', 'feat'))

        rst = self.update_feat(graphs.ndata['feat'] +  nfeat)
        return rst


class GCN(nn.Module):
    def __init__(self, dataset_name, embed_dim, output_dim, num_layer, 
                       norm_type='bn', aggregator_type='mean', pooling_type="mean", 
                       activation=F.relu, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layer = num_layer
        # input layer
        self.atom_encoder = OGB_NodeEncoder(dataset_name, embed_dim)
        # convolutional layer & bond layer 
        self.bond_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() 
        for i in range(num_layer-1):
            self.bond_layers.append(OGB_EdgeEncoder(dataset_name, embed_dim))
            self.conv_layers.append(GCNConvLayer(embed_dim))
            # self.conv_layers.append(GCNConvLayer_SparseAdj(embed_dim))
            self.norm_layers.append(Norms(norm_type, embed_dim))
        # output layer
        self.predict = nn.Linear(embed_dim, output_dim)     

        # modules in GNN
        self.pooling = Global_Pooling(pooling_type)

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
            # norm in batch graphs
            h_e = self.bond_layers[layer](efeat)
            h_n = self.conv_layers[layer](graphs, h_n, h_e)
            self.conv_feature.append(h_n)
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
