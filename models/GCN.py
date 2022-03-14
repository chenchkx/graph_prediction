
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
    def __init__(self, dataset_name, embed_dim):
        super(GCNConvLayer, self).__init__()

        self.project_node_feat = nn.Linear(embed_dim, embed_dim)
        self.project_edge_feat = OGB_EdgeEncoder(dataset_name, embed_dim)
        self.project_residual =  nn.Embedding(1, embed_dim)


    def get_degs_norm(self, graphs):
        degs = (graphs.in_degrees().float() + 1).to(graphs.device)
        norm = torch.pow(degs, -0.5).unsqueeze(-1) 
        graphs.ndata['norm'] = norm
        graphs.apply_edges(fn.u_mul_v('norm', 'norm', 'norm'))
        norm = graphs.edata.pop('norm')

        return degs, norm

    def forward(self, graphs, nfeat, efeat):
        graphs = graphs.local_var()
        degs, norm =self.get_degs_norm(graphs)
        nfeat = self.project_node_feat(nfeat)
        efeat = self.project_edge_feat(efeat)

        graphs.ndata['feat'] = nfeat
        graphs.apply_edges(fn.copy_u('feat', 'e'))
        graphs.edata['e'] = norm * F.relu(graphs.edata['e'] + efeat)
        graphs.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'feat'))

        residual_nfeat = nfeat + self.project_residual.weight
        residual_nfeat = F.relu(residual_nfeat)
        residual_nfeat = residual_nfeat * 1. / degs.view(-1, 1)

        rst = graphs.ndata['feat'] + residual_nfeat
        return rst


class GCN(nn.Module):
    def __init__(self, dataset_name, embed_dim, output_dim, num_layer, 
                       norm_type='bn', pooling_type="mean", 
                       activation=F.relu, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layer = num_layer
        # input layer
        self.atom_encoder = OGB_NodeEncoder(dataset_name, embed_dim)
        # middle layer. i.e., convolutional layer
        self.bond_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() 
        for i in range(num_layer):
            self.conv_layers.append(GCNConvLayer(dataset_name, embed_dim))
            self.norm_layers.append(Norms(norm_type, embed_dim))
        # output layer
        self.predict = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim//2, output_dim)
        )   

        # other modules in GNN
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.pooling = Global_Pooling(pooling_type)


    def forward(self, graphs, nfeat, efeat):
        # initializing node features h_n
        h_n = self.atom_encoder(nfeat)
        self.conv_feature = []
        self.norm_feature = []
        for layer in range(self.num_layer):
            x = h_n
            # conv_layer
            h_n = self.conv_layers[layer](graphs, h_n, efeat)
            self.conv_feature.append(h_n)
            h_n = self.norm_layers[layer](graphs, h_n)
            self.norm_feature.append(h_n)
            # activation & residual 
            # if layer != self.num_layer - 1:
            h_n = self.dropout(self.activation(h_n))
            # h_n = h_n + x                   
        # pooling & prediction
        g_n = self.pooling(graphs, h_n)
        pre = self.predict(g_n)
        return pre
