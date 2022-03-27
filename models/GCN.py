
import torch 
import torch.nn as nn
import dgl.function as fn
from models.encoder.ogb_encoder import OGB_NodeEncoder, OGB_EdgeEncoder
from models.norm.gnn_norm import GNN_Norm
from models.pool.global_pool import GlobalPooling
from models.activation.local_activation import LocalActivation

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
        

    def forward(self, graphs, nfeat, efeat):
        graphs = graphs.local_var()
        degs = (graphs.in_degrees().float() + 1).to(graphs.device)
        efeat = self.project_edge_feat(efeat)

        graphs.ndata['h_n'] = nfeat
        graphs.edata['h_e'] = efeat
        graphs.update_all(fn.u_add_e('h_n', 'h_e', 'm'), fn.sum('m', 'neigh'))

        rst = self.project_node_feat((nfeat + graphs.ndata['neigh']) / degs.view(-1, 1))

        return rst


class GCN(nn.Module):
    def __init__(self, dataset_name, embed_dim, output_dim, num_layer, 
                       norm_type='bn', pooling_type="mean", 
                       activation='relu', dropout=0.5):
        super(GCN, self).__init__()
        self.num_layer = num_layer
        # input layer
        self.atom_encoder = OGB_NodeEncoder(dataset_name, embed_dim)
        # middle layer. i.e., convolutional layer
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() 
        for i in range(num_layer):
            self.conv_layers.append(GCNConvLayer(dataset_name, embed_dim))
            self.norm_layers.append(GNN_Norm(norm_type, embed_dim))
        # output layer
        # self.predict = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim//2),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(embed_dim//2, output_dim)
        # )   
        self.predict = nn.Linear(embed_dim, output_dim)  
        
        # other modules in GNN
        self.activation = LocalActivation(activation)
        self.dropout = nn.Dropout(dropout)
        self.pooling = GlobalPooling(pooling_type)


    def forward(self, graphs, nfeat, efeat):
        # initializing node features h_n
        h_n = self.atom_encoder(nfeat)
        self.conv_feature = []
        self.norm_feature = []
        for layer in range(self.num_layer):
            x = h_n
            # conv_layer & norm layer
            h_n = self.conv_layers[layer](graphs, h_n, efeat)
            self.conv_feature.append(h_n)
            h_n = self.norm_layers[layer](graphs, h_n)
            self.norm_feature.append(h_n)
            # activation
            h_n = self.dropout(self.activation(h_n))
            # h_n = h_n + x                   
        # pooling & prediction
        g_n = self.pooling(graphs, h_n)
        pre = self.predict(g_n)
        return pre
