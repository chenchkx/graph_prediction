
import os
import time
import torch
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ogb.graphproppred.dataset_dgl import DglGraphPropPredDataset, collate_dgl

from optims.optim_acc_with_mce import ModelOptLearning_MCE
from optims.optim_sync_ogb_mol import ModelOptLearning_OGB_HIV
from optims.optim_sync_ogb_mol_statistics import ModelOptLearning_OGB_HIV_Statistics
from optims.optim_sync_ogb_ppa_statistics import ModelOptLearning_OGB_PPA_Statistics
from models.GIN import GIN
from models.GCN import GCN
nfs_dataset_path1 = '/mnt/nfs/ckx/datasets/ogb/graph/'
nfs_dataset_path2 = '/nfs4-p1/ckx/datasets/ogb/graph/'

# add node instance weight
def add_node_weight(dataset):
 
    for g in dataset:
        ## coefficients in terms of mean value
        ## version 1.0
        row, col = g[0].edges()
        num_of_nodes = g[0].num_nodes()
        adj = torch.zeros(num_of_nodes, num_of_nodes)
        for i in range(row.shape[0]):
            adj[row[i]][col[i]]=1.0        
        A_array = adj.detach().numpy()
        G = nx.from_numpy_matrix(A_array)

        node_weight = torch.zeros(num_of_nodes,1)
        node_weight_g = torch.zeros(num_of_nodes,1)
        for i in range(len(A_array)):
            s_indexes = []
            for j in range(len(A_array)):
                s_indexes.append(i)
                if(A_array[i][j]==1):
                    s_indexes.append(j)      
            subgraph_nodes = len(list(G.subgraph(s_indexes).nodes))
            subgraph_edges = G.subgraph(s_indexes).number_of_edges() + subgraph_nodes
            subgraph_nodes = subgraph_nodes + 1
            instance_energy = subgraph_edges/(subgraph_nodes*(subgraph_nodes-1))
            node_weight[i] = instance_energy*(subgraph_nodes**2)
            node_weight_g[i] = instance_energy*subgraph_nodes

            # s_node = len(list(G.subgraph(s_indexes).nodes))
            # if s_node == 1:
            #     node_weight_g[i] = 1
            # else:
            #     s_edge = G.subgraph(s_indexes).number_of_edges()
            #     i_engy = 2*s_edge/(s_node*(s_node-1))
            #     node_weight_g[i] = i_engy*(s_node**2)


        ## coefficients in terms of graph structure
        # row, col = g[0].edges()
        # num_of_nodes = g[0].num_nodes()
        # adj = torch.zeros(num_of_nodes, num_of_nodes)
        # for i in np.arange(row.shape[0]):
        #     adj[row[i]][col[i]]=1.0

        # A_array = adj.detach().numpy()
        # G = nx.from_numpy_matrix(A_array)
    
        # sub_graphs = []
        # subgraph_nodes_list = []
        # sub_graphs_adj = []
        # sub_graph_edges = []
        # new_adj = torch.zeros(A_array.shape[0], A_array.shape[0])

        # for i in np.arange(len(A_array)):
        #     s_indexes = []
        #     for j in np.arange(len(A_array)):
        #         s_indexes.append(i)
        #         if(A_array[i][j]==1):
        #             s_indexes.append(j)
        #     sub_graphs.append(G.subgraph(s_indexes))

        # for i in np.arange(len(sub_graphs)):
        #     subgraph_nodes_list.append(list(sub_graphs[i].nodes))

        # for index in np.arange(len(sub_graphs)):
        #     sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())

        # for index in np.arange(len(sub_graphs)):
        #     sub_graph_edges.append(sub_graphs[index].number_of_edges())

        # for node in np.arange(len(subgraph_nodes_list)):
        #     sub_adj = sub_graphs_adj[node]
        #     for neighbors in np.arange(len(subgraph_nodes_list[node])):
        #         index = subgraph_nodes_list[node][neighbors]
        #         count = torch.tensor(0).float()
        #         if(index==node):
        #             continue
        #         else:
        #             c_neighbors = set(subgraph_nodes_list[node]).intersection(subgraph_nodes_list[index])
        #             if index in c_neighbors:
        #                 nodes_list = subgraph_nodes_list[node]
        #                 sub_graph_index = nodes_list.index(index)
        #                 c_neighbors_list = list(c_neighbors)
        #                 for i, item1 in enumerate(nodes_list):
        #                     if(item1 in c_neighbors):
        #                         for item2 in c_neighbors_list:
        #                             j = nodes_list.index(item2)
        #                             count += sub_adj[i][j]

        #             new_adj[node][index] = count/2
        #             new_adj[node][index] = new_adj[node][index]/(len(c_neighbors)*(len(c_neighbors)-1))
        #             new_adj[node][index] = new_adj[node][index] * (len(c_neighbors)**2)


        g[0].ndata['snorm_n'] = torch.FloatTensor(g[0].num_nodes()).fill_(1/g[0].num_nodes()**0.5) 
        g[0].ndata['batch_nodes'] = torch.FloatTensor(g[0].num_nodes()).fill_(g[0].num_nodes()).unsqueeze(1) 
        g[0].ndata['node_weight'] = node_weight
        g[0].ndata['node_weight_normed'] = node_weight/node_weight.sum()
        g[0].ndata['node_weight_normed_power'] = node_weight**2/node_weight.sum()
        g[0].ndata['node_weight_g'] = node_weight_g
        g[0].ndata['node_weight_g_normed'] = node_weight_g/node_weight_g.sum()
        g[0].ndata['node_weight_g_normed_power'] = node_weight_g**2/node_weight_g.sum()
        g[0].ndata['degrees'] = g[0].in_degrees() + 1
        g[0].ndata['degrees_normed'] = g[0].ndata['degrees']/g[0].ndata['degrees'].sum()
        g[0].ndata['degrees_normed_power'] = torch.mean(g[0].ndata['degrees'].float())*g[0].ndata['degrees']/g[0].ndata['degrees'].sum()


### load and preprocess dataset 
def load_process_dataset(args):    
    # check nfs dataset path
    if os.path.exists(nfs_dataset_path1):
        args.datadir = nfs_dataset_path1
    elif os.path.exists(nfs_dataset_path2):
        args.datadir = nfs_dataset_path2
    dataset = DglGraphPropPredDataset(name=args.dataset, root=args.datadir)

    # preprocess the node features in ogbg-ppa dataset
    if 'ppa' in args.dataset:
        for g in dataset:
            g[0].ndata['feat'] = torch.zeros(g[0].num_nodes(), dtype=int)
    # add node instance weight
    if args.node_weight:
        start_time = time.time()
        add_node_weight(dataset)
        end_time = time.time()

    # split_idx for training, valid and test 
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_dgl, num_workers=0)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, 
                              collate_fn=collate_dgl, num_workers=0)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, 
                              collate_fn=collate_dgl, num_workers=0)    
    print('- ' * 30)
    if args.node_weight:
        print(f'{args.dataset} dataset loaded, preprocessing using {end_time-start_time} seconds.')
    else:
        print(f'{args.dataset} dataset loaded, without preprocessing.')
    print('- ' * 30)        
    return dataset, train_loader, valid_loader, test_loader


### load gnn model
def load_model(args):
    if args.model == 'GCN':
        model = GCN(args.embed_dim, args.output_dim, args.num_layer, args).to(args.device)
    elif args.model == 'GIN':
        model = GIN(args.dataset, args.embed_dim, args.output_dim, args.num_layer, args.norm_type).to(args.device)
    print('- ' * 30)
    print(f'{args.model} with {args.pool_type} pool, {args.norm_type} norm, {args.activation} act, {args.dropout} dropout')
    print('- ' * 30)   
    return model


### load model optimizing and learning class
def ModelOptLoading(model, optimizer, 
                    train_loader, valid_loader, test_loader,
                    args):
    if 'mol' in args.dataset:
        if 'ogb' in args.loss_type:
            modelOptm = ModelOptLearning_OGB_HIV_Statistics(
                                    model=model, 
                                    optimizer=optimizer,
                                    train_loader=train_loader,
                                    valid_loader=valid_loader,
                                    test_loader=test_loader,
                                    args=args)
        elif 'mce' in args.loss_type:
            modelOptm = ModelOptLearning_MCE(
                                    model=model, 
                                    optimizer=optimizer,
                                    train_loader=train_loader,
                                    valid_loader=valid_loader,
                                    test_loader=test_loader,
                                    args=args)     

    elif 'ppa' in args.dataset:
        modelOptm = ModelOptLearning_OGB_PPA_Statistics(
                                model=model, 
                                optimizer=optimizer,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                test_loader=test_loader,
                                args=args)

    return modelOptm

def reset_batch_size(num_graphs):
    if num_graphs < 50000:
        return 128 if num_graphs < 10000 else 256
    else:
        return 512 if num_graphs < 100000 else 1024

def get_ogb_output_dim(dataset, args):
    if 'mol' in args.dataset:
        if 'mce' in args.loss_type:
            return int(dataset.num_classes)
        elif 'ogb' in args.loss_type:
            return int(dataset.num_tasks)
    elif 'ppa' in args.dataset:
        return int(dataset.num_classes)

### add new arguments
def args_(args, dataset): 

    args.task_type = dataset.task_type
    args.eval_metric = dataset.eval_metric    
    args.batch_size = reset_batch_size(len(dataset))
    args.output_dim = get_ogb_output_dim(dataset, args)
    args.identity = (f"{args.dataset}-"+
                     f"{args.model}-"+
                     f"{args.num_layer}-"+
                     f"{args.embed_dim}-"+
                     f"{args.pool_type}-"+
                     f"{args.norm_type}-"+
                     f"{args.norm_affine}-"+
                     f"{args.activation}-"+
                     f"{args.dropout}-"+
                     f"{args.lr_warmup_type}-"+
                     f"{args.lr}-"+  
                     f"{args.weight_decay}-"+
                     f"{args.loss_type}-"+
                     f"{args.batch_size}-"+
                     f"{args.seed}-"+
                     f"{args.runs}"
                     )
    if not os.path.exists(args.logs_perf_dir):
        os.mkdir(args.logs_perf_dir)
    args.perf_xlsx_dir = os.path.join(args.logs_perf_dir, 'xlsx')
    args.perf_imgs_dir = os.path.join(args.logs_perf_dir, 'imgs')
    args.perf_dict_dir = os.path.join(args.logs_perf_dir, 'dict')
    args.perf_best_dir = os.path.join(args.logs_perf_dir, 'best') 

    if not os.path.exists(args.logs_stas_dir):
        os.mkdir(args.logs_stas_dir)
    args.stas_xlsx_dir = os.path.join(args.logs_stas_dir, 'xlsx')
    args.stas_imgs_dir = os.path.join(args.logs_stas_dir, 'imgs')

    return args


def print_best_log(args,  eopch_slice=0, 
                   metric_list='all'):
    key_metric=f'valid-{args.eval_metric}'
    logs_table = pd.read_excel(os.path.join(args.perf_xlsx_dir, args.identity+'.xlsx'))
    metric_log = logs_table[key_metric]
    if "classification" in args.task_type:
        best_epoch = metric_log[eopch_slice:].idxmax()  
    else:
        best_epoch = metric_log[eopch_slice:].idxmin() 
    best_frame = logs_table.loc[best_epoch]
    if not os.path.exists(args.perf_best_dir):
        os.mkdir((args.perf_best_dir))
    best_frame.to_excel(os.path.join(args.perf_best_dir, args.identity+'.xlsx'))

    if metric_list == 'all':
        print(best_frame)
    else:
        for metric in metric_list:
            print(f'{metric }: {best_frame[metric]}')
    return 0


def plot_logs(args, metric_list='all'):
    if not os.path.exists(args.perf_imgs_dir):
        os.mkdir(args.perf_imgs_dir)
    logs_table = pd.read_excel(os.path.join(args.perf_xlsx_dir, args.identity+'.xlsx'))

    for metric in metric_list:
        metric_epochs = logs_table[metric]
        plt.plot(range(len(metric_epochs)), metric_epochs, 'g-')
        plt.savefig(os.path.join(args.perf_imgs_dir, args.identity + f'-{metric}.png'))
        plt.close()
    return 0 


### set random seed
def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if args.device >= 0:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
