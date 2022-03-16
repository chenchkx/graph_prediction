
import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ogb.graphproppred.dataset_dgl import DglGraphPropPredDataset, collate_dgl

from optims.optim_sync_ogb_mol import ModelOptLearning_OGB_HIV
from optims.optim_sync_ogb_mol_statistics import ModelOptLearning_OGB_HIV_Statistics
from optims.optim_sync_ogb_ppa_statistics import ModelOptLearning_OGB_PPA_Statistics
from models.GIN import GIN
from models.GCN import GCN
nfs_dataset_path1 = '/mnt/nfs/ckx/datasets/ogb/graph/'
nfs_dataset_path2 = '/nfs4-p1/ckx/datasets/ogb/graph/'

### load dataset 
def load_data(args):    
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
    # split_idx for training, valid and test 
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_dgl, num_workers=0)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, 
                              collate_fn=collate_dgl, num_workers=0)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, 
                              collate_fn=collate_dgl, num_workers=0)    
    print('- ' * 30)
    print(f'{args.dataset} dataset loaded.')
    print('- ' * 30)
    return dataset, train_loader, valid_loader, test_loader


### load gnn model
def load_model(args):
    if args.model == 'GCN':
        model = GCN(args.dataset, args.embed_dim, args.output_dim, args.num_layer, 
                    norm_type = args.norm_type, pooling_type = args.pool_type).to(args.device)
    elif args.model == 'GIN':
        model = GIN(args.dataset, args.embed_dim, args.output_dim, args.num_layer, args.norm_type).to(args.device)
    print('- ' * 30)
    print(f'{args.model} network with {args.norm_type} norm, {args.pool_type} pooling')
    print('- ' * 30)   
    return model


### load model optimizing and learning class
def ModelOptLoading(model, optimizer, 
                    train_loader, valid_loader, test_loader,
                    args):
    if 'mol' in args.dataset:
        modelOptm = ModelOptLearning_OGB_HIV_Statistics(
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

def get_ogb_output_dim(dataset, dataset_name):
    if 'mol' in dataset_name:
        return dataset.num_tasks
    elif 'ppa' in dataset_name:
        return int(dataset.num_classes)

### add new arguments
def args_(args, dataset): 

    args.task_type = dataset.task_type
    args.eval_metric = dataset.eval_metric    
    args.batch_size = reset_batch_size(len(dataset))
    args.output_dim = get_ogb_output_dim(dataset, args.dataset)
    args.identity = (f"{args.dataset}-"+
                     f"{args.model}-"+
                     f"{args.num_layer}-"+
                     f"{args.embed_dim}-"+
                     f"{args.norm_type}-"+
                     f"{args.pool_type}-"+
                     f"{args.batch_size}-"+
                     f"{args.lr_warmup_type}-"+
                     f"{args.lr}-"+
                     f"{args.dropout}-"+
                     f"{args.weight_decay}-"+
                     f"{args.loss_type}-"+
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
