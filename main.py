
import os
import torch
import argparse
from utils.utils import *
from xmlrpc.client import boolean
from warnings import simplefilter

torch.set_num_threads(10)
dir_path = os.path.dirname(__file__)
simplefilter(action='ignore', category=FutureWarning)

def main(args):
    set_seed(args)

    dataset, train_loader, valid_loader, test_loader = load_process_dataset(args)
    args = args_(args, dataset)

    model = load_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  
    modelOptm = ModelOptLoading(model=model, 
                                optimizer=optimizer,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                test_loader=test_loader,
                                args=args)
    modelOptm.optimizing()
    metric_list = ['train-loss','train-rocauc', 'valid-rocauc', 'test-rocauc']
    print_best_log(args, eopch_slice=args.epoch_slice)

    # plot_logs(args, metric_list)

    print('optmi')

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--datadir", type=str, default='datasets')
    parser.add_argument("--dataset", type=str, default='ogbg-molbbbp')

    parser.add_argument("--model", type=str, default='GCN', choices='GIN, GCN')
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--epoch_slice", type=int, default=0)
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--pool_type", type=str, default="mean", choices=['dke', 'mean', 'sum', 'max'])
    parser.add_argument("--norm_type", type=str, default='xn6')
    parser.add_argument("--activation", type=str, default='relu', choices=['relu', 'None'])
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr_warmup_type", type=str, default='cosine', choices=['step','cosine','linear','None'])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--loss_type", type=str, default='ogb', choices=['ogb', 'mce',  'bce'], 
                        help='ogb: the loss and metric are consistent with those in ogb paper')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=0, 
                        help='running times of the program')
    
    parser.add_argument("--logs_perf_dir", type=str, default=os.path.join(dir_path,'logs_perf'), 
                        help="logs' files of the loss and performance")
    parser.add_argument("--logs_stas_dir", type=str, default=os.path.join(dir_path,'logs_stas'), 
                        help="statistics' files of the avg and std")                        
    parser.add_argument("--norm_affine", action="store_true")
    parser.add_argument("--node_weight", default=True)
    parser.add_argument("--state_dict", action="store_true")

    args = parser.parse_args()
    
    main(args)



