
import os
import torch
import argparse
from utils.utils import *
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

dir_path = os.path.dirname(__file__)

def main(args):

    dataset, train_loader, valid_loader, test_loader = load_data(args)
    args = args_(args, dataset)
    set_seed(args)

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
    parser.add_argument("--datadir", type=str, default='/mnt/nfs/ckx/datasets/ogb/graph/')
    parser.add_argument("--dataset", type=str, default='ogbg-molbbbp')

    parser.add_argument("--model", type=str, default='GCN', choices='GIN, GCN')
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--epoch_slice", type=int, default=0)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--norm_type", type=str, default='bn', choices=['bn', 'gn', 'None', 'mn'])
    parser.add_argument("--pool_type", type=str, default="dke", choices=['dke', 'sum', 'mean', 'max'])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss_type", type=str, default='ogb', choices='ogb, bce, mce', 
                        help='ogb: the loss and metric are consistent with those in ogb paper')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=0, 
                        help='running times of the program')
    
    parser.add_argument("--logs_perf_dir", type=str, default=os.path.join(dir_path,'logs_perf'), 
                        help="logs' files of the loss and performance")
    parser.add_argument("--logs_stas_dir", type=str, default=os.path.join(dir_path,'logs_stas'), 
                        help="statistics' files of the avg and std")
    parser.add_argument("--state_dict", action="store_true")

    args = parser.parse_args()
    
    main(args)



