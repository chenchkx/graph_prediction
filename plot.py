

import torch
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils_plot import args_, get_metric

dir_path = os.path.dirname(__file__)

### add arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--datadir", type=str, default='/nfs4-p1/ckx/datasets/ogb/graph/')
parser.add_argument("--dataset", type=str, default='ogbg-molbbbp')

parser.add_argument("--model", type=str, default='GCN', choices='GIN, GCN')
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--epoch_slice", type=int, default=0)
parser.add_argument("--num_layer", type=int, default=5)
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--norm_type", type=str, default='None', choices=['None', 'bn', 'gn', 'mn'])
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--loss_type", type=str, default='ogb', choices='ogb, bce, mce', 
                    help='ogb: the loss and metric are consistent with those in ogb paper')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--runs", type=int, default=0, 
                    help='running times of the program')

parser.add_argument("--logs_perf_dir", type=str, default=os.path.join(dir_path,'logs_perf'), 
                    help="logs' files of the loss and performance")
parser.add_argument("--logs_stas_dir", type=str, default=os.path.join(dir_path,'logs_stas'), 
                    help="statistics' files of the avg and std")
parser.add_argument("--state_dict", action="store_true")

args = parser.parse_args()
args = args_(args)
if not os.path.exists(args.perf_imgs_dir):
    os.mkdir(args.perf_imgs_dir)


curve_set = 'test'
curve_metric = 'loss1' # loss or metric
if curve_metric != 'loss':
    curve_metric = get_metric(args)
### 'train-loss' 'train-rocauc'  'train-ap'
# 'valid-loss' 'valid-rocauc'  'valid-ap'
# 'test-loss' 'test-rocauc'  'test-ap'
metric_selected = (f"{curve_set}-"+ f"{curve_metric}")


## 
args.norm_type = 'None'
args = args_(args)
xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
logs_table = pd.read_excel(xlsx_path)
logs_epochs = logs_table[metric_selected]
plt.plot(range(len(logs_epochs)), logs_epochs, label='None')

### 
# args.norm_type = 'bn'
# args = args_(args)
# xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
# logs_table = pd.read_excel(xlsx_path)
# logs_epochs = logs_table[metric_selected]
# plt.plot(range(len(logs_epochs)), logs_epochs, label='bn')


# args.norm_type = 'gn'
# args = args_(args)
# xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
# logs_table = pd.read_excel(xlsx_path)
# logs_epochs = logs_table[metric_selected]
# plt.plot(range(len(logs_epochs)), logs_epochs, label='gn')


# args.norm_type = 'mn'
# args = args_(args)
# xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
# logs_table = pd.read_excel(xlsx_path)
# logs_epochs = logs_table[metric_selected]
# plt.plot(range(len(logs_epochs)), logs_epochs, label='mn')


args.norm_type = 'mix'
args = args_(args)
plt.legend()
plt.show()
plt.savefig(os.path.join(args.perf_imgs_dir, args.identity + f'-{metric_selected}.png'))
plt.close()




