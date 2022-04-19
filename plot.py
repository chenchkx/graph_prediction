

import torch
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils_plot import args_, get_metric

dir_path = os.path.dirname(__file__)

### add arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--datadir", type=str, default='datasets')
parser.add_argument("--dataset", type=str, default='ogbg-molhiv')

parser.add_argument("--model", type=str, default='GCN', choices='GIN, GCN')
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--epoch_slice", type=int, default=0)
parser.add_argument("--num_layer", type=int, default=4)
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--pool_type", type=str, default="mean", choices=['dke', 'sum', 'mean', 'max'])
parser.add_argument("--norm_type", type=str, default='xn3', choices=['bn', 'gn', 'xn', 'xn2', 'xn3', 'xn4', 'None'])
parser.add_argument("--activation", type=str, default='relu', choices=['relu', 'None'])
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--lr_warmup_type", type=str, default='None', choices=['step','cosine','linear','None'])
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--loss_type", type=str, default='ogb', choices=['ogb', 'bce', 'mce'], 
                    help='ogb: the loss and metric are consistent with those in ogb paper')
parser.add_argument("--batch_size", type=int, default=128)
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
# args.weight_decay = 1e-4

# ## 
# args.norm_type = 'None'
# args = args_(args)
# xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
# logs_table = pd.read_excel(xlsx_path)
# logs_epochs = logs_table[metric_selected][0:args.epochs]
# plt.plot(range(len(logs_epochs)), logs_epochs, label='None')

###
args.norm_type = 'bn'
args = args_(args)
xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
logs_table = pd.read_excel(xlsx_path)
logs_epochs = logs_table[metric_selected][0:args.epochs]
plt.plot(range(len(logs_epochs)), logs_epochs, label='bn')


args.norm_type = 'bnf'
args = args_(args)
xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
logs_table = pd.read_excel(xlsx_path)
logs_epochs = logs_table[metric_selected][0:args.epochs]
plt.plot(range(len(logs_epochs)), logs_epochs, label='bnf')


# args.norm_type = 'bnm'
# args = args_(args)
# xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
# logs_table = pd.read_excel(xlsx_path)
# logs_epochs = logs_table[metric_selected][0:args.epochs]
# plt.plot(range(len(logs_epochs)), logs_epochs, label='bnm')

### 
# args.norm_type = 'gn'
# args = args_(args)
# xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
# logs_table = pd.read_excel(xlsx_path)
# logs_epochs = logs_table[metric_selected][0:args.epochs]
# plt.plot(range(len(logs_epochs)), logs_epochs, label='gn')

# ###
# args.norm_type = 'in'
# args = args_(args)
# xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
# logs_table = pd.read_excel(xlsx_path)
# logs_epochs = logs_table[metric_selected][0:args.epochs]
# plt.plot(range(len(logs_epochs)), logs_epochs, label='in')



# ###
# args.norm_type = 'xn'
# args = args_(args)
# xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
# logs_table = pd.read_excel(xlsx_path)
# logs_epochs = logs_table[metric_selected][0:args.epochs]
# plt.plot(range(len(logs_epochs)), logs_epochs, label='xn')


#
args.norm_type = 'xn1'
args = args_(args)
xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
logs_table = pd.read_excel(xlsx_path)
logs_epochs = logs_table[metric_selected][0:args.epochs]
plt.plot(range(len(logs_epochs)), logs_epochs, label='xn1')

args.norm_type = 'xn2'
args = args_(args)
xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
logs_table = pd.read_excel(xlsx_path)
logs_epochs = logs_table[metric_selected][0:args.epochs]
plt.plot(range(len(logs_epochs)), logs_epochs, label='xn2')

# args.norm_type = 'xn3'
# args = args_(args)
# xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
# logs_table = pd.read_excel(xlsx_path)
# logs_epochs = logs_table[metric_selected][0:args.epochs]
# plt.plot(range(len(logs_epochs)), logs_epochs, label='xn3')

# args.norm_type = 'xn4'
# args = args_(args)
# xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
# logs_table = pd.read_excel(xlsx_path)
# logs_epochs = logs_table[metric_selected][0:args.epochs]
# plt.plot(range(len(logs_epochs)), logs_epochs, label='xn4')


args.norm_type = 'xn5'
args = args_(args)
xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
logs_table = pd.read_excel(xlsx_path)
logs_epochs = logs_table[metric_selected][0:args.epochs]
plt.plot(range(len(logs_epochs)), logs_epochs, label='xn5')

# args.norm_type = 'xn6'
# args = args_(args)
# xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
# logs_table = pd.read_excel(xlsx_path)
# logs_epochs = logs_table[metric_selected][0:args.epochs]
# plt.plot(range(len(logs_epochs)), logs_epochs, label='xn6')

# args.norm_type = 'xn7'
# args = args_(args)
# xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
# logs_table = pd.read_excel(xlsx_path)
# logs_epochs = logs_table[metric_selected][0:args.epochs]
# plt.plot(range(len(logs_epochs)), logs_epochs, label='xn7')

# args.norm_type = 'xn8'
# args = args_(args)
# xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
# logs_table = pd.read_excel(xlsx_path)
# logs_epochs = logs_table[metric_selected][0:args.epochs]
# plt.plot(range(len(logs_epochs)), logs_epochs, label='xn8')


# args.norm_type = 'xn9'
# args = args_(args)
# xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
# logs_table = pd.read_excel(xlsx_path)
# logs_epochs = logs_table[metric_selected][0:args.epochs]
# plt.plot(range(len(logs_epochs)), logs_epochs, label='xn9')

# args.norm_type = 'xn10'
# args = args_(args)
# xlsx_path = os.path.join(args.perf_xlsx_dir, args.identity + ".xlsx")
# logs_table = pd.read_excel(xlsx_path)
# logs_epochs = logs_table[metric_selected][0:args.epochs]
# plt.plot(range(len(logs_epochs)), logs_epochs, label='xn10')


args.norm_type = 'mix'
args = args_(args)
plt.legend()
plt.show()
plt.savefig(os.path.join(args.perf_imgs_dir, args.identity + f'-{metric_selected}.png'))
plt.close()




