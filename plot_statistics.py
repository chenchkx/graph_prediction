

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
parser.add_argument("--dataset", type=str, default='ogbg-molhiv')

parser.add_argument("--model", type=str, default='GCN', choices='GIN, GCN')
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--epoch_slice", type=int, default=0)
parser.add_argument("--num_layer", type=int, default=5)
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--norm_type", type=str, default='gn', choices=['None', 'bn', 'gn', 'mn'])
parser.add_argument("--pool_type", type=str, default="mean", choices=['dke', 'sum', 'mean', 'max'])
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--loss_type", type=str, default='ogb', choices='ogb, bce, mce', 
                    help='ogb: the loss keep sync with that used in ogb paper')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--runs", type=int, default=0, 
                    help='running times of program')

parser.add_argument("--logs_perf_dir", type=str, default=os.path.join(dir_path,'logs_perf'), 
                    help="logs' files of the loss and performance")
parser.add_argument("--logs_stas_dir", type=str, default=os.path.join(dir_path,'logs_stas'), 
                    help="statistics' files of the avg and std")
parser.add_argument("--state_dict", action="store_true")

args = parser.parse_args()
args = args_(args)
if not os.path.exists(args.stas_imgs_dir):
    os.mkdir(args.stas_imgs_dir)



dataset_component = 'valid'
### choose the norm type
args.norm_type = 'ln2'
args = args_(args)
xlsx_path = os.path.join(args.stas_xlsx_dir, args.identity + f"-{dataset_component}.xlsx")
logs_table = pd.read_excel(xlsx_path)

### plot conv avg and std information
# plot avg
logs_epochs = logs_table['avg_conv_feature']
plt.plot(range(len(logs_epochs)), logs_epochs, label='avg_conv_feature')
logs_epochs = logs_table['min_avg_conv_feature']
plt.plot(range(len(logs_epochs)), logs_epochs, label='min_avg_conv_feature')
logs_epochs = logs_table['max_avg_conv_feature']
plt.plot(range(len(logs_epochs)), logs_epochs, label='max_avg_conv_feature')
plt.legend()
plt.show()
plt.savefig(os.path.join(args.stas_imgs_dir, args.identity + f'-{dataset_component}-avg-conv-stas.png'))
plt.close()

# plot std
logs_epochs = logs_table['std_conv_feature']
plt.plot(range(len(logs_epochs)), logs_epochs, label='std_conv_feature')
logs_epochs = logs_table['min_std_conv_feature']
plt.plot(range(len(logs_epochs)), logs_epochs, label='min_std_conv_feature')
logs_epochs = logs_table['max_std_conv_feature']
plt.plot(range(len(logs_epochs)), logs_epochs, label='max_std_conv_feature')
plt.legend()
plt.show()
plt.savefig(os.path.join(args.stas_imgs_dir, args.identity + f'-{dataset_component}-std-conv-stas.png'))
plt.close()


### plot norm avg and std information
# plot avg
logs_epochs = logs_table['avg_norm_feature']
plt.plot(range(len(logs_epochs)), logs_epochs, label='avg_norm_feature')
logs_epochs = logs_table['min_avg_norm_feature']
plt.plot(range(len(logs_epochs)), logs_epochs, label='min_avg_norm_feature')
logs_epochs = logs_table['max_avg_norm_feature']
plt.plot(range(len(logs_epochs)), logs_epochs, label='max_avg_norm_feature')
plt.legend()
plt.show()
plt.savefig(os.path.join(args.stas_imgs_dir, args.identity + f'-{dataset_component}-avg-norm-stas.png'))
plt.close()

# plot std
logs_epochs = logs_table['std_norm_feature']
plt.plot(range(len(logs_epochs)), logs_epochs, label='std_norm_feature')
logs_epochs = logs_table['min_std_norm_feature']
plt.plot(range(len(logs_epochs)), logs_epochs, label='min_std_norm_feature')
logs_epochs = logs_table['max_std_norm_feature']
plt.plot(range(len(logs_epochs)), logs_epochs, label='max_std_norm_feature')
plt.legend()
plt.show()
plt.savefig(os.path.join(args.stas_imgs_dir, args.identity + f'-{dataset_component}-std-norm-stas.png'))
plt.close()