

import os
import torch

def load_logs():
    return 0

def get_batchsize(args):
    if args.dataset in ['ogbg-molmuv']:
        return 512
    elif args.dataset in ['ogbg-molhiv']:
        return 256
    elif args.dataset in ['ogbg-molpcba','ogbg-ppa']:
        return 1024
    else: 
        return 128

### add new arguments
def args_(args): 
    args.batch_size = get_batchsize(args)
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



def get_metric(args):

    if args.dataset in ['ogbg-molhiv','ogbg-molbace',
                        'ogbg-molbbbp','ogbg-molclintox',
                        'ogbg-molsider','ogbg-moltox21',
                        'ogbg-moltoxcast',
                        ]:
        return 'rocauc'
    elif args.dataset in ['ogbg-molpcba','ogbg-molmuv']:
        return 'ap'
    elif args.dataset in ['ogbg-molesol','ogbg-molfreesolv',
                          'ogbg-mollipo']:
        return 'rmse'
    elif args.dataset in ['']:
        return 'acc'
        