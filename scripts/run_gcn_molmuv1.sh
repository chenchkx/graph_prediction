#!/usr/bin/env bash

set -e


device=3
dataset='ogbg-molmuv'
model='GCN'
bs=256
nlayer=4
lr_warmup_type='None'

for lr in 1e-3 5e-4;do
    for seed in 0;do
       for wd in 0;do

        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --num_layer $nlayer \
               --norm_type 'xn' \
               --batch_size $bs \
               --lr_warmup_type $lr_warmup_type \
               --lr $lr \
               --seed $seed \
               --weight_decay $wd

       done
    done
done

