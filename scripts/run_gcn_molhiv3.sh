#!/usr/bin/env bash

set -e


device=3
dataset='ogbg-molhiv'
model='GCN'
nlayer=4
lr_warmup_type='None'
epochs=350

for lr in 1e-3 5e-4;do
    for seed in 0;do
       for wd in 0;do

        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --epochs $epochs \
               --num_layer $nlayer \
               --norm_type 'xn' \
               --lr_warmup_type $lr_warmup_type \
               --lr $lr \
               --seed $seed \
               --runs 0 \
               --weight_decay $wd

       done
    done
done

