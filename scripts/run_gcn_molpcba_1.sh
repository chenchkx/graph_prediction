#!/usr/bin/env bash

set -e


device=2
dataset='ogbg-molpcba'
model='GCN'
bs=256

for lr in 1e-3 1e-4;do
    for seed in 0;do
       for wd in 0 5e-4 5e-5;do

        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --norm_type 'mn' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed \
               --weight_decay $wd
               
        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --norm_type 'bn' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed \
               --weight_decay $wd
       done
    done
done
