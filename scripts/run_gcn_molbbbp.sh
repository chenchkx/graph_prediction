#!/usr/bin/env bash

set -e


device=1
dataset='ogbg-molbbbp'
model='GCN'
bs=64

for lr in 1e-3;do
    for seed in 0;do
       for wd in 5e-4;do
        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --norm_type 'None' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed \
               --weight_decay $wd

        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --norm_type 'gn' \
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
               
        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --norm_type 'mn' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed \
               --weight_decay $wd
       done
    done
done


