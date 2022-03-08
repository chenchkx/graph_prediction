#!/usr/bin/env bash

set -e


device=0
dataset='ogbg-molpcba'
model='GIN'
bs=512

for lr in 1e-3 1e-4;do
    for seed in 1 2;do
        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --norm_type 'gn' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed 
               
               
        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --norm_type 'None' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed 

    done
done

