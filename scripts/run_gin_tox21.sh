#!/usr/bin/env bash

set -e


device=1
dataset='ogbg-moltox21'
model='GIN'
bs=128

for lr in 1e-3 1e-4;do
    for seed in 1 2;do
        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --norm_type 'None' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed 

        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --norm_type 'bn' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed

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
               --norm_type 'mn' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed 
    done
done

