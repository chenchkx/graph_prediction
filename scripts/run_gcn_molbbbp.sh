#!/usr/bin/env bash

set -e


device=0
dataset='ogbg-molbbbp'
model='GCN'
nlayer=4
bs=128

for lr in 1e-3;do
    for seed in 0 1;do
       for wd in 0 5e-4 1e-4;do

        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --num_layer $nlayer \
               --norm_type 'bn' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed \
               --weight_decay $wd

        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --num_layer $nlayer \
               --norm_type 'gn' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed \
               --weight_decay $wd

        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --num_layer $nlayer \
               --norm_type 'in' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed \
               --weight_decay $wd

        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --num_layer $nlayer \
               --norm_type 'xn' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed \
               --weight_decay $wd

        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --num_layer $nlayer \
               --norm_type 'None' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed \
               --weight_decay $wd
       done
    done
done


