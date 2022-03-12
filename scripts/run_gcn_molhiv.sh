#!/usr/bin/env bash

set -e


device=0
dataset='ogbg-molhiv'
model='GCN'
bs=128
nlayer=4

for lr in 1e-3;do
    for seed in 0;do
       for wd in 0;do

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

