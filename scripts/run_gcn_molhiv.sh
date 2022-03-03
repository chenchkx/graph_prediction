#!/usr/bin/env bash

set -e


device=0
dataset='ogbg-molhiv'
model='GCN'
bs=128

for lr in 1e-3;do
    for seed in 0;do
       for wd in 0 5e-4 5e-5;do
       #  python main.py \
       #         --device $device \
       #         --dataset $dataset \
       #         --model $model \
       #         --norm_type 'None' \
       #         --batch_size $bs \
       #         --lr $lr \
       #         --seed $seed \
       #         --weight_decay $wd

       #  python main.py \
       #         --device $device \
       #         --dataset $dataset \
       #         --model $model \
       #         --norm_type 'gn' \
       #         --batch_size $bs \
       #         --lr $lr \
       #         --seed $seed \
       #         --weight_decay $wd

        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --norm_type 'bn' \
               --pool_type 'mean' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed \
               --weight_decay $wd

        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --norm_type 'bn' \
               --pool_type 'sum' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed \
               --weight_decay $wd

        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --norm_type 'bn' \
               --pool_type 'dke' \
               --batch_size $bs \
               --lr $lr \
               --seed $seed \
               --weight_decay $wd
               
       #  python main.py \
       #         --device $device \
       #         --dataset $dataset \
       #         --model $model \
       #         --norm_type 'mn' \
       #         --batch_size $bs \
       #         --lr $lr \
       #         --seed $seed \
       #         --weight_decay $wd
       done
    done
done

