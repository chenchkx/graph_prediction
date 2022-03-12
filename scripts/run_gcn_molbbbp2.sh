#!/usr/bin/env bash

set -e


device=0
dataset='ogbg-molbbbp'
model='GCN'
nlayer=4
bs=128

for lr in 1e-3 5e-4;do
    for seed in 0;do
       for wd in 0 5e-4 1e-4;do

        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --num_layer $nlayer \
               --norm_type 'bn' \
               --pool_type 'mean'\
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
               --pool_type 'mean'\
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
               --pool_type 'mean'\
               --batch_size $bs \
               --lr $lr \
               --seed $seed \
               --weight_decay $wd


        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --num_layer $nlayer \
               --norm_type 'ln' \
               --pool_type 'mean'\
               --batch_size $bs \
               --lr $lr \
               --seed $seed \
               --weight_decay $wd


        python main.py \
               --device $device \
               --dataset $dataset \
               --model $model \
               --num_layer $nlayer \
               --norm_type 'ln2' \
               --pool_type 'mean'\
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
               --pool_type 'mean'\
               --batch_size $bs \
               --lr $lr \
               --seed $seed \
               --weight_decay $wd


        # python main.py \
        #        --device $device \
        #        --dataset $dataset \
        #        --model $model \
        #        --norm_type 'bn' \
        #        --pool_type 'mean' \
        #        --batch_size $bs \
        #        --lr $lr \
        #        --seed $seed \
        #        --weight_decay $wd

       #  python main.py \
       #         --device $device \
       #         --dataset $dataset \
       #         --model $model \
       #         --norm_type 'bn' \
       #         --pool_type 'sum' \
       #         --batch_size $bs \
       #         --lr $lr \
       #         --seed $seed \
       #         --weight_decay $wd


       #  python main.py \
       #         --device $device \
       #         --dataset $dataset \
       #         --model $model \
       #         --norm_type 'bn' \
       #         --pool_type 'dke' \
       #         --batch_size $bs \
       #         --lr $lr \
       #         --seed $seed \
       #         --weight_decay $wd


       done
    done
done


