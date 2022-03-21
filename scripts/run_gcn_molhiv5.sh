#!/usr/bin/env bash

set -e


device=0
dataset='ogbg-molhiv'
model='GCN'
nlayer=4
lr_warmup_type='None'
epochs=350
seed=0
wd=0.0

for lr in 1e-3;do
for activation in 'relu';do
for dropout in 0.5;do

    python main.py \
            --device $device \
            --dataset $dataset \
            --model $model \
            --epochs $epochs \
            --num_layer $nlayer \
            --norm_type 'xn5' \
            --activation $activation \
            --dropout $dropout \
            --lr_warmup_type $lr_warmup_type \
            --lr $lr \
            --seed $seed \
            --weight_decay $wd

done
done
done

