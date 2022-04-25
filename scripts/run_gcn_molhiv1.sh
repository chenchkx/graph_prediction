#!/usr/bin/env bash

set -e


device=3
dataset='ogbg-molhiv'
model='GCN'
epochs=500
nlayer=50
norm_type='xn1'
activation='relu'
dropout=0.5
lr_warmup_type='linear'
seed=0

for lr in 1e-3 1e-2;do
for seed in 0;do
for wd in 0.0;do
    python main.py \
            --device $device \
            --dataset $dataset \
            --model $model \
            --epochs $epochs \
            --num_layer $nlayer \
            --norm_type $norm_type \
            --activation $activation \
            --dropout $dropout \
            --lr_warmup_type $lr_warmup_type \
            --lr $lr \
            --seed $seed \
            --weight_decay $wd\
            --norm_affine\
            --node_weight
done
done
done

