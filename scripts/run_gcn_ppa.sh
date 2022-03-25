#!/usr/bin/env bash

set -e


device=2
dataset='ogbg-ppa'
model='GCN'
epochs=400
nlayer=4
norm_type='bnf'
activation='relu'
dropout=0.5
lr_warmup_type='step'


for lr in 1e-3;do
for seed in 0;do
for wd in 0;do

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
            --weight_decay $wd

done
done
done