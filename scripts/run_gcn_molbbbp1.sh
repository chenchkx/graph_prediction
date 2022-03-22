#!/usr/bin/env bash

set -e


device=1
dataset='ogbg-molbbbp'
model='GCN'
epochs=500
nlayer=4
norm_type='xn1'
activation='relu'
dropout=0.5
lr_warmup_type='cosine'
wd=0.0


for lr in 1e-3;do
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
            --weight_decay $wd

done
done
done

