#!/usr/bin/env bash

set -e


device=1
dataset='ogbg-molmuv'
model='GCN'
epochs=500
nlayer=4
norm_type='xn2'
activation='relu'
dropout=0.5
lr_warmup_type='cosine'
seed=0

for lr in 1e-3 1e-4;do
for nlayer in 4 10;do
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

