#!/usr/bin/env bash

set -e


device=2
dataset='ogbg-molhiv'
model='GCN'
epochs=500
nlayer=10
norm_type='bnf'
activation='relu'
dropout=0.5
lr_warmup_type='None'
seed=0

# for lr in 1e-3;do
# for seed in 0;do
# for wd in 0.0 1e-4;do

#     python main.py \
#             --device $device \
#             --dataset $dataset \
#             --model $model \
#             --epochs $epochs \
#             --num_layer $nlayer \
#             --norm_type $norm_type \
#             --activation $activation \
#             --dropout $dropout \
#             --lr_warmup_type $lr_warmup_type \
#             --lr $lr \
#             --seed $seed \
#             --weight_decay $wd

# done
# done
# done




for lr in 1e-5;do
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


# norm_type='bnm'
# for lr in 1e-3 1e-4;do
# for nlayer in 4 10;do
# for wd in 1e-4;do

#     python main.py \
#             --device $device \
#             --dataset $dataset \
#             --model $model \
#             --epochs $epochs \
#             --num_layer $nlayer \
#             --norm_type $norm_type \
#             --activation $activation \
#             --dropout $dropout \
#             --lr_warmup_type $lr_warmup_type \
#             --lr $lr \
#             --seed $seed \
#             --weight_decay $wd

# done
# done
# done
