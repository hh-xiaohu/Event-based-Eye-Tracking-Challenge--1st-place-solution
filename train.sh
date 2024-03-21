#!/usr/bin/env bash

batch_size=128
lr=0.002
n_time_bins=4
num_epochs=1000
train_length=45
val_length=45
test_length=45
train_stride=10
val_stride=10
test_stride=10
spatial_factor=0.125
map_type=binary
dataset=t_v 


run_name="gru_mamba_${batch_size}_${lr}_len${train_length}_stride_${train_stride}_sf${spatial_factor}_${dataset}_4"

python3 train.py --batch_size $batch_size --lr $lr --n_time_bins $n_time_bins --num_epochs $num_epochs \
--train_length $train_length --val_length $val_length --test_length $test_length \
--train_stride $train_stride --val_stride $val_stride --test_stride $test_stride \
--spatial_factor $spatial_factor --map_type $map_type --dataset $dataset --run_name $run_name
