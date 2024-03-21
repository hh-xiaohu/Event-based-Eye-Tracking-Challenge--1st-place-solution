#!/usr/bin/env bash

test_length=45
test_stride=10
spatial_factor=0.125
map_type=binary
# checkpoint=/media/hanh/mix/eye_track/eye_baseline/trainset_only/model_last_epoch1000.pth
checkpoint=/media/hanh/mix/eye_track/eye_baseline/trainset_validset/model_last_epoch1000.pth
# run_name="gru_mamba_${batch_size}_${lr}_len${train_length}_stride_${train_stride}_sf${spatial_factor}_${dataset}"

python3 test.py --test_length $test_length --test_stride $test_stride \
--spatial_factor $spatial_factor --map_type $map_type --checkpoint $checkpoint