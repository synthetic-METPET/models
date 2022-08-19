#!/usr/bin/env bash
set -eu

data_root="path/to/dataset"
name="checkpoints_name"
test_image_dir="directory_name_where_test_images_stored"
num_test=$(find "$data_root/test/$test_image_dir" -type f | wc -l)

python test.py \
        --dataroot "$data_root" \
        --name "$name" \
        --model pix2pix \
        --preprocess none \
        --dataset_mode aligned \
        --split test \
        --specified_dir "$test_image_dir" \
        --eval \
        --num_test "$num_test"
        --results_dir ./results \
        --gpu_ids 0,1
