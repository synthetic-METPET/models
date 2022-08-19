#!/usr/bin/env bash
set -eu

data_root="path/to/dataset"
name="checkpoints_name"

python train.py \
        --dataroot "$data_root" \
        --name "$name" \
        --model pix2pix \
        --batch_size 64 \
        --n_epochs 100 \
        --n_epochs_decay 100 \
        --preprocess resize_and_crop \
        --dataset_mode mix_aligned \
        --mix_ratio 6:4 \
        --gpu_ids 0,1
