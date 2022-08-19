# Overview
This is the repository for synthetic-METPET.

The code has been modified and implemented based on the code for pix2pix shown in [the GitHub repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
Therefore, please refer to the repository for the details of pix2pix.

# Directory tree
Directories should be constructed as follows.
```
dataset_root
 ├── train
 │   ├── cropped
 │   └── whole
 ├── val
 │   ├── cropped
 │   └── whole
 └── test
     └── test_image_dir
```
- `cropped`: the directory of disease-cropped images.
- `whole`: the directory of full images.
- `test_image_dir`: the directory of test images.

# Brief Usage
## Training phase

```bash
#!./scripts/train_pix2pix.sh

python train.py \
        --dataroot (path/to/dataset) \
        --name (checkpoint_name) \
        --model pix2pix \
        --batch_size 64 \
        --n_epochs 100 \
        --n_epochs_decay 100 \
        --preprocess resize_and_crop \
        --dataset_mode mix_aligned \
        --mix_ratio 6:4 \
        --gpu_ids 0,1

```

- `mix_ratio`: the ratio of disease-cropped image pairs to full image pairs.


## Testing phase
```bash
#!./scripts/test_pix2pix.sh

python test.py \
        --dataroot (path/to/dataset) \
        --name (checkpoint_name) \
        --model pix2pix \
        --preprocess none \
        --dataset_mode aligned \
        --split test \
        --specified_dir (test_image_directory_name) \
        --eval \
        --num_test (the_number_of_test_images) \
        --results_dir ./results \
        --gpu_ids 0,1

```


# CUDA VERSION
CUDA Version = 11.4

# Citation
The manuscript containing this repository is been submitting.
After the peer-reviwing process, we will cite the paper here.

# Lisence of pix2pix
For pix2pix software
Copyright (c) 2016, Phillip Isola and Jun-Yan Zhu
All rights reserved.
