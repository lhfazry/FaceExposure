#!/bin/bash
CUDA_VISIBLE_DEVICES=$CUDA_ID python main.py \
    --mode=test \
    --data_dir=datasets/cropped \
    --ckpt_path=$CKPT \
    --batch_size=8 \
    --num_workers=4 \
    --accelerator=gpu \
    --variant=small