#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_dir=/workspace/Exposure \
    --pretrained=pretrained/swin_small_patch4_window7_224_22k.pth \
    --batch_size=2 \
    --num_workers=4 \
    --accelerator=gpu \
    --variant=small \
    --max_epoch=20