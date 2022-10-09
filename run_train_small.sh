#!/bin/bash
CUDA_VISIBLE_DEVICES=$CUDA_ID python main.py \
    --data_dir=datasets/Exposure \
    --pretrained=pretrained/swin_small_patch4_window7_224_22k.pth \
    --batch_size=4 \
    --num_workers=4 \
    --accelerator=cpu \
    --variant=small \
    --max_epoch=20