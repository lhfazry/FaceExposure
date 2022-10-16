#!/bin/bash
CUDA_VISIBLE_DEVICES=$CUDA_ID python crop_videos.py \
    --input_dir=datasets/Exposure \
    --output_dir=datasets/cropped2 \
    --dim=128