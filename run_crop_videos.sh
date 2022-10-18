#!/bin/bash
CUDA_VISIBLE_DEVICES=$CUDA_ID python scripts/crop_videos.py \
    --input_dir=datasets/Exposure \
    --output_dir=datasets/cropped \
    --dim=128