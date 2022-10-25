#!/bin/bash
mkdir -p lightning_logs/exposure

CUDA_VISIBLE_DEVICES=$CUDA_ID python main.py \
    --data_dir=datasets/cropped \
    --pretrained=pretrained/swin_small_patch4_window7_224_22k.pth \
    --batch_size=4 \
    --num_workers=4 \
    --accelerator=gpu \
    --variant=small \
    --max_epoch=30 \
    --upsampling=1 \
    --sampling_strategy=truncate \
    --csv_file=datasets/video_exposure_nonetral.csv