#!/bin/bash
mkdir -p lightning_logs/exposure

CUDA_VISIBLE_DEVICES=$CUDA_ID python main.py \
    --data_dir=datasets/cropped \
    --pretrained=pretrained/swin_base_patch244_window877_kinetics400_22k.pth \
    --batch_size=4 \
    --num_workers=4 \
    --accelerator=gpu \
    --variant=base \
    --max_epoch=100 \
    --upsampling=0 \
    --train_augmentation=0 \
    --max_frames=512 \
    --frame_dim=64 \
    --sampling_strategy=truncate \
    --csv_file=datasets/video_exposure_nonetral.csv