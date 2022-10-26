from vidaug import augmentors as va
import random
import os
import pathlib
import collections

import numpy as np
import torch
import cv2
from pathlib import Path


def loadvideo(filename: str, frame_dim):
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    #frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    #frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_dim, frame_dim, 3), np.uint8) # (F, W, H, C)

    for count in range(frame_count):
        ret, frame = capture.read()
        
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_dim, frame_dim))
        v[count] = frame

        count += 1

    # capture.release()
    #v = v.transpose((3, 0, 1, 2)) #(C, F, H, W)

    assert v.size > 0

    return v, fps # (F, W, H, C)

def save_video(name, video, fps):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    data = cv2.VideoWriter(name, fourcc, float(fps), (video.shape[1], video.shape[2]))

    for v in video:
        #v = cv2.cvtColor(v, cv2.COLOR_RGB2BGR)
        data.write(v)

    data.release()

dataset_dir = 'datasets'
video_dir = os.path.join(dataset_dir, 'cropped')
files = os.listdir(video_dir)

vid_augs1 = va.Sequential([va.RandomRotate(degrees=10)])
vid_augs2 = va.Sequential([va.HorizontalFlip()]) 
vid_augs3 = va.Sequential([va.VerticalFlip()])  
vid_augs4 = va.Sequential([va.GaussianBlur(0.75)])

vid_augs = va.Sequential([
    va.RandomRotate(degrees=5), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
    va.HorizontalFlip(), # horizontally flip the video with 50% probability
    va.VerticalFlip(),
    va.GaussianBlur(random.random()),
])

for i in range(10):
    path_file = Path(os.path.join(video_dir, files[i]))
    video, fps = loadvideo(os.path.join(video_dir, files[i]), 128)#.astype(np.float32)
    print(f"video shape: {video.shape}")
    auged = np.asarray(vid_augs(video))#.astype(np.uint8)
    print(f"auged shape: {auged.shape}")
    save_video(os.path.join(dataset_dir, f"{path_file.stem}.avi"), auged, fps)

