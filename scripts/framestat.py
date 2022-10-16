import cv2
import os
import numpy as np
import argparse
import logging
import pandas as pd
from glob import glob
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--video_dir", type=str, default=None, help="Video directory")

params = parser.parse_args()

def load_video(filename: str):
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
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8) # (F, H, W, C)

    for count in range(frame_count):
        ret, frame = capture.read()
        
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.resize(frame, (frame_dim, frame_dim))
        v[count] = frame

        count += 1

    #capture.release()
    #v = v.transpose((3, 0, 1, 2)) #(C, F, H, W)

    assert v.size > 0

    return fps, v

def calc_frame_stat(dir):
    videos = glob(os.path.join(dir, '*.mp4'))
    number_of_frames = []

    for video in videos:
        filename = Path(video).name
        fps, frames = load_video(video)
        number_of_frames.append(frames[0])

    df = pd.DataFrame(number_of_frames)
    df.describe()

if __name__ == '__main__':
    video_dir = params.video_dir

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    calc_frame_stat(video_dir)