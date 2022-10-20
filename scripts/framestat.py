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

def get_frame_count(filename: str):
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
    capture.release()
    return frame_count

def calc_frame_stat(dir):
    videos = glob(os.path.join(dir, '*.mp4'))
    number_of_frames = []

    for video in videos:
        filename = Path(video).name
        frame_count = get_frame_count(video)

        #logging.info(f"{filename} : {frame_count}")

        number_of_frames.append(frame_count)

    df = pd.DataFrame(number_of_frames)
    logging.info(df.describe())

if __name__ == '__main__':
    video_dir = params.video_dir

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    calc_frame_stat(video_dir)