import os
import pathlib
import collections

import numpy as np
import torch

from torch.nn.functional import one_hot
import torch.utils.data
import cv2  # pytype: disable=attribute-error
from vidaug import augmentors as va
import random
import pandas as pd

class ExposureDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", frame_dim=128, augmented=False, max_frames = 256):

        self.folder = pathlib.Path(root)
        self.augmented = augmented
        self.max_frames = max_frames
        self.frame_dim = frame_dim

        if not os.path.exists(root):
            raise ValueError("Path does not exist: " + root)

        df = pd.read_csv(os.path.join(root, "image20_exposure.csv"))
        self.df = df[df["split"] == split]

        #print(df.columns)

        #self.df = df.astype({
        #    "neutral": float, 
        #    "happy": float, 
        #    "sad": float,
        #    "contempt": float, 
        #    "anger": float, 
        #    "disgust": float, 
        #    "surprised": float, 
        #    "fear": float
        #})

        self.vid_augs = va.Sequential([
            #va.RandomCrop(size=(240, 180)), # randomly crop video with a size of (240 x 180)
            va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
            va.HorizontalFlip(), # horizontally flip the video with 50% probability
            va.VerticalFlip(),
            va.GaussianBlur(random.random())
        ])
            
    def __getitem__(self, index):
        row = self.df.iloc[index].to_dict()
        path = os.path.join(self.folder, row["video_name"])
        #print(f"Load video from: {path}")

        # Load video into np.array
        video = loadvideo(path, self.frame_dim).astype(np.float32) / 255.
        #key = os.path.splitext(self.fnames[index])[0]

        video = np.moveaxis(video, 0, 1) #(F, C, H, W)
        F, C, H, W = video.shape
        sampling_rate = 1

        if F > 1024:
            sampling_rate = 4
        elif F > 768:
            sampling_rate = 3
        elif F > 512:
            sampling_rate = 2

        video = video[::sampling_rate,:,:,:]

        if video.shape[0] > self.max_frames:
            video = video[:self.max_frames,:,:,:]
        elif video.shape[0] < self.max_frames:
            pads = np.zeros((self.max_frames - F, H, W, 3))
            video = np.concatenate(video, pads)

        assert video.shape[0] == self.max_frames

        #print(f'before video size: {nvideo.shape}')
        if self.augmented:
            # (3, 0, 1, 2) ==> (0, 1, 2, 3)
            # (F, C, H, W)
            vid = video.transpose((0, 2, 3, 1)) # (F, H, W, C) 
            vid = np.asarray(self.vid_augs(vid)) # (F, H, W, C)
            vid = vid.transpose((0, 3, 1, 2)) # (F, C, H, W)
        
        #saved_video = nvideo.transpose((0, 2, 3, 1))
        #print(f'after video size: {saved_video.shape}: {filename}')
        #save_video(filename + ".avi", np.asarray(saved_video).astype(np.uint8), 50)

        label = np.array([
            row["neutral"],
            row["happy"],
            row["sad"],
            row["contempt"],
            row["anger"],
            row["disgust"],
            row["surprised"],
            row["fear"]
        ])

        #row['video'] = video 
        #row["neutral"] = one_hot(torch.tensor(row["neutral"]), num_classes=2)
        #row["happy"] = one_hot(torch.tensor(row["happy"]), num_classes=2)
        #row["sad"] = one_hot(torch.tensor(row["sad"]), num_classes=2)
        #row["contempt"] = one_hot(torch.tensor(row["contempt"]), num_classes=2)
        #row["anger"] = one_hot(torch.tensor(row["anger"]), num_classes=2)
        #row["disgust"] = one_hot(torch.tensor(row["disgust"]), num_classes=2)
        #row["surprised"] = one_hot(torch.tensor(row["surprised"]), num_classes=2)
        #row["fear"] = one_hot(torch.tensor(row["fear"]), num_classes=2)

        #print(f"neutral tensor: {row['neutral']}")
        return {'video': video, 'label': label}
            
    def __len__(self):
        return len(self.df)

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

    capture.release()
    v = v.transpose((3, 0, 1, 2)) #(C, F, H, W)

    assert v.size > 0

    return v

def save_video(name, video, fps):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    data = cv2.VideoWriter(name, fourcc, float(fps), (video.shape[1], video.shape[2]))

    for v in video:
        data.write(v)

    data.release()
