import cv2
import os
import numpy as np
import argparse
from pathlib import Path
from deepface import DeepFace
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default=None, help="Input directory")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
parser.add_argument("--dim", type=int, default=None, help="Spatial dimension")

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

    capture.release()
    #v = v.transpose((3, 0, 1, 2)) #(C, F, H, W)

    assert v.size > 0

    return fps, v

def save_video(name, video, fps):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    data = cv2.VideoWriter(name, fourcc, float(fps), (video.shape[1], video.shape[2]))

    for v in video:
        data.write(v)

    data.release()

def crop_videos(input_dir, output_dir, dim):
    videos = glob(os.path.join(input_dir, '*.mp4'))

    for video in videos:
        filename = Path(video).name
        print(f"Processing: {filename}")
        fps, frames = load_video(video)
        faces = []

        for i in range(frames.shape[0]):
            face = DeepFace.detectFace(img_path = frames[i,:,:,:].squeeze(), 
                target_size = dim, 
                detector_backend = 'retinaface'
            )

            faces.append(face)

        cropped = np.stack(faces, axis=0)
        out_filename = os.path.join(output_dir, filename)
        print(f"Finished. Cropped shape: {cropped.shape}")
        save_video(out_filename, np.stack(faces, axis=0), fps)
        print(f"Saved into: {out_filename}")

if __name__ == '__main__':
    input_dir = params.input_dir
    output_dir = params.output_dir
    dim = params.dim

    crop_videos(input_dir, output_dir, (dim, dim))