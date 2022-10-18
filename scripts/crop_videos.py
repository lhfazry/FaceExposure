import cv2
import os
import numpy as np
import argparse
import logging
from pathlib import Path
from deepface import DeepFace
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default=None, help="Input directory")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
parser.add_argument("--detector", type=str, default='ssd', help="Backend detector")
parser.add_argument("--dim", type=int, default=128, help="Spatial dimension")

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

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.resize(frame, (frame_dim, frame_dim))
        #frame = image_resize(frame, height = image_size)
        v[count] = frame

        count += 1

    capture.release()
    #v = v.transpose((3, 0, 1, 2)) #(C, F, H, W)

    assert v.size > 0

    return fps, v

def save_video(name, video, fps, convert_to_bgr = True):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    data = cv2.VideoWriter(name, fourcc, float(fps), (video.shape[1], video.shape[2]))

    for frame in video:
        if convert_to_bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        data.write(frame)

    data.release()

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def image_resize2(img, target_size):
    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)

        diff_0 = target_size[0] - img.shape[0]  
        diff_1 = target_size[1] - img.shape[1]
        
        img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')

    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)
        
    return img

def crop_videos(input_dir, output_dir, detector = 'ssd', dim = (128, 128)):
    videos = glob(os.path.join(input_dir, '*.mp4'))

    for idx, video in enumerate(videos):
        filename = Path(video).name
        out_filename = os.path.join(output_dir, filename)

        if os.path.exists(out_filename):
            logging.info(f"File {filename} already cropped. Skipping")
            continue

        fps, frames = load_video(video)
        logging.info(f"Processing {idx + 1} of {len(videos)}: {filename}, shape: {frames.shape}")

        faces = []

        for i in range(frames.shape[0]):
            try:
                face = DeepFace.detectFace(img_path = image_resize(frames[i,:,:,:].squeeze(), height=256), 
                    target_size = dim, 
                    detector_backend = detector,
                    align = False
                )

                faces.append((face * 255).astype(np.uint8))
            except:
                pass
                #logging.info(f"No face detected on frame: {i}. Skipping")

        cropped = np.stack(faces, axis=0)
        logging.info(f"Finished. Cropped shape: {cropped.shape}. Total removed frames: {len(frames) - len(faces)}")
        save_video(out_filename, np.stack(faces, axis=0), fps)
        logging.info(f"Saved into: {out_filename}")

def crop_videos2(input_dir, output_dir, dim):
    face_cascade = cv2.CascadeClassifier('face_detection/haarcascade_frontalface_default.xml')
    videos = glob(os.path.join(input_dir, '*.mp4'))

    for idx, video in enumerate(videos):
        filename = Path(video).name
        out_filename = os.path.join(output_dir, filename)

        if os.path.exists(out_filename):
            logging.info(f"File {filename} already cropped. Skipping")
            continue

        fps, frames = load_video(video)
        logging.info(f"Processing {idx + 1} of {len(videos)}: {filename}, shape: {frames.shape}")

        faces = []

        for i in range(frames.shape[0]):
            frame = image_resize(frames[i,:,:,:].squeeze(), height=256)#frames[i,:,:,:].squeeze()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces  = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(detected_faces) > 0:
                x, y, w, h = detected_faces[0]
                cropped_frame = frame[y:y+h, x:x+w]

                #logging.info(f"Cropped frame shape: {cropped_frame.shape}")
                cropped_frame = image_resize2(cropped_frame, dim)
                faces.append(cropped_frame)
            else:
                pass
                #logging.info(f"No face detected on frame {i}")

        if len(faces) > 0:
            cropped = np.stack(faces, axis=0)
            logging.info(f"Finished. Cropped shape: {cropped.shape}. Total removed frames: {len(frames) - len(faces)}")
            save_video(out_filename, np.stack(faces, axis=0), fps)
            logging.info(f"Saved into: {out_filename}")
        else:
            logging.info(f"No face on all frame")

if __name__ == '__main__':
    input_dir = params.input_dir
    output_dir = params.output_dir
    dim = params.dim
    detector = params.detector

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #crop_videos(input_dir, output_dir, detector, (dim, dim))
    crop_videos2(input_dir, output_dir, (dim, dim))