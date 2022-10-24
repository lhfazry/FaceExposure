import pytorch_lightning as pl
import os
import pandas as pd
import cv2
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from datasets.ExposureDataset import ExposureDataset
from skmultilearn.model_selection import iterative_train_test_split
from utils.mlsmote import get_minority_instace

class ExposuretDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "datasets/Exposure", 
            batch_size: int = 32, 
            num_workers: int = 8, 
            dataset_mode: str = 'repeat',
            sampling_strategy: str = 'truncate',
            upsampling: int = 0,
            min_frames = 80,
            max_frames = 512,
            csv_file: str = 'datasets/video_exposure.csv'):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_mode = dataset_mode
        self.csv_file = csv_file
        self.sampling_strategy = sampling_strategy
        self.upsampling = upsampling
        self.max_frames = max_frames
        self.min_frames = min_frames

        #self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        df = pd.read_csv(self.csv_file)
        valid_rows = []

        for index, row in df.iterrows():
            video_path = os.path.join(self.data_dir, row["video_name"])

            if row["video_name"].startswith('_'):
                continue

            if os.path.exists(video_path) and self.count_frame(video_path) > self.min_frames:
                valid_rows.append(row)

        df = pd.DataFrame(valid_rows)

        data_train, label_train, data_test, label_test = iterative_train_test_split(
            df[["video_name"]].values,
            df[["neutral", "happy", "sad", "contempt", "anger", "disgust", "surprised", "fear"]].values,
            test_size = 0.2
        )

        data_train, label_train, data_val, label_val = iterative_train_test_split(
            data_train,
            label_train,
            test_size = 0.25
        )

        self.data_train, self.label_train = data_train, label_train
        self.data_val, self.label_val = data_val, label_val
        self.data_test, self.label_test = data_test, label_test

        if self.upsampling == 1:
            self.data_train, self.label_train = self.upsample_data(self.data_train, self.label_train)

    def upsample_data(self, data_train, label_train):
        X = pd.DataFrame(data_train)
        y = pd.DataFrame(label_train)

        X_min, y_min = get_minority_instace(X, y)
        print(X_min)
        X_min['video_name'] = X_min['video_name'].apply(lambda x : f"_{x}")
        #X = X.drop(X_min.index)
        #y = y.drop(y_min.index)

        for _ in range(9):
            X = X.append(X_min, ignore_index = True)
            y = y.append(y_min, ignore_index = True)

        return X.to_numpy(), y.to_numpy()
            
    def setup(self, stage = None):
        print(f'setup: {self.data_dir}')

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = ExposureDataset(root=self.data_dir,
                                data=self.data_train,
                                label=self.label_train,
                                augmented=True,
                                sampling_strategy=self.sampling_strategy)
            
            self.val_set   = ExposureDataset(root=self.data_dir, 
                                data=self.data_val,
                                label=self.label_val,
                                sampling_strategy=self.sampling_strategy)

        if stage == "validate" or stage is None:
            self.val_set   = ExposureDataset(root=self.data_dir, 
                                data=self.data_val,
                                label=self.label_val,
                                sampling_strategy=self.sampling_strategy)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set   = ExposureDataset(root=self.data_dir, 
                                data=self.data_test,
                                label=self.label_test,
                                sampling_strategy=self.sampling_strategy)

        if stage == "predict" or stage is None:
            self.predict_set   = ExposureDataset(root=self.data_dir,
                                data=self.data_test,
                                label=self.label_test,
                                sampling_strategy=self.sampling_strategy)

    def train_dataloader(self):
        return DataLoader(self.train_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True,
            drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            drop_last=True)

    def count_frame(self, filename: str):
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        capture = cv2.VideoCapture(filename)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame_count