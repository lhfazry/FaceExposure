import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from datasets.ExposureDataset import ExposureDataset

class ExposuretDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "datasets/Exposure", 
            batch_size: int = 32, 
            num_workers: int = 8, 
            dataset_mode: str = 'repeat',
            csv_file: str = 'datasets/video_exposure.csv'):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_mode = dataset_mode
        self.csv_file = csv_file

        #self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        #MNIST(self.data_dir, train=True, download=True)
        #MNIST(self.data_dir, train=False, download=True)
        print('prepare data')

    def setup(self, stage = None):
        print(f'setup: {self.data_dir}')

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = ExposureDataset(root=self.data_dir,
                                split="train", augmented=True,
                                csv_file=self.csv_file)
            
            self.val_set   = ExposureDataset(root=self.data_dir, split="val",
                                csv_file=self.csv_file)

        if stage == "validate" or stage is None:
            self.val_set   = ExposureDataset(root=self.data_dir, split="val",
                                csv_file=self.csv_file)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set   = ExposureDataset(root=self.data_dir, split="test",
                                csv_file=self.csv_file)

        if stage == "predict" or stage is None:
            self.predict_set   = ExposureDataset(root=self.data_dir, split="test",
                                csv_file=self.csv_file)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size, num_workers=self.num_workers)