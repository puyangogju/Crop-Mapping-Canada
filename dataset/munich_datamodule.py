import utils.custom_transform as T
import opendatasets as od
import os
import lightning.pytorch as pl
from dataset.munich_dataset import MunichDataset
from torch.utils.data import DataLoader
from pathlib import Path


class MunichDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.download_dir = data_dir
        self.data_dir = data_dir / 'sentinel2-munich480' / 'munich480'
        self.batch_size = batch_size
        self.num_workers = os.cpu_count()
        self.test_name = f'munich_test'
        self.transform = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip()])

    def prepare_data(self):
        od.download_kaggle_dataset("https://www.kaggle.com/datasets/artelabsuper/sentinel2-munich480",
                                   data_dir=self.download_dir)

    def setup(self, stage=None):
        self.sentinel_train = MunichDataset(self.data_dir, tileids=Path('tileids') / 'train_fold0.tileids',
                                            seqlength=32, transform=self.transform)
        self.sentinel_val = MunichDataset(self.data_dir, tileids=Path('tileids') / 'test_fold0.tileids', seqlength=32)
        self.sentinel_test = MunichDataset(self.data_dir, tileids=Path('tileids') / 'eval.tileids', seqlength=32)
        self.classes = self.sentinel_train.classes

    def train_dataloader(self):
        return DataLoader(self.sentinel_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.sentinel_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.sentinel_test, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
