import os
from pathlib import Path
import opendatasets as od
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import utils.custom_transform as T
from dataset.lombardia_dataset import LombardiaDataset


class LombardiaDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, test_id='A'):
        super().__init__()
        self.download_dir = data_dir
        self.data_dir = data_dir / 'sentinel2-crop-mapping'
        self.batch_size = batch_size
        self.num_workers = os.cpu_count()
        self.test_id = test_id
        self.test_name = f'lombardia_test{test_id}'
        self.transform = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip()])

    def prepare_data(self):
        od.download_kaggle_dataset("https://www.kaggle.com/datasets/ignazio/sentinel2-crop-mapping",
                                   data_dir=self.download_dir)

    def setup(self, stage=None):
        self.sentinel_train = LombardiaDataset(
            root_dirs=[self.data_dir / 'lombardia', self.data_dir / 'lombardia2'],
            years=['data2016', 'data2017', 'data2018'],
            classes_path=self.data_dir / 'lombardia-classes' / 'classes25pc.txt',
            seqlength=32,
            tileids=Path('tileids') / 'train_fold0.tileids',
            transform=self.transform
        )
        
        self.sentinel_val = LombardiaDataset(
            root_dirs=[self.data_dir / 'lombardia', self.data_dir / 'lombardia2'],
            years=['data2016', 'data2017', 'data2018'],
            classes_path=self.data_dir / 'lombardia-classes' / 'classes25pc.txt',
            seqlength=32,
            tileids=Path('tileids') / 'test_fold0.tileids',
        )

        self.sentinel_test_A = LombardiaDataset(
            root_dirs=[self.data_dir / 'lombardia3'],
            years=['data2019'],
            classes_path=self.data_dir / 'lombardia-classes' / 'classes25pc.txt',
            seqlength=32,
            tileids=Path('tileids') / 'testA.tileids'
        )

        self.sentinel_test_Y = LombardiaDataset(
            root_dirs=[self.data_dir / 'lombardia', self.data_dir / 'lombardia2'],
            years=['data2019'],
            classes_path=self.data_dir / 'lombardia-classes' / 'classes25pc.txt',
            seqlength=32,
            tileids=Path('tileids') / 'testY2019.tileids'
        )

        self.classes = self.sentinel_train.classes

    def train_dataloader(self):
        return DataLoader(self.sentinel_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.sentinel_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        if self.test_id == 'A':
            return DataLoader(self.sentinel_test_A, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
        elif self.test_id == 'Y':
            return DataLoader(self.sentinel_test_Y, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)
