import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import pprint
import rasterio
import random

LABEL_FILENAME = "y.tif"


class LombardiaDataset(torch.utils.data.Dataset):
    """
    If the first label is for example "1|unknown" then this will be replaced with a 0 (zero).
    If you want to ignore other labels, then remove them from the classes.txt file and
    this class will assigne label 0 (zero).
    Warning: this tecnique is not stable!
    """

    def __init__(self, root_dirs, years, classes_path, seqlength, tileids=None, transform=None):
        self.seqlength = seqlength
        self.transform = transform
        # labels read from groudtruth files (y.tif)
        # useful field to check the available labels
        self.unique_labels = np.array([], dtype=float)
        cls_info = read_classes(classes_path)
        self.classids = cls_info[0]
        self.classes = cls_info[1]

        if type(years) is not list:
            years = [years]
        self.data_dirs = years

        if type(root_dirs) is not list:
            root_dirs = [root_dirs]
        self.root_dirs = [str(r).rstrip("/") for r in root_dirs]
        self.name = ""
        self.samples = list()
        self.ndates = list()
        for root_dir in self.root_dirs:
            print("Reading dataset info:", root_dir)
            self.name += os.path.basename(root_dir) + '_'

            for d in self.data_dirs:
                if not os.path.isdir(os.path.join(root_dir, d)):
                    sys.exit('The directory ' + os.path.join(root_dir, d) + " does not exist!")

            stats = dict(
                rejected_nopath=0,
                rejected_length=0,
                total_samples=0)

            dirs = []
            if tileids is None:
                # files = os.listdir(self.data_dirs)
                for d in self.data_dirs:
                    dirs_name = os.listdir(os.path.join(root_dir, d))
                    dirs_path = [os.path.join(root_dir, d, f) for f in dirs_name]
                    dirs.extend(dirs_path)
            else:
                # tileids e.g. "tileids/train_fold0.tileids" path of line separated tileids specifying
                with open(os.path.join(root_dir, tileids), 'r') as f:
                    files = [el.replace("\n", "") for el in f.readlines()]
                for d in self.data_dirs:
                    dirs_path = [os.path.join(root_dir, d, f) for f in files]
                    dirs.extend(dirs_path)

            for path in tqdm(dirs):

                if not os.path.exists(path):
                    stats["rejected_nopath"] += 1
                    continue
                if not os.path.exists(os.path.join(path, LABEL_FILENAME)):
                    stats["rejected_nopath"] += 1
                    continue
                ndates = len(get_dates(path))

                stats["total_samples"] += 1
                self.samples.append(path)
                self.ndates.append(ndates)

            pprint.pprint(stats)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        if path.endswith(os.sep):
            path = path[:-1]

        label, profile = read(os.path.join(path, LABEL_FILENAME))

        profile["name"] = self.samples[idx]

        # unique dates sorted ascending
        dates = get_dates(path, n=self.seqlength)

        x10 = list()
        for date in dates:
            x10.append(read(os.path.join(path, date + ".tif"))[0])

        x10 = np.array(x10) * 1e-4

        # replace stored ids available in classes csv with indexes
        label = label[0]
        self.unique_labels = np.unique(np.concatenate([label.flatten(), self.unique_labels]))
        new = np.zeros(label.shape, int)
        for cl, i in zip(self.classids, range(len(self.classids))):
            for c in cl:
                new[label == c] = i
        label = new

        label = torch.from_numpy(label)
        x10 = torch.from_numpy(x10)
        x = x10

        # permute channels with time_series (t x c x h x w) -> (c x t x h x w)
        x = x.permute(1, 0, 2, 3)

        x = x.float()

        if self.transform is not None:
            label, x = self.transform(label, x)

        label = label.long()

        folders = self.samples[idx].split('/')
        filename = os.path.join(folders[-3], folders[-2], folders[-1])

        return x, label, filename


def get_dates(path, n=None):
    """
    extracts a list of unique dates from dataset sample

    :param path: to dataset sample folder
    :param n: choose n random samples from all available dates
    :return: list of unique dates in YYYYMMDD format
    """

    files = os.listdir(path)
    dates = list()
    for f in files:
        f = f.split("_")[0]
        if len(f) == 8:  # 20160101
            dates.append(f)

    dates = list(set(dates))

    if n is not None:
        dates = random.sample(dates, n)

    dates.sort()
    return dates


def read_classes(csv):
    with open(csv, 'r') as f:
        classes = f.readlines()

    ids = list()
    names = list()
    for row in classes:
        row = row.replace("\n", "")
        if '|' in row:
            cls_info = row.split('|')
            # we can have multiple id
            id_info = cls_info[0].split(',')
            id_info = [int(x) for x in id_info]
            # ids.append(int(cls_info[0]))
            ids.append(id_info)
            names.append(cls_info[1])
    return ids, names


def read(file):
    with rasterio.open(file) as src:
        return src.read(), src.profile