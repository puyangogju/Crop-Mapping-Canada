import os
import random
import numpy as np
import rasterio
import torch
import torch.nn.functional as F

LABEL_FILENAME = "y.tif"


class MunichDataset(torch.utils.data.Dataset):
    """
    If the first label is for example "1|unknown" then this will be replaced with a 0 (zero).
    If you want to ignore other labels, then remove them from the classes.txt file and
    this class will assigne label 0 (zero).
    Warning: this tecnique is not stable!
    """

    def __init__(self, root_dir, seqlength=32, tileids=None, transform=None):
        self.root_dir = root_dir
        self.name = os.path.basename(root_dir)
        self.data_dirs = [d for d in os.listdir(self.root_dir) if d.startswith("data")]
        self.seqlength = seqlength
        self.transform = transform
        self.munich_format = None
        self.src_labels = None
        self.dst_labels = None
        # labels read from groudtruth files (y.tif)
        # useful field to check the available labels
        self.unique_labels = np.array([], dtype=float)

        stats = dict(
            rejected_nopath=0,
            rejected_length=0,
            total_samples=0)

        # statistics
        self.samples = list()

        self.ndates = list()

        dirs = []
        if tileids is None:
            # files = os.listdir(self.data_dirs)
            for d in self.data_dirs:
                dirs_name = os.listdir(os.path.join(self.root_dir, d))
                dirs_path = [os.path.join(self.root_dir, d, f) for f in dirs_name]
                dirs.extend(dirs_path)
        else:
            # tileids e.g. "tileids/train_fold0.tileids" path of line separated tileids specifying
            with open(os.path.join(self.root_dir, tileids), 'r') as f:
                files = [el.replace("\n", "") for el in f.readlines()]
            for d in self.data_dirs:
                dirs_path = [os.path.join(self.root_dir, d, f) for f in files]
                dirs.extend(dirs_path)

        self.classids, self.classes = read_classes(os.path.join(self.root_dir, "classes.txt"))

        for path in dirs:
            if not os.path.exists(path):
                stats["rejected_nopath"] += 1
                continue
            if not os.path.exists(os.path.join(path, LABEL_FILENAME)):
                stats["rejected_nopath"] += 1
                continue

            ndates = len(get_dates(path))

            if ndates < self.seqlength:
                stats["rejected_length"] += 1
                continue  # skip shorter sequence lengths

            stats["total_samples"] += 1
            self.samples.append(path)
            self.ndates.append(ndates)

        print_stats(stats)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # path = os.path.join(self.data_dir, self.samples[idx])
        path = self.samples[idx]
        if path.endswith(os.sep):
            path = path[:-1]

        label, profile = read(os.path.join(path, LABEL_FILENAME))

        profile["name"] = self.samples[idx]

        # unique dates sorted ascending
        dates = get_dates(path, n=self.seqlength)

        x10 = list()
        x20 = list()
        x60 = list()

        for date in dates:
            if self.munich_format is None:
                self.munich_format = os.path.exists(os.path.join(path, date + "_10m.tif"))

            if self.munich_format:  # munich dataset
                x10.append(read(os.path.join(path, date + "_10m.tif"))[0])
                x20.append(read(os.path.join(path, date + "_20m.tif"))[0])
                x60.append(read(os.path.join(path, date + "_60m.tif"))[0])
            else:  # IREA dataset
                x10.append(read(os.path.join(path, date + ".tif"))[0])

        x10 = np.array(x10) * 1e-4
        if self.munich_format:
            x20 = np.array(x20) * 1e-4
            x60 = np.array(x60) * 1e-4

        # replace stored ids with index in classes csv
        label = label[0]
        self.unique_labels = np.unique(np.concatenate([label.flatten(), self.unique_labels]))
        new = np.zeros(label.shape, int)
        for cl, i in zip(self.classids, range(len(self.classids))):
            new[label == cl] = i

        label = new

        label = torch.from_numpy(label)
        x10 = torch.from_numpy(x10)
        if self.munich_format:
            x20 = torch.from_numpy(x20)
            x60 = torch.from_numpy(x60)

            x20 = F.interpolate(x20, size=x10.shape[2:4])
            x60 = F.interpolate(x60, size=x10.shape[2:4])

            x = torch.cat((x10, x20, x60), 1)
        else:
            x = x10

        # permute channels with time_series (t x c x h x w) -> (c x t x h x w)
        x = x.permute(1, 0, 2, 3)

        x = x.float()

        if self.transform is not None:
            label, x = self.transform(label, x)

        label = label.long()

        folders = self.samples[idx].split('/')
        filename = os.path.join(folders[-2], folders[-1])

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
            id, cl = row.split('|')
            ids.append(int(id))
            names.append(cl)

    return ids, names


def read(file):
    with rasterio.open(file) as src:
        return src.read(), src.profile


def print_stats(stats):
    print_lst = list()
    for k, v in zip(stats.keys(), stats.values()):
        print_lst.append("{}:{}".format(k, v))
    print('\n', ", ".join(print_lst))
