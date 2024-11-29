import argparse
from pathlib import Path


def initialize():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt_path', type=Path, default=None, help='checkpoint or pretrained path')
    parser.add_argument('--dataset', type=str, default='munich', choices=['lombardia', 'munich'])
    parser.add_argument('--test_id', type=str, default='A', choices=['A', 'Y'])
    parser.add_argument('--arch', type=str, default='swin_unetr', choices=['deeplabv3', 'fpn', 'swin_unetr', 'unet'])
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--data_dir', type=Path, default=Path.cwd().parent)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--gpu_or_cpu', type=str, default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('--gpus', type=int, default=[0], nargs='+')
    return parser

