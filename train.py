from dataset.lombardia_datamodule import LombardiaDataModule
from dataset.munich_datamodule import MunichDataModule
import utils.options as opt
from pathlib import Path
from model import Model
import os
import lightning.pytorch as pl
from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger


if __name__ == '__main__':
    args = opt.initialize().parse_args()

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1,
        save_top_k=1,
        monitor='val_acc',
        mode='max',
        save_last=True
    )

    os.environ["CUDA_VISIBLE_DEVICES"]=','.join(map(str, args.gpus))
    pl.seed_everything(42, workers=True)

    trainer = pl.Trainer(
        accelerator=args.gpu_or_cpu,
        devices=find_usable_cuda_devices(1),
        logger=CSVLogger(Path("logs", args.arch, args.dataset), name="train"),
        benchmark=True,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback]
    )

    if args.dataset == 'munich':
        model = Model(arch=args.arch, depth=32, in_channels=13, out_classes=18)
        datamodule = MunichDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    elif args.dataset == 'lombardia':
        model = Model(arch=args.arch, depth=32, in_channels=9, out_classes=8)
        datamodule = LombardiaDataModule(data_dir=args.data_dir, batch_size=args.batch_size)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=args.ckpt_path
    )


# Munich:
# nohup python train.py --arch deeplabv3 --dataset munich --gpus 0 > ./nohup/log_deeplab_munich.txt 2>&1 &
# nohup python train.py --arch fpn --dataset munich --gpus 1 > ./nohup/log_fpn_munich.txt 2>&1 &
# nohup python train.py --arch swin_unetr --dataset munich --gpus 2 > ./nohup/log_swin_munich.txt 2>&1 &
# nohup python train.py --arch unet --dataset munich --gpus 3 > ./nohup/log_unet_munich.txt 2>&1 &

# Munich resume:
# nohup python train.py --arch deeplabv3 --dataset munich --gpus 0 --ckpt_path ./logs/deeplabv3_munich/version_0/checkpoints/last.ckpt >> ./nohup/log_deeplab_munich.txt 2>&1 &
# nohup python train.py --arch fpn --dataset munich --gpus 1 --ckpt_path ./logs/fpn_munich/version_0/checkpoints/last.ckpt >> ./nohup/log_fpn_munich.txt 2>&1 &
# nohup python train.py --arch swin_unetr --dataset munich --gpus 2 --ckpt_path ./logs/swin_unetr_munich/version_0/checkpoints/last.ckpt >> ./nohup/log_swin_munich.txt 2>&1 &
# nohup python train.py --arch unet --dataset munich --gpus 3 --ckpt_path ./logs/unet_munich/version_0/checkpoints/last.ckpt >> ./nohup/log_unet_munich.txt 2>&1 &

# Lombardia:
# nohup python train.py --arch deeplabv3 --dataset lombardia --gpus 0 > ./nohup/log_deeplab_lombardia.txt 2>&1 &
# nohup python train.py --arch fpn --dataset lombardia --gpus 1 > ./nohup/log_fpn_lombardia.txt 2>&1 &
# nohup python train.py --arch swin_unetr --dataset lombardia --gpus 2 > ./nohup/log_swin_lombardia.txt 2>&1 &
# nohup python train.py --arch unet --dataset lombardia --gpus 3 > ./nohup/log_unet_lombardia.txt 2>&1 &

# Lombardia resume:
# nohup python train.py --arch deeplabv3 --dataset lombardia --gpus 0 --ckpt_path ./logs/deeplabv3_munich/version_0/checkpoints/last.ckpt >> ./nohup/log_deeplab_lombardia.txt 2>&1 &
# nohup python train.py --arch fpn --dataset lombardia --gpus 1 --ckpt_path ./logs/fpn_munich/version_0/checkpoints/last.ckpt >> ./nohup/log_fpn_lombardia.txt 2>&1 &
# nohup python train.py --arch swin_unetr --dataset lombardia --gpus 2 --ckpt_path ./logs/swin_unetr_munich/checkpoints/version_0/last.ckpt >> ./nohup/log_swin_lombardia.txt 2>&1 &
# nohup python train.py --arch unet --dataset lombardia --gpus 3 --ckpt_path ./logs/unet_munich/version_0/checkpoints/last.ckpt >> ./nohup/log_unet_lombardia.txt 2>&1 &
