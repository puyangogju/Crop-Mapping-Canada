from dataset.munich_datamodule import MunichDataModule
from dataset.lombardia_datamodule import LombardiaDataModule
import os
import utils.options as opt
from pathlib import Path
from model import Model
import lightning.pytorch as pl
from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.loggers import CSVLogger

if __name__ == '__main__':
    args = opt.initialize().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=','.join(map(str, args.gpus))
    pl.seed_everything(42, workers=True)

    trainer = pl.Trainer(
        accelerator=args.gpu_or_cpu,
        devices=find_usable_cuda_devices(1),
        logger=CSVLogger(Path("logs", args.arch, args.dataset), name="test")
    )

    if args.dataset == 'munich':
        model = Model(arch=args.arch, depth=32, in_channels=13, out_classes=18)
        datamodule = MunichDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    elif args.dataset == 'lombardia':
        model = Model(arch=args.arch, depth=32, in_channels=9, out_classes=8)
        datamodule = LombardiaDataModule(data_dir=args.data_dir, batch_size=args.batch_size, test_id=args.test_id)

    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=args.ckpt_path
    )


# Munich:
# python test.py --arch deeplabv3 --dataset munich --ckpt_path ./pretrained/deeplab_munich.ckpt
# python test.py --arch fpn --dataset munich --ckpt_path ./pretrained/fpn_munich.ckpt
# python test.py --arch swin_unetr --dataset munich --ckpt_path ./pretrained/swin_munich.ckpt
# python test.py --arch unet --dataset munich --ckpt_path ./pretrained/unet_munich.ckpt

# Lombardia A:
# python test.py --arch deeplabv3 --dataset lombardia --test_id A --ckpt_path ./pretrained/deeplab_lombardia.ckpt
# python test.py --arch fpn --dataset lombardia --test_id A --ckpt_path ./pretrained/fpn_lombardia.ckpt
# python test.py --arch swin_unetr --dataset lombardia --test_id A --ckpt_path ./pretrained/swin_lombardia.ckpt
# python test.py --arch unet --dataset lombardia --test_id A --ckpt_path ./pretrained/unet_lombardia.ckpt

# Lombardia Y:
# python test.py --arch deeplabv3 --dataset lombardia --test_id Y --ckpt_path ./pretrained/deeplab_lombardia.ckpt
# python test.py --arch fpn --dataset lombardia --test_id Y --ckpt_path ./pretrained/fpn_lombardia.ckpt
# python test.py --arch swin_unetr --dataset lombardia --test_id Y --ckpt_path ./pretrained/swin_lombardia.ckpt
# python test.py --arch unet --dataset lombardia --test_id Y --ckpt_path ./pretrained/unet_lombardia.ckpt
