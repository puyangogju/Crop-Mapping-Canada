import lightning.pytorch as pl
import torch.nn.functional as F
from pathlib import Path
from zoo.deeplabv3_3d import DeepLabV3_3D
from zoo.fpn_3d import FPN_3D
from zoo.swin_unetr import SwinUNETRSentinel
from zoo.unet_3d import UNet_3D
import torch
from torch.nn import CrossEntropyLoss
from utils.metrics import ConfusionMatrix
from monai.losses import DiceCELoss
import numpy as np


class Model(pl.LightningModule):
    def __init__(self, arch='swin_unetr', depth=32, in_channels=13, out_classes=18):
        super().__init__()
        self.out_classes = out_classes
        self.arch = arch

        if arch == 'deeplabv3':
            self.model = DeepLabV3_3D(depth=depth, in_channels=in_channels, out_classes=out_classes)
            self.loss_fn = CrossEntropyLoss(ignore_index=0)
        elif arch == 'fpn':
            self.model = FPN_3D(depth=depth, in_channels=in_channels, out_classes=out_classes)
            self.loss_fn = CrossEntropyLoss(ignore_index=0)
        elif arch == 'swin_unetr':
            self.model = SwinUNETRSentinel(depth=depth, in_channels=in_channels, out_classes=out_classes)
            self.loss_fn = DiceCELoss(sigmoid=True, include_background=False)
        elif arch == 'unet':
            self.model = UNet_3D(depth=depth, in_channels=in_channels, out_classes=out_classes)
            self.loss_fn = CrossEntropyLoss(ignore_index=0)

        self.best_val_acc = 0
        self.labels = list(range(out_classes))
        self.conf_matrix = ConfusionMatrix(self.labels, ignore_class=0)


    def forward(self, x):
        pred = self.model(x)
        return pred

    def training_step(self, batch, batch_idx):
        x, y, _ = batch  # batch_size x 48 x 48
        y_hat = self(x)  # batch_size x out_classes x 48 x 48

        if self.arch == 'swin_unetr':  # DiceCELoss requires y one hot encoded
            y_enc = F.one_hot(y.to(torch.int64), num_classes=self.out_classes)
            y_enc = y_enc.permute(0, 3, 1, 2)
            loss = self.loss_fn(y_hat, y_enc)
        else:
            loss = self.loss_fn(y_hat, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.trainer.datamodule.batch_size)
        return loss

    def on_validation_epoch_start(self):
        self.conf_matrix = ConfusionMatrix(list(range(self.out_classes)), ignore_class=0)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        if self.arch == 'swin_unetr':  # DiceCELoss requires y one hot encoded
            y_enc = F.one_hot(y.to(torch.int64), num_classes=self.out_classes)
            y_enc = y_enc.permute(0, 3, 1, 2)
            loss = self.loss_fn(y_hat, y_enc)
        else:
            loss = self.loss_fn(y_hat, y)

        self.log('val_loss', loss, on_step=True, on_epoch=True, batch_size=self.trainer.datamodule.batch_size)

        y_hat = torch.argmax(y_hat, dim=1)
        self.metrics_v, self.metrics_scalar = self.conf_matrix(y_hat, y)

    def on_validation_epoch_end(self):
        overall_acc = self.metrics_scalar['OA']
        self.log('val_acc', overall_acc)

        is_best = self.metrics_scalar['OA'] > self.best_val_acc
        self.best_val_acc = self.metrics_scalar['OA'] if is_best else self.best_val_acc
        print(f"Overall Accuracy: {overall_acc}  {'**new best result' if is_best else ''}")

        if is_best:
            self.save_metrics()    

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        self.metrics_v, self.metrics_scalar = self.conf_matrix(y_hat, y)

    def on_test_epoch_end(self):
        self.save_metrics()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, weight_decay=1e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, verbose=True)
        return [optimizer], [scheduler]
    
    def save_metrics(self):
        if self.trainer.sanity_checking:
            return

        cls_names = np.array(self.trainer.datamodule.classes)[self.conf_matrix.get_labels()]
        with open(Path(self.logger.log_dir) / ("best_result.txt" if self.trainer.validating else "result.txt"), 'w') as f:
            f.write('classes:\n' + np.array2string(cls_names) + '\n')
            for k, v in self.metrics_v.items():
                f.write(k + '\n')
                if len(v.shape) == 1:
                    for ki, vi in zip(cls_names, v):
                        f.write("%.2f" % vi + '\t' + ki + '\n')
                elif len(v.shape) == 2:  # confusion matrix
                    num_gt = np.sum(v, axis=1)
                    f.write('\n'.join(
                        [''.join(['{:10}'.format(item) for item in row]) + '  ' + lab + '(%d)' % tot
                         for row, lab, tot in zip(v, cls_names, num_gt)]))
                    f.write('\n')
            str_metrics = ''.join(['%s| %f | ' % (key, value) for (key, value) in self.metrics_scalar.items()])
            f.write(f"\n{str_metrics}")
