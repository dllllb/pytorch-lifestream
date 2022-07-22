import logging
from copy import deepcopy

import pytorch_lightning as pl
import torch
import torchmetrics
from omegaconf import DictConfig

from ptls.data_load.padded_batch import PaddedBatch

logger = logging.getLogger(__name__)


class SequenceToTarget(pl.LightningModule):
    """LightningModule for supervised task or for inference

    For binary classification problem use settings like:
        head=torch.nn.Sequential(
            ...
            torch.nn.Linear(in_size, 1),
            torch.nn.Sigmoid(),
            torch.nn.Flatten(start_dim=0),
        ),
        loss=BCELoss(),
        metric_list=torchmetrics.AUROC(num_classes=2),

    For multiclass problem use settings like:
        head=torch.nn.Sequential(
            torch.nn.Linear(in_size, num_classes),
            torch.nn.LogSoftmax(dim=1),
        ),
        loss=torch.nn.NLLLoss(),
        metric_list=torchmetrics.Accuracy(compute_on_step=False),

    For regression problem use settings like:
        head=torch.nn.Sequential(
            torch.nn.Linear(in_size, 1),
            torch.nn.Flatten(start_dim=0),
        ),
        loss=torch.nn.MSELoss(),
        metric_list=torchmetrics.MeanSquaredError(),

    For `seq_encoder` inference:
        Just set `seq_encoder` and `head` (if needed), keep other parameters as None.
        Next use `torch.vstack(trainer.predict(sequence_to_target, dl_inference))`
        This call `seq_encoder.forward` which provide embeddings.

    Parameters
    ----------
    seq_encoder:
        Sequence encoder. May be pretrained or with random initialisation
    head:
        Head layers for your problem. May be simple or multilayer or None.
    loss:
        Your loss for specific problem.
    metric_list:
        One or dict of metrics for specific problem: {'metric_name1': metric1, 'metric_name2': metric2, ...}.
    optimizer_partial:
        optimizer init partial. Network parameters are missed.
    lr_scheduler_partial:
        scheduler init partial. Optimizer are missed.
    pretrained_lr:
        lr for seq_encoder. Can be one of:
        - 'freeze' - seq_encoder will be frozen
        - float - correspond seq_encoder lr, other network parameters have lr from optimizer_partial
        - None - all network parameters have lr from optimizer_partial
    train_update_n_steps:
        Interval of train metric update. Use it for large train dataset with heavy metrics.
        Update train metrics on each step when `train_update_n_steps is None`

    """

    def __init__(self,
                 seq_encoder: torch.nn.Module,
                 head: torch.nn.Module=None,
                 loss: torch.nn.Module=None,
                 metric_list: torchmetrics.Metric=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                 pretrained_lr=None,
                 train_update_n_steps=None,
                 ):
        super().__init__()

        self.save_hyperparameters(ignore=[
            'seq_encoder', 'head', 'loss', 'metric_list', 'optimizer_partial', 'lr_scheduler_partial'])

        self.seq_encoder = seq_encoder
        self.head = head
        self.loss = loss

        if type(metric_list) is dict or type(metric_list) is DictConfig:
            metric_list = [(k, v) for k, v in metric_list.items()]
        else:
            if type(metric_list) is not list:
                metric_list = [metric_list]
            metric_list = [(m.__class__.__name__, m) for m in metric_list]

        self.train_metrics = torch.nn.ModuleDict([(name, deepcopy(mc)) for name, mc in metric_list])
        self.valid_metrics = torch.nn.ModuleDict([(name, deepcopy(mc)) for name, mc in metric_list])
        self.test_metrics = torch.nn.ModuleDict([(name, deepcopy(mc)) for name, mc in metric_list])

        self.optimizer_partial = optimizer_partial
        self.lr_scheduler_partial = lr_scheduler_partial

    def forward(self, x):
        x = self.seq_encoder(x)
        if self.head is not None:
            x = self.head(x)
        return x

    def training_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        loss = self.loss(y_h, y)
        self.log('loss', loss)
        if isinstance(x, PaddedBatch):
            self.log_dict(
                {"seq_len": x.seq_lens.float().mean(),
                 "y": y.unsqueeze(-1).sum(1).squeeze().float().mean(),
                }, prog_bar=True)
        train_update_n_steps = self.hparams.train_update_n_steps
        if train_update_n_steps is None or \
                train_update_n_steps is not None and self.global_step % train_update_n_steps == 0:
            for name, mf in self.train_metrics.items():
                mf(y_h, y)
        return loss

    def training_epoch_end(self, outputs):
        for name, mf in self.train_metrics.items():
            self.log(f'train_{name}', mf.compute(), prog_bar=False)

    def validation_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        for name, mf in self.valid_metrics.items():
            mf(y_h, y)

    def validation_epoch_end(self, outputs):
        for name, mf in self.valid_metrics.items():
            self.log(f'val_{name}', mf.compute(), prog_bar=True)

    def test_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        for name, mf in self.test_metrics.items():
            mf(y_h, y)

    def test_epoch_end(self, outputs):
        for name, mf in self.test_metrics.items():
            value = mf.compute().item()
            self.log(f'test_{name}', value, prog_bar=False)

    def configure_optimizers(self):
        if self.hparams.pretrained_lr is not None:
            if self.hparams.pretrained_lr == 'freeze':
                for p in self.seq_encoder.parameters():
                    p.requires_grad = False
                logger.info('Created optimizer with frozen encoder')
                parameters = self.parameters()
            else:
                parameters = [
                    {'params': self.seq_encoder.parameters(), 'lr': self.hparams.pretrained_lr},
                    {'params': self.head.parameters()},  # use predefined lr from `self.optimizer_partial`
                ]
                logger.info('Created optimizer with two lr groups')
        else:
            parameters = self.parameters()

        optimizer = self.optimizer_partial(parameters)
        scheduler = self.lr_scheduler_partial(optimizer)
        return [optimizer], [scheduler]
