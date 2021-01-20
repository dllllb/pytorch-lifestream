import logging
from copy import deepcopy

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional.classification import auroc

from dltranz.loss import get_loss
from dltranz.seq_encoder import create_encoder
from dltranz.train import get_optimizer, get_lr_scheduler
from dltranz.models import create_head_layers

logger = logging.getLogger(__name__)


class EpochAuroc(pl.metrics.Metric):
    def __init__(self):
        super().__init__(compute_on_step=False)

        self.add_state('y_hat', default=[])
        self.add_state('y', default=[])

    def update(self, y_hat, y):
        self.y_hat.append(y_hat)
        self.y.append(y)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)
        return auroc(y_hat, y)


class SequenceClassify(pl.LightningModule):
    def __init__(self, params, pretrained_encoder=None):
        super().__init__()
        self.save_hyperparameters('params')

        self.loss = get_loss(params)

        if pretrained_encoder is not None:
            self._seq_encoder = deepcopy(pretrained_encoder)
            self._is_pretrained_encoder = True
        else:
            self._seq_encoder = create_encoder(params, is_reduce_sequence=True)
            self._is_pretrained_encoder = False
        self._head = create_head_layers(params, self._seq_encoder)

        # metrics
        d_metrics = {
            'auroc': EpochAuroc,
            'accuracy': pl.metrics.Accuracy,
        }
        params_score_metric = params['score_metric']
        if type(params_score_metric) is str:
            params_score_metric = [params_score_metric]
        metric_cls = [(name, d_metrics[name]) for name in params_score_metric]
        self.valid_metrics = torch.nn.ModuleDict([(name, mc()) for name, mc in metric_cls])
        self.test_metrics = torch.nn.ModuleDict([(name, mc()) for name, mc in metric_cls])

    @property
    def seq_encoder(self):
        return self._seq_encoder

    def forward(self, x):
        x = self._seq_encoder(x)
        x = self._head(x)
        return x

    def training_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        loss = self.loss(y_h, y)
        self.log('loss', loss)
        self.log('seq_len', x.seq_lens.float().mean(), prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        for name, mf in self.valid_metrics.items():
            mf(y_h, y)

    def validation_epoch_end(self, outputs):
        for name, mf in self.valid_metrics.items():
            self.log(f'valid_{name}', mf.compute(), prog_bar=True)

    def test_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        for name, mf in self.test_metrics.items():
            mf(y_h, y)

    def test_epoch_end(self, outputs):
        for name, mf in self.test_metrics.items():
            self.log(f'test_{name}', mf.compute())

    def configure_optimizers(self):
        params = self.hparams.params
        if self._is_pretrained_encoder:
            optimizer = self.get_pretrained_optimizer()
        else:
            optimizer = get_optimizer(self, params)
        scheduler = get_lr_scheduler(optimizer, params)
        return [optimizer], [scheduler]

    def get_pretrained_optimizer(self):
        params = self.hparams.params

        if params['pretrained.lr'] == 'freeze':
            self._seq_encoder.freeze()
            logger.info('Created optimizer with frozen encoder')
            return get_optimizer(self, params)

        parameters = [
            {'params': self._seq_encoder.parameters(), 'lr': params['pretrained.lr']},
            {'params': self._head.parameters(), 'lr': params['train.lr']},
        ]
        logger.info('Created optimizer with two lr groups')

        return torch.optim.Adam(parameters, lr=params['train.lr'], weight_decay=params['train.weight_decay'])
