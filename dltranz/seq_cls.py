import logging
from copy import deepcopy

import pytorch_lightning as pl
import torch
import numpy as np
from pytorch_lightning.metrics.functional.classification import auroc

from dltranz.loss import get_loss, cross_entropy, mape_metric, mse_loss, r_squared
from dltranz.seq_encoder import create_encoder
from dltranz.train import get_optimizer, get_lr_scheduler
from dltranz.models import create_head_layers

logger = logging.getLogger(__name__)


# columns indexes for distribution targets task:
COLUMNS_IX = {'neg_sum': 0,
              'neg_distribution': 1, 
              'pos_sum': 2, 
              'pos_distribution': 3}


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


class DistributionTargets(pl.metrics.Metric):
    def __init__(self, col_ix):
        super().__init__(compute_on_step=False)

        self.add_state('y_hat', default=[])
        self.add_state('y', default=[])
        self.col_ix = col_ix
        self.sign = -1 if self.col_ix == COLUMNS_IX['neg_sum'] else 1

    def update(self, y_hat, y):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y = torch.tensor(np.array(y[:, self.col_ix].tolist()), device=device)
        y_hat = y_hat[self.col_ix]
        self.y_hat.append(y_hat)
        self.y.append(y)


class CrossEntropy(DistributionTargets):
    def __init__(self, col_ix):
        super().__init__(col_ix)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)
        return cross_entropy(y_hat, y)


class MSE(DistributionTargets):
    def __init__(self, col_ix):
        super().__init__(col_ix)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)
        return mse_loss(y_hat, torch.log(self.sign * y[:, None] + 1))


class MAPE(DistributionTargets):
    def __init__(self, col_ix):
        super().__init__(col_ix)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)
        return mape_metric(self.sign * torch.exp(y_hat - 1), y[:, None])


class R_squared(DistributionTargets):
    def __init__(self, col_ix):
        super().__init__(col_ix)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)
        return r_squared(self.sign * torch.exp(y_hat - 1), y[:, None])


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
            'auroc': EpochAuroc(),
            'accuracy': pl.metrics.Accuracy(),
            'R2n': R_squared(COLUMNS_IX['neg_sum']),
            'MSEn': MSE(COLUMNS_IX['neg_sum']),
            'MAPEn': MAPE(COLUMNS_IX['neg_sum']),
            'R2p': R_squared(COLUMNS_IX['pos_sum']),
            'MSEp': MSE(COLUMNS_IX['pos_sum']),
            'MAPEp': MAPE(COLUMNS_IX['pos_sum']),
            'CEn': CrossEntropy(COLUMNS_IX['neg_distribution']),
            'CEp': CrossEntropy(COLUMNS_IX['pos_distribution'])
        }
        params_score_metric = params['score_metric']
        if type(params_score_metric) is str:
            params_score_metric = [params_score_metric]
        metric_cls = [(name, d_metrics[name]) for name in params_score_metric]

        self.valid_metrics = torch.nn.ModuleDict([(name, mc) for name, mc in metric_cls])
        self.test_metrics = torch.nn.ModuleDict([(name, mc) for name, mc in metric_cls])


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
            self.log(f'val_{name}', mf.compute(), prog_bar=True)

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
