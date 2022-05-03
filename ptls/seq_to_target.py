import logging
from copy import deepcopy

import pytorch_lightning as pl
import torch
import numpy as np
from torchmetrics.functional.classification import auroc
import torchmetrics

from ptls.loss import get_loss, cross_entropy, kl, mape_metric, mse_loss, r_squared
from ptls.seq_encoder import create_encoder
from ptls.train import get_optimizer, get_lr_scheduler
from ptls.models import create_head_layers
from ptls.trx_encoder import PaddedBatch
from collections import defaultdict


logger = logging.getLogger(__name__)


class EpochAuroc(torchmetrics.Metric):
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
        return auroc(y_hat, y.long())


class DistributionTargets(torchmetrics.Metric):
    def __init__(self, col_name):
        super().__init__(compute_on_step=False)

        self.add_state('y_hat', default=[])
        self.add_state('y', default=[])
        self.col_name = col_name
        self.sign = -1 if self.col_name == 'neg_sum' else 1

    def update(self, y_hat, y):
        y_hat = y_hat[self.col_name]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y = torch.tensor(np.array(y[self.col_name].tolist(), dtype='float64'), device=device)
        self.y_hat.append(y_hat)
        self.y.append(y)


class CrossEntropy(DistributionTargets):
    def __init__(self, col_name):
        super().__init__(col_name)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)
        return cross_entropy(y_hat, y)


class KL(DistributionTargets):
    def __init__(self, col_name):
        super().__init__(col_name)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)
        return kl(y_hat, y)


class MSE(DistributionTargets):
    def __init__(self, col_name):
        super().__init__(col_name)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)
        return mse_loss(y_hat, torch.log(self.sign * y[:, None] + 1))


class MAPE(DistributionTargets):
    def __init__(self, col_name):
        super().__init__(col_name)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)
        return mape_metric(self.sign * torch.exp(y_hat - 1), y[:, None])


class LogAccuracy(torchmetrics.Accuracy):
    def __init__(self, **params):
        super().__init__(**params)

        self.add_state('correct', default=torch.tensor([0.0]))
        self.add_state('total', default=torch.tensor([0.0]))

    def update(self, preds, target):
        if len(preds.shape) == 1:
            preds = (preds > 0.5).to(dtype=target.dtype)
        else:
            preds = torch.argmax(preds, dim=1)
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class R_squared(DistributionTargets):
    def __init__(self, col_name):
        super().__init__(col_name)

    def compute(self):
        y_hat = torch.cat(self.y_hat)
        y = torch.cat(self.y)
        return r_squared(self.sign * torch.exp(y_hat - 1), y[:, None])


class SequenceToTarget(pl.LightningModule):
    def __init__(self, params, pretrained_encoder=None):
        super().__init__()
        self.train_update_n_steps = params.get('train_update_n_steps', None)

        self.metrics_test = defaultdict(list)  # here we accumulate metrics on each test end
        self.metrics_train = defaultdict(list)  # here we accumulate metrics on some train bathes (called outside)

        head_params = dict(params['head_layers']).get('CombinedTargetHeadFromRnn', None)
        self.pos, self.neg = (head_params.get('pos', True), head_params.get('neg', True)) if head_params else (0, 0)
        self.cols_ix = params.get('columns_ix', {'neg_sum': 0,
                                                 'neg_distribution': 1,
                                                 'pos_sum': 2,
                                                 'pos_distribution': 3})

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
            'accuracy': LogAccuracy(),
            'R2n': R_squared('neg_sum'),
            'MSEn': MSE('neg_sum'),
            'MAPEn': MAPE('neg_sum'),
            'R2p': R_squared('pos_sum'),
            'MSEp': MSE('pos_sum'),
            'MAPEp': MAPE('pos_sum'),
            'CEn': CrossEntropy('neg_distribution'),
            'CEp': CrossEntropy('pos_distribution'),
            'KLn': KL('neg_distribution'),
            'KLp': KL('pos_distribution')
        }
        params_score_metric = params['score_metric']
        if type(params_score_metric) is str:
            params_score_metric = [params_score_metric]
        metric_cls = [(name, d_metrics[name]) for name in params_score_metric]

        self.train_metrics = torch.nn.ModuleDict([(name, mc) for name, mc in metric_cls])
        self.valid_metrics = torch.nn.ModuleDict([(name, mc) for name, mc in deepcopy(metric_cls)])
        self.test_metrics = torch.nn.ModuleDict([(name, mc) for name, mc in deepcopy(metric_cls)])


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
        if isinstance(x, PaddedBatch):
            self.log('seq_len', x.seq_lens.float().mean(), prog_bar=True)
        if self.train_update_n_steps and self.global_step % self.train_update_n_steps == 0:
            for name, mf in self.train_metrics.items():
                mf(y_h, y)
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
            value = mf.compute().item()
            self.log(f'test_{name}', value, prog_bar=True)
            self.metrics_test[name] += [value]

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
