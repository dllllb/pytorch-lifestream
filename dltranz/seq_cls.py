import logging

import pytorch_lightning as pl
import torch

from dltranz.loss import get_loss
from dltranz.train import get_optimizer, get_lr_scheduler
from dltranz.models import model_by_type


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
        return pl.metrics.functional.classification.auroc(y_hat, y)


class SequenceClassify(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()

        self.loss = get_loss(params)

        model_f = model_by_type(params['model_type'])
        self.model = model_f(params)

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
    def category_max_size(self):
        params = self.hparams.params

        if params['model_type'] == 'pretrained':
            return self.model[0].category_max_size
        return {k: v['in'] for k, v in params['trx_encoder.embeddings'].items()}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        loss = self.loss(y_h, y)
        self.log('loss', loss)
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
        if params['model_type'] == 'pretrained':
            optimizer = self.get_pretrained_optimizer()
        else:
            optimizer = get_optimizer(self, params)
        scheduler = get_lr_scheduler(optimizer, params)
        return [optimizer], [scheduler]

    def get_pretrained_optimizer(self):
        params = self.hparams.params

        p_model = self.model[0]
        head = self.model[1]

        if params['pretrained.lr'] == 'freeze':
            p_model.freeze()
            logger.info('Created optimizer with frozen encoder')
            return get_optimizer(self, params)

        parameters = [
            {'params': p_model.parameters(), 'lr': params['pretrained.lr']},
            {'params': head.parameters(), 'lr': params['train.lr']},
        ]
        logger.info('Created optimizer with two lr groups')

        return torch.optim.Adam(parameters, lr=params['train.lr'], weight_decay=params['train.weight_decay'])
