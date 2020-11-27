import pytorch_lightning as pl
import numpy as np

from dltranz.agg_feature_model import AggFeatureModel
from dltranz.metric_learn.metric import BatchRecallTopPL
from dltranz.train import get_optimizer, get_lr_scheduler
from dltranz.metric_learn.losses import get_loss
from dltranz.metric_learn.sampling_strategies import get_sampling_strategy
from dltranz.metric_learn.ml_models import ml_model_by_type


class SequenceMetricLearning(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()

        self.sampling_strategy = get_sampling_strategy(params)
        self.loss = get_loss(params, self.sampling_strategy)

        model_f = ml_model_by_type(params['model_type'])
        self.model = model_f(params)

        self.validation_metric = BatchRecallTopPL(**params['validation_metric_params'])

    @property
    def category_max_size(self):
        params = self.hparams.params
        return {k: v['in'] for k, v in params['trx_encoder.embeddings'].items()}

    @property
    def embedding_size(self):
        params = self.hparams.params
        if params['model_type'] == 'rnn':
            return params['rnn.hidden_size']
        if params['model_type'] == 'transf':
            return params['transf.input_size']
        if params['model_type'] == 'agg_features':
            return AggFeatureModel.output_size(params['trx_encoder'])
        if params['model_type'] == 'cpc_model':
            return params['rnn.hidden_size']

        raise NotImplementedError(f'Unknown model_type {params.model_type}')

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
        self.validation_metric(y_h, y)

    def validation_epoch_end(self, outputs):
        self.log('recall_top_k', self.validation_metric.compute(), prog_bar=True)

    def configure_optimizers(self):
        params = self.hparams.params
        optimizer = get_optimizer(self, params)
        scheduler = get_lr_scheduler(optimizer, params)
        return [optimizer], [scheduler]
