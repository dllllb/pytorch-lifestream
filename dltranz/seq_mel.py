import pytorch_lightning as pl
import numpy as np

from dltranz.metric_learn.metric import metric_Recall_top_K
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

        self.validation_metric_params = params['validation_metric_params']

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
        recall_top_k = metric_Recall_top_K(y_h, y, **self.validation_metric_params)
        return {'recall_top_k': recall_top_k}

    def validation_epoch_end(self, outputs):
        recall_top_k = np.mean([x['recall_top_k'] for x in outputs])
        self.log('recall_top_k', recall_top_k, prog_bar=True)

    def configure_optimizers(self):
        params = self.hparams.params
        optimizer = get_optimizer(self, params)
        scheduler = get_lr_scheduler(optimizer, params)
        return [optimizer], [scheduler]
