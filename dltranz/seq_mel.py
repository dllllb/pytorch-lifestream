import pytorch_lightning as pl

from dltranz.train import get_optimizer, get_lr_scheduler
from dltranz.metric_learn.losses import get_loss
from dltranz.metric_learn.sampling_strategies import get_sampling_strategy
from dltranz.metric_learn.ml_models import ml_model_by_type


class SequenceMetricLearning(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.hparams = params

        self.sampling_strategy = get_sampling_strategy(params)
        self.loss = get_loss(params, self.sampling_strategy)

        model_f = ml_model_by_type(params['model_type'])
        self.model = model_f(params)

    def forward(self, x):
        self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_h = self(x)
        loss = self.loss(y_h, y)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.hparams)
        scheduler = get_lr_scheduler(optimizer, self.hparams)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
