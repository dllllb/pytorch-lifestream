import pytorch_lightning as pl
from pyhocon import ConfigFactory

from dltranz.loss import get_loss
from dltranz.train import get_optimizer, get_lr_scheduler
from dltranz.models import model_by_type


class SequenceClassify(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.hparams = params

        self.loss = get_loss(params)

        model_f = model_by_type(params['model_type'])
        self.model = model_f(params)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        loss = self.loss(y_h, y)
        return pl.TrainResult(minimize=loss)

    def configure_optimizers(self):
        params = ConfigFactory.from_dict(self.hparams)
        optimizer = get_optimizer(self, params)
        scheduler = get_lr_scheduler(optimizer, params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
