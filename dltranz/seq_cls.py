import pytorch_lightning as pl

from dltranz.loss import get_loss
from dltranz.train import get_optimizer, get_lr_scheduler
from dltranz.models import model_by_type


class SequenceClassify(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()

        self.loss = get_loss(params)

        model_f = model_by_type(params['model_type'])
        self.model = model_f(params)

        # metrics
        self.train_accuracy = pl.metrics.Accuracy()
        self.valid_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        loss = self.loss(y_h, y)
        self.log('loss', loss)
        self.log('train_acc_step', self.train_accuracy(y_h, y))
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_acc_epoch', self.train_accuracy.compute())

    def validation_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        self.log('valid_acc_step', self.valid_accuracy(y_h, y))

    def validation_epoch_end(self, outputs):
        self.log('valid_acc_epoch', self.valid_accuracy.compute())

    def test_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        self.log('test_acc_step', self.test_accuracy(y_h, y))

    def test_epoch_end(self, outputs):
        self.log('test_acc_epoch', self.test_accuracy.compute())

    def configure_optimizers(self):
        params = self.hparams.params
        optimizer = get_optimizer(self, params)
        scheduler = get_lr_scheduler(optimizer, params)
        return [optimizer], [scheduler]
