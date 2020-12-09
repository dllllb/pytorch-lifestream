import pytorch_lightning as pl
from dltranz.seq_encoder import create_encoder
from dltranz.train import get_optimizer, get_lr_scheduler, ReduceLROnPlateauWrapper


class ABSModule(pl.LightningModule):
    @property
    def metric_name(self):
        raise NotImplementedError()

    @property
    def is_requires_reduced_sequence(self):
        raise NotImplementedError()

    def get_loss(self):
        raise NotImplementedError()

    def get_validation_metric(self):
        raise NotImplementedError()

    def shared_step(self, x, y):
        """

        Args:
            x:
            y:

        Returns: y_h, y

        """
        raise NotImplementedError()

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()

        self._loss = self.get_loss()
        self._seq_encoder = create_encoder(params, is_reduce_sequence=self.is_requires_reduced_sequence)
        self._validation_metric = self.get_validation_metric()

    @property
    def seq_encoder(self):
        return self._seq_encoder

    def forward(self, x):
        return self._seq_encoder(x)

    def training_step(self, batch, _):
        y_h, y = self.shared_step(*batch)
        loss = self._loss(y_h, y)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        if self._head is not None:
            y_h = self._head(y_h)
        self._validation_metric(y_h, y)

    def validation_epoch_end(self, outputs):
        self.log(self.metric_name, self._validation_metric.compute(), prog_bar=True)

    def configure_optimizers(self):
        params = self.hparams.params
        optimizer = get_optimizer(self, params)
        scheduler = get_lr_scheduler(optimizer, params)
        if isinstance(scheduler, ReduceLROnPlateauWrapper):
            scheduler = {
                'scheduler': scheduler,
                'monitor': self.metric_name,
            }
        return [optimizer], [scheduler]

