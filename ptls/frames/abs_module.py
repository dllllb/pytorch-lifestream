import torch
import pytorch_lightning as pl
from ptls.data_load.padded_batch import PaddedBatch


class ABSModule(pl.LightningModule):
    @property
    def metric_name(self):
        raise NotImplementedError()

    @property
    def is_requires_reduced_sequence(self):
        raise NotImplementedError()

    def shared_step(self, x, y):
        """

        Args:
            x:
            y:

        Returns: y_h, y

        """
        raise NotImplementedError()

    def __init__(self, validation_metric=None,
                 seq_encoder=None,
                 loss=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None):
        """
        Parameters
        ----------
        params : dict
            params for creating an encoder
        seq_encoder : torch.nn.Module
            sequence encoder, if not provided, will be constructed from params
        """
        super().__init__()
        # self.save_hyperparameters()

        self._loss = loss
        self._seq_encoder = seq_encoder
        self._seq_encoder.is_reduce_sequence = self.is_requires_reduced_sequence
        self._validation_metric = validation_metric

        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial

    @property
    def seq_encoder(self):
        return self._seq_encoder

    def forward(self, x):
        return self._seq_encoder(x)

    def training_step(self, batch, _):
        y_h, y = self.shared_step(*batch)
        out = self._loss(y_h, y)
        if type(out) is tuple:
            loss, info = out
            for k, v in info.items():
                self.log(k, v)
        else:
            self.log('loss', out)
        if type(batch) is tuple:
            x, y = batch
            if isinstance(x, PaddedBatch):
                self.log('seq_len', x.seq_lens.float().mean(), prog_bar=True)
        else:
            # this code should not be reached
            self.log('seq_len', -1, prog_bar=True)
            raise AssertionError('batch is not a tuple')
        return loss

    def validation_step(self, batch, _):
        y_h, y = self.shared_step(*batch)
        out = self._loss(y_h, y)
        if type(out) is tuple:
            loss, info = out
            for k, v in info.items():
                self.log("valid_" + k, v)
        else:
            self.log('valid_loss', out)
        self._validation_metric(y_h, y)

    def on_validation_epoch_end(self):
        metric = self._validation_metric.compute()
        if hasattr(self._validation_metric, "_multimetric"):
            for i, m in enumerate(metric):
                self.log(f'valid/{self.metric_name}'+"_"+str(i), m, prog_bar=True)
        else:
            self.log(f'valid/{self.metric_name}', metric, prog_bar=True)
        self._validation_metric.reset()

    def configure_optimizers(self):
        optimizer = self._optimizer_partial(self.parameters())
        scheduler = self._lr_scheduler_partial(optimizer)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = {
                'scheduler': scheduler,
                'monitor': self.metric_name,
            }
        return [optimizer], [scheduler]
