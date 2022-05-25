import pytorch_lightning as pl
from hydra.utils import instantiate
from ptls.seq_encoder import create_encoder
from ptls.train import get_optimizer, get_lr_scheduler, ReduceLROnPlateauWrapper
from ptls.trx_encoder import PaddedBatch


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
                       optimizer=None,
                       lr_scheduler_wrapper=None,
                       lr_scheduler=None):
        """
        Parameters
        ----------
        params : dict
            params for creating an encoder
        seq_encoder : torch.nn.Module
            sequence encoder, if not provided, will be constructed from params
        """
        super().__init__()
        self.save_hyperparameters()

        self._loss = lambda *args, **kwargs: loss(*args, **kwargs)[0]
        self._seq_encoder = seq_encoder(is_reduce_sequence=self.is_requires_reduced_sequence)
        self._validation_metric = validation_metric

        self._optimizer = optimizer
        self._lr_scheduler_wrapper = lr_scheduler_wrapper
        self._lr_scheduler = lr_scheduler

    @property
    def seq_encoder(self):
        return self._seq_encoder

    def forward(self, x):
        return self._seq_encoder(x)

    def training_step(self, batch, _):
        y_h, y = self.shared_step(*batch)
        loss = self._loss(y_h, y)
        self.log('loss', loss)
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
        self._validation_metric(y_h, y)

    def validation_epoch_end(self, outputs):
        self.log(self.metric_name, self._validation_metric.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = self._optimizer(self.parameters())
        scheduler = self._lr_scheduler_wrapper(self._lr_scheduler(optimizer))
        if isinstance(scheduler, ReduceLROnPlateauWrapper):
            scheduler = {
                'scheduler': scheduler,
                'monitor': self.metric_name,
            }
        return [optimizer], [scheduler]

