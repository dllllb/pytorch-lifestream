import torch
import pytorch_lightning as pl
from ptls.data_load.padded_batch import PaddedBatch


class ABSModule(pl.LightningModule):
    """Abstract class for all modules in the project.
    Defines shared logic for training and validation steps.
    
    Args:
        validation_metric:
            Metric for validation step. It is used to calculate the quality of the model.
        seq_encoder:
            Model which calculate embeddings for original raw transaction sequences
            `seq_encoder` is trained by `CoLESModule` to get better representations of input sequences
        loss:
            loss object from `ptls.frames.coles.losses`.
            There are paired and triplet loss. They are required sampling strategy
            from `ptls.frames.coles.sampling_strategies`. Sampling strategy takes a relevant pairs or triplets from
            pairwise distance matrix.
        optimizer_partial:
            optimizer init partial. Network parameters are missed.
        lr_scheduler_partial:
            scheduler init partial. Optimizer are missed.

    """

    def __init__(self, 
                 validation_metric=None,
                 seq_encoder=None,
                 loss=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None):
        super().__init__()

        self._loss = loss
        self._seq_encoder = seq_encoder
        self._seq_encoder.is_reduce_sequence = self.is_requires_reduced_sequence
        self._validation_metric = validation_metric

        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial
        self._col_names = None
        self._seq_len = None
 
    @property
    def metric_name(self):
        raise NotImplementedError()

    @property
    def is_requires_reduced_sequence(self):
        raise NotImplementedError()

    def shared_step(self, x: PaddedBatch, y: torch.Tensor) -> tuple:
        """Method for shared logic between training and validation steps

        Args:
            x: features
            y: target

        Returns: y_h, y

        """
        raise NotImplementedError()


    @property
    def seq_encoder(self):
        return self._seq_encoder

    def forward(self, x):
        names = self._col_names
        seq_len = self._seq_len
        return self._seq_encoder(x, names, seq_len)

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

    def on_validation_epoch_end(self):
        self.log(f'valid/{self.metric_name}', self._validation_metric.compute(), prog_bar=True)
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
