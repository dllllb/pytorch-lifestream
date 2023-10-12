import pytorch_lightning as pl
from ptls.frames.affiliation.losses import AffiliationLoss, SKLMetric
from ptls.frames.affiliation.modules import MergeProjectionHead
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
import torch
from ptls.data_load.padded_batch import PaddedBatch


class AffiliationModule(pl.LightningModule):
    def __init__(self,
                 seq_encoder_long: SeqEncoderContainer = None,
                 seq_encoder_short: SeqEncoderContainer = None,
                 head=None,
                 loss=None,
                 validation_metric=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None):
        super().__init__()

        if head is None:
            head = MergeProjectionHead(h_dims=[64, 64])

        if loss is None:
            loss = AffiliationLoss()

        if validation_metric is None:
            validation_metric = {metric: SKLMetric(metric) for metric in
                                 ['rocauc', 'f1', 'recall', 'precision', 'accuracy']}

        self._loss = loss
        self._seq_encoder_long = seq_encoder_long
        self._seq_encoder_short = seq_encoder_short
        self._seq_encoder_long.is_reduce_sequence = self.is_requires_reduced_sequence
        self._seq_encoder_short.is_reduce_sequence = self.is_requires_reduced_sequence
        self._validation_metric = validation_metric

        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial

        self._head = head

    @property
    def metric_name(self):
        return 'metric'

    @property
    def is_requires_reduced_sequence(self):
        return True

    @property
    def seq_encoder(self):
        return self._seq_encoder_long

    @property
    def seq_encoder_short(self):
        return self._seq_encoder_short

    def shared_step(self, long, short, y, n_repeats):
        embs_long = self._seq_encoder_long(long)
        embs_short = self._seq_encoder_long(short)
        y_h = self._head(embs_long, embs_short, n_repeats)
        return y_h, y

    def forward(self, x):
        return self._seq_encoder_long(x)

    def training_step(self, batch, _):
        y_h, y = self.shared_step(*batch)
        loss = self._loss(y_h, y)
        self.log('loss', loss)
        if type(batch) is tuple:
            long, short, y, _ = batch
            if isinstance(long, PaddedBatch):
                self.log('seq_len_long', long.seq_lens.float().mean(), prog_bar=True)
            if isinstance(short, PaddedBatch):
                self.log('seq_len_short', short.seq_lens.float().mean(), prog_bar=True)
        else:
            # this code should not be reached
            self.log('seq_len', -1, prog_bar=True)
            raise AssertionError('batch is not a tuple')
        return loss

    def validation_step(self, batch, _):
        y_h, y = self.shared_step(*batch)
        if type(self._validation_metric) == dict:
            for metric in self._validation_metric.values():
                metric(y_h, y)
        else:
            self._validation_metric(y_h, y)

    def on_validation_epoch_end(self):
        if type(self._validation_metric) == dict:
            for name, metric in self._validation_metric.items():
                self.log(name, metric.compute(), logger=True, prog_bar=True)
                metric.reset()
        else:
            self.log(self.metric_name, self._validation_metric.compute(), logger=True, prog_bar=True)
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
