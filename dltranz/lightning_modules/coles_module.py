import pytorch_lightning as pl

from dltranz.metric_learn.metric import BatchRecallTopPL
from dltranz.models import create_head
from dltranz.seq_encoder import create_encoder
from dltranz.train import get_optimizer, get_lr_scheduler
from dltranz.metric_learn.losses import get_loss
from dltranz.metric_learn.sampling_strategies import get_sampling_strategy


class CoLESModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()

        self.sampling_strategy = get_sampling_strategy(params)
        self.loss = get_loss(params, self.sampling_strategy)

        self._seq_encoder = create_encoder(params, is_reduce_sequence=True)
        self._head = create_head(params)

        self.validation_metric = BatchRecallTopPL(**params['validation_metric_params'])

    @property
    def category_max_size(self):
        return self._seq_encoder.category_max_size

    @property
    def embedding_size(self):
        return self._seq_encoder.embedding_size

    def forward(self, x):
        return self._seq_encoder(x)

    def training_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        if self._head is not None:
            y_h = self._head(y_h)
        loss = self.loss(y_h, y)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        if self._head is not None:
            y_h = self._head(y_h)
        self.validation_metric(y_h, y)

    def validation_epoch_end(self, outputs):
        self.log('recall_top_k', self.validation_metric.compute(), prog_bar=True)

    def configure_optimizers(self):
        params = self.hparams.params
        optimizer = get_optimizer(self, params)
        scheduler = get_lr_scheduler(optimizer, params)
        return [optimizer], [scheduler]
