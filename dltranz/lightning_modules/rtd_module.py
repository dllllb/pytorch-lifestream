import pytorch_lightning as pl
import torch
from torch.nn import BCELoss

from dltranz.custom_layers import Squeeze
from dltranz.seq_cls import EpochAuroc
from dltranz.seq_encoder import create_encoder
from dltranz.seq_encoder.utils import AllStepsHead, FlattenHead
from dltranz.train import get_optimizer, get_lr_scheduler, ReduceLROnPlateauWrapper


class RtdModule(pl.LightningModule):
    metric_name = 'valid_auroc'

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()

        self.loss = BCELoss()

        self._seq_encoder = create_encoder(params, is_reduce_sequence=False)
        self._head = torch.nn.Sequential(
            AllStepsHead(
                torch.nn.Sequential(
                    torch.nn.Linear(self._seq_encoder.embedding_size, 1),
                    torch.nn.Sigmoid(),
                    Squeeze(),
                )
            ),
            FlattenHead(),
        )

        self.validation_metric = EpochAuroc()

    @property
    def seq_encoder(self):
        return self._seq_encoder

    def forward(self, x):
        return self._seq_encoder(x)

    def training_step(self, batch, _):
        x, y = batch
        y_h = self._seq_encoder(x)
        y_h = self._head(y_h)
        loss = self.loss(y_h, y)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_h = self._seq_encoder(x)
        y_h = self._head(y_h)
        self.validation_metric(y_h, y)

    def validation_epoch_end(self, outputs):
        self.log(self.metric_name, self.validation_metric.compute(), prog_bar=True)

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
