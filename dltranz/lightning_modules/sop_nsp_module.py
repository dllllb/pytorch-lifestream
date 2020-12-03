import pytorch_lightning as pl
from torch.nn import BCELoss

from dltranz.baselines.sop import SentencePairsHead
from dltranz.seq_cls import EpochAuroc
from dltranz.seq_encoder import create_encoder
from dltranz.train import get_optimizer, get_lr_scheduler, ReduceLROnPlateauWrapper


class SopNspModule(pl.LightningModule):
    metric_name = 'valid_auroc'

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()

        self.loss = BCELoss()

        self._seq_encoder = create_encoder(params, is_reduce_sequence=True)
        self._head = SentencePairsHead(self._seq_encoder, self._seq_encoder.embedding_size, params['head'])

        self.validation_metric = EpochAuroc()

    @property
    def seq_encoder(self):
        return self._seq_encoder

    def forward(self, x):
        return self._seq_encoder(x)

    def training_step(self, batch, _):
        x, y = batch
        y_h = self._head(x)
        loss = self.loss(y_h, y)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_h = self._head(x)
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
