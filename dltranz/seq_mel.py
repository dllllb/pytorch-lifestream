import pytorch_lightning as pl


class SequenceMetricLearning(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.hparams = params

    def forward(self, x):
        pass

    def training_step(self, batch, batch_nb):
        pass

    def configure_optimizers(self):
        pass
