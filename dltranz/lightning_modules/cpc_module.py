import pytorch_lightning as pl
import torch

from dltranz.baselines.cpc import CPC_Loss, CpcAccuracyPL
from dltranz.seq_encoder import create_encoder
from dltranz.train import get_optimizer, get_lr_scheduler, ReduceLROnPlateauWrapper
from dltranz.trx_encoder import PaddedBatch


class CpcModule(pl.LightningModule):
    metric_name = 'cpc_accuracy'

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()

        self.loss = CPC_Loss(n_negatives=params['cpc.n_negatives'])

        if params['encoder_type'] != 'rnn':
            raise NotImplementedError(f'Only rnn encoder supported in CpcModule. Found {params.encoder_type}')

        self._seq_encoder = create_encoder(params, is_reduce_sequence=False)
        linear_size = self._seq_encoder.model[0].output_size
        embedding_size = self._seq_encoder.embedding_size
        self._linears = torch.nn.ModuleList([torch.nn.Linear(embedding_size, linear_size)
                                             for _ in range(params['cpc.n_forward_steps'])])

        self.validation_metric = CpcAccuracyPL(self.loss)

    @property
    def seq_encoder(self):
        return self._seq_encoder

    def forward(self, x):
        return self._seq_encoder(x)

    def shared_step(self, x):
        trx_encoder = self._seq_encoder.model[0]
        seq_encoder = self._seq_encoder.model[1]

        base_embeddings = trx_encoder(x)
        context_embeddings = seq_encoder(base_embeddings)

        me = []
        for l in self._linears:
            me.append(l(context_embeddings.payload))
        mapped_ctx_embeddings = PaddedBatch(torch.stack(me, dim=3), context_embeddings.seq_lens)

        return base_embeddings, context_embeddings, mapped_ctx_embeddings

    def training_step(self, batch, _):
        x, y = batch
        embeddings = self.shared_step(x)
        loss = self.loss(embeddings, y)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        embeddings = self.shared_step(x)
        self.validation_metric(embeddings, y)

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
