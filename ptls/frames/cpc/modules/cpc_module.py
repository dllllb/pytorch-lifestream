import torch

from ptls.frames.abs_module import ABSModule
from ptls.frames.cpc.losses.cpc_loss import CPC_Loss
from ptls.frames.cpc.metrics.cpc_accuracy import CpcAccuracy
from ptls.nn.seq_encoder import RnnSeqEncoder
from ptls.data_load.padded_batch import PaddedBatch


class CpcModule(ABSModule):
    """Contrastive Predictive Coding ([CPC](https://arxiv.org/abs/1807.03748))

    Original sequence are encoded by `TrxEncoder`.
    Hidden representation `z` is an embedding for each individual transaction.
    Next `RnnEncoder` used for `context` calculation from `z`.
    Linear predictors are used to predict next trx embedding by context.
    The loss function tends to make future trx embedding and they predict closer.
    Negative sampling are used to avoid trivial solution.

    Parameters
        seq_encoder:
            Model which calculate embeddings for original raw transaction sequences
            `seq_encoder` is trained by `CoLESModule` to get better representations of input sequences
        head:
            Not used
        loss:
            Keep None. CPCLoss used by default
        validation_metric:
            Keep None. CPCAccuracy used by default
        optimizer_partial:
            optimizer init partial. Network parameters are missed.
        lr_scheduler_partial:
            scheduler init partial. Optimizer are missed.

    """
    def __init__(self, validation_metric=None,
                       seq_encoder=None,
                       head=None,
                       n_negatives=40, n_forward_steps=6,
                       optimizer_partial=None,
                       lr_scheduler_partial=None):

        self.save_hyperparameters('n_negatives', 'n_forward_steps')

        loss = CPC_Loss(n_negatives=n_negatives, n_forward_steps=n_forward_steps)

        if validation_metric is None:
            validation_metric = CpcAccuracy(loss)

        if seq_encoder is not None and not isinstance(seq_encoder, RnnSeqEncoder):
            raise NotImplementedError(f'Only rnn encoder supported in CpcModule. Found {type(seq_encoder)}')

        seq_encoder.seq_encoder.is_reduce_sequence = False

        super().__init__(validation_metric,
                         seq_encoder,
                         loss,
                         optimizer_partial,
                         lr_scheduler_partial)

        linear_size = self.seq_encoder.trx_encoder.output_size
        embedding_size = self.seq_encoder.embedding_size
        self._linears = torch.nn.ModuleList([torch.nn.Linear(embedding_size, linear_size)
                                             for _ in range(loss.n_forward_steps)])

    @property
    def metric_name(self):
        return 'cpc_accuracy'

    @property
    def is_requires_reduced_sequence(self):
        return False

    def shared_step(self, x, y):
        trx_encoder = self._seq_encoder.trx_encoder
        seq_encoder = self._seq_encoder.seq_encoder

        base_embeddings = trx_encoder(x)
        context_embeddings = seq_encoder(base_embeddings)

        me = []
        for l in self._linears:
            me.append(l(context_embeddings.payload))
        mapped_ctx_embeddings = PaddedBatch(torch.stack(me, dim=3), context_embeddings.seq_lens)

        return (base_embeddings, context_embeddings, mapped_ctx_embeddings), y
