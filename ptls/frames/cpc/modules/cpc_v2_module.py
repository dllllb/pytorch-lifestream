import torch

from ptls.frames.abs_module import ABSModule
from ptls.frames.cpc.losses.cpc_loss import CPC_Loss
from ptls.frames.cpc.metrics.cpc_accuracy import CpcAccuracy
from ptls.nn.seq_encoder import RnnSeqEncoder
from ptls.data_load.padded_batch import PaddedBatch


class CpcV2Module(ABSModule):
    """Contrastive Predictive Coding ([CPC](https://arxiv.org/abs/1807.03748))

    Original sequence splits at small parts which are encoded by `seq_encoder` with reduce.
    Hidden representation `z` is an embedding vector for small sequence split.
    Next `aggregator` `RnnEncoder` used for `context` calculation from `z`.
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
        aggregator:
            RnnEncoder which are used to calculate context
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
                       aggregator=None,
                       optimizer_partial=None,
                       lr_scheduler_partial=None):

        self.save_hyperparameters('n_negatives', 'n_forward_steps')

        loss = CPC_Loss(n_negatives=n_negatives, n_forward_steps=n_forward_steps)
        if validation_metric is None:
            validation_metric = CpcAccuracy(loss)

        if seq_encoder is not None and not isinstance(seq_encoder, RnnSeqEncoder):
            raise NotImplementedError(f'Only rnn encoder supported in CpcModule. Found {type(seq_encoder)}')

        super().__init__(validation_metric,
                         seq_encoder,
                         loss,
                         optimizer_partial,
                         lr_scheduler_partial)

        # Define aggregator
        self.agg_rnn = aggregator

        # Define linear transformation for every cpc.n_forward_steps
        self.n_forward_steps = loss.n_forward_steps
        embedding_size = aggregator.hidden_size
        linear_size = seq_encoder.rnn_encoder.hidden_size
        self._linears = torch.nn.ModuleList([torch.nn.Linear(embedding_size, linear_size)
                                             for _ in range(self.n_forward_steps)])


    @property
    def metric_name(self):
        return 'cpc_accuracy'

    @property
    def is_requires_reduced_sequence(self):
        return True

    def shared_step(self, x, y):
        """
        x: list of PaddedBatch,
        where PaddedBatch = (
            payload = {key: [padded tensor] whit shape [batch_size, max_subseq_len]},
            seq_lens = (1 x batch_size)
        )
        """
        split_embeddings = []
        for padded_batch in x:
            split_embeddings.append(self._seq_encoder(padded_batch))

        base_embeddings = PaddedBatch(torch.stack(split_embeddings, dim=1), None)
        context_embeddings = self.agg_rnn(base_embeddings)


        me = []
        for l in self._linears:
            me.append(l(context_embeddings.payload))

        batch_size, max_seq_len, _ = base_embeddings.payload.shape

        mapped_ctx_embeddings = PaddedBatch(
            torch.stack(me, dim=3),
            max_seq_len * torch.ones(batch_size, device=x[0].device)
        )
        return (base_embeddings, context_embeddings, mapped_ctx_embeddings), y

    def validation_step(self, batch, _):
        y_h, y = self.shared_step(batch, None)
        self._validation_metric(y_h, y)

    def training_step(self, batch, _):
        y_h, y = self.shared_step(batch, None)
        loss = self._loss(y_h, y)
        self.log('loss', loss)
        return loss
