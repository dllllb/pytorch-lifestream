import torch

from ptls.lightning_modules.AbsModule import ABSModule
from ptls.trx_encoder import PaddedBatch
from ptls.lightning_modules.cpc_module import CPC_Loss, CpcAccuracyPL
from ptls.seq_encoder.rnn_encoder import RnnEncoder


class CpcV2Module(ABSModule):
    def __init__(self, validation_metric=None,
                       seq_encoder=None,
                       head=None,
                       loss=None,
                       aggregator=None,
                       optimizer_partial=None,
                       lr_scheduler_partial=None):

        if loss is None:
            loss = CPC_Loss(n_negatives=40, n_forward_steps=6)
        if validation_metric is None:
            validation_metric = CpcAccuracyPL(loss)

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
