from ptls.lightning_modules.AbsModule import ABSModule
from ptls.metric_learn.losses import ContrastiveLoss
from ptls.metric_learn.sampling_strategies import HardNegativePairSelector
from ptls.metric_learn.metric import BatchRecallTopPL
from ptls.models import Head
from ptls.seq_encoder.abs_seq_encoder import AbsSeqEncoder


class CoLESModule(ABSModule):
    def __init__(self, validation_metric=None,
                       seq_encoder: AbsSeqEncoder=None,
                       head=None,
                       loss=None,
                       optimizer_partial=None,
                       lr_scheduler_partial=None):

        if loss is None:
            sampling_strategy = HardNegativePairSelector(neg_count=5)
            loss = ContrastiveLoss(margin=0.5,
                                   sampling_strategy=sampling_strategy)

        if validation_metric is None:
            validation_metric = BatchRecallTopPL(K=4, metric='cosine')

        if head is None:
            head = Head(input_size=seq_encoder.embedding_size, use_norm_encoder=True)

        super().__init__(validation_metric,
                         seq_encoder,
                         loss,
                         optimizer_partial,
                         lr_scheduler_partial)

        self._head = head

    @property
    def metric_name(self):
        return 'recall_top_k'

    @property
    def is_requires_reduced_sequence(self):
        return True

    def shared_step(self, x, y):
        y_h = self(x)
        if self._head is not None:
            y_h = self._head(y_h)
        return y_h, y
