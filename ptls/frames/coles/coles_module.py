from ptls.frames.abs_module import ABSModule
from ptls.frames.coles.losses.contrastive_loss import ContrastiveLoss
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.frames.coles.sampling_strategies.hard_negative_pair_selector import HardNegativePairSelector
from ptls.nn.head import Head
from ptls.nn.seq_encoder.containers import SeqEncoderContainer


class CoLESModule(ABSModule):
    """Contrastive Learning for Event Sequences ([CoLES](https://arxiv.org/abs/2002.08232))

    Subsequences are sampled from original sequence.
    Samples from the same sequence are `positive` examples
    Samples from the different sequences are `negative` examples
    Embeddings for all samples are calculated.
    Paired distances between all embeddings are calculated.
    The loss function tends to make positive distances smaller and negative ones larger.

    Parameters
        seq_encoder:
            Model which calculate embeddings for original raw transaction sequences
            `seq_encoder` is trained by `CoLESModule` to get better representations of input sequences
        head:
            Model which helps to train. Not used during inference
            Can be normalisation layer which make embedding l2 length equals 1
            Can be MLP as `projection head` like in SymCLR framework.
        loss:
            loss object from `ptls.frames.coles.losses`.
            There are paired and triplet loss. They are required sampling strategy
            from `ptls.frames.coles.sampling_strategies`. Sampling strategy takes a relevant pairs or triplets from
            pairwise distance matrix.
        validation_metric:
            Keep None. `ptls.frames.coles.metric.BatchRecallTopK` used by default.
        optimizer_partial:
            optimizer init partial. Network parameters are missed.
        lr_scheduler_partial:
            scheduler init partial. Optimizer are missed.

    """
    def __init__(self,
                 seq_encoder: SeqEncoderContainer = None,
                 head=None,
                 loss=None,
                 validation_metric=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None):
        if head is None:
            head = Head(use_norm_encoder=True)

        if loss is None:
            loss = ContrastiveLoss(margin=0.5,
                                   sampling_strategy=HardNegativePairSelector(neg_count=5))

        if validation_metric is None:
            validation_metric = BatchRecallTopK(K=4, metric='cosine')

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