import torch

from ptls.frames.abs_module import ABSModule
from ptls.frames.coles.losses import ContrastiveLoss
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.frames.coles.sampling_strategies.hard_negative_pair_selector import HardNegativePairSelector
from ptls.nn.head import Head
from ptls.nn.seq_encoder.containers import SeqEncoderContainer


class ColesSupervisedModule(ABSModule):
    """Contrastive Learning for Event Sequences ([CoLES](https://arxiv.org/abs/2002.08232)) (unsupervised)
    with auxiliary loss based on class labels from dataset (supervised, works for labeled data)

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
            This loss applied for contrastive learning at augmentation subsample labels
            loss object from `ptls.frames.coles.losses`.
            There are paired and triplet loss. They are required sampling strategy
            from `ptls.frames.coles.sampling_strategies`. Sampling strategy takes a relevant pairs or triplets from
            pairwise distance matrix.
        l_loss:
            This loss applied for contrastive learning at auxiliary class labels
            loss object from `ptls.frames.coles.losses`.
            There are paired and triplet loss. They are required sampling strategy
            from `ptls.frames.coles.sampling_strategies`. Sampling strategy takes a relevant pairs or triplets from
            pairwise distance matrix.
        supervised_loss_w:
            weight for auxiliary losses
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
                 l_loss=None,
                 contrastive_loss_w=1.0,
                 supervised_loss_w=0.1,
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
        self.l_loss = l_loss
        self.contrastive_loss_w = contrastive_loss_w
        self.supervised_loss_w = supervised_loss_w

    @property
    def metric_name(self):
        return 'recall_top_k'

    @property
    def is_requires_reduced_sequence(self):
        return True

    def shared_step(self, x, y, l):
        y_h = self(x)
        if self._head is not None:
            y_h = self._head(y_h)
        return y_h, y, l

    def training_step(self, batch, _):
        x = batch[0]
        y_h, y, labels = self.shared_step(*batch)

        loss = self._loss(y_h, y)  # unsupervised loss

        # supervised losses
        l_loss = 0.0
        for label_ix in range(labels.size(1)):
            l = labels[:, label_ix]
            ix_labeled = l >= 0
            if ix_labeled.sum() == 0:
                continue
            l_loss_i = self.l_loss(y_h[ix_labeled], l[ix_labeled])
            self.log(f'loss_{label_ix}/loss', l_loss_i)
            l_loss = l_loss + l_loss_i

            l_unique, l_counts = torch.unique(l, return_counts=True)
            for _l, _c in zip(l_unique, l_counts.float()):
                self.log(f'loss_{label_ix}/label_{_l.item()}_count', _c)

        self.log('loss', loss)
        self.log('seq_len', x.seq_lens.float().mean(), prog_bar=True)
        return self.contrastive_loss_w * loss + self.supervised_loss_w * l_loss
        # return self.supervised_loss_w * l_loss

    def validation_step(self, batch, _):
        y_h, y, l = self.shared_step(*batch)
        self._validation_metric(y_h, y)

    def validation_epoch_end(self, outputs):
        self.log(f'valid/{self.metric_name}', self._validation_metric.compute(), prog_bar=True)
        self._validation_metric.reset()
