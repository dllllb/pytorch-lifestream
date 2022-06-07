import torch
from functools import partial

from ptls.lightning_modules.AbsModule import ABSModule
from ptls.metric_learn.losses import ContrastiveLoss
from ptls.metric_learn.sampling_strategies import HardNegativePairSelector
from ptls.metric_learn.metric import BatchRecallTopPL


class EmbModule(ABSModule):
    """Deprecated. The same as `ptls.lightning_modules.coles_module.CoLESModule`

    pl.LightningModule for training CoLES embeddings

    Parameters
    ----------
    seq_encoder : torch.nn.Module
        sequence encoder
    head : torch.nn.Module
        head of th model
    loss : torch.nn.Module
        Loss function for training
    lr : float. Default: 1e-3
        Learning rate
    weight_decay: float. Default: 0.0
        weight decay for optimizer
    lr_scheduler_step_size : int. Default: 100
        Period of learning rate decay.
    lr_scheduler_step_gamma: float. Default: 0.1
        Multiplicative factor of learning rate decay.
    """
    def __init__(
        self,
        seq_encoder: torch.nn.Module,
        head: torch.nn.Module,
        loss: torch.nn.Module = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler_step_size: int = 100,
        lr_scheduler_step_gamma: float = 0.1,
        validation_metric = None):

        if loss is None:
            sampling_strategy = HardNegativePairSelector(neg_count=5)
            loss = ContrastiveLoss(margin=0.5,
                                   sampling_strategy=sampling_strategy)

            optimizer_partial = partial(torch.optim.Adam,
                                        lr=lr,
                                        weight_decay=weight_decay)
            lr_scheduler_partial = partial(torch.optim.lr_scheduler.StepLR,
                                           step_size=lr_scheduler_step_size,
                                           gamma=lr_scheduler_step_gamma)

        if validation_metric is None:
            validation_metric = BatchRecallTopPL(K=4, metric='cosine')

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
