import torch
import torchmetrics

from ptls.loss import BCELoss
from ptls.custom_layers import Squeeze
from ptls.frames.abs_module import ABSModule
from ptls.nn.seq_encoder.utils import AllStepsHead, FlattenHead


class RtdModule(ABSModule):
    """Replaced Token Detection (RTD) from [ELECTRA](https://arxiv.org/abs/2003.10555)

    Original sequence are encoded by `TrxEncoder`.
    Randomly sampled trx representations are replaced by an other trx representation from batch
    `seq_encoder` find replaced transactions.

    """
    def __init__(self, validation_metric=None,
                       seq_encoder=None,
                       head=None,
                       loss=None,
                       optimizer_partial=None,
                       lr_scheduler_partial=None):

        if validation_metric is None:
            validation_metric = torchmetrics.AUROC(num_classes=2)
        if loss is None:
            loss = BCELoss()

        super().__init__(validation_metric,
                         seq_encoder,
                         loss,
                         optimizer_partial,
                         lr_scheduler_partial)

        if head is None:
            self._head = torch.nn.Sequential(
                AllStepsHead(
                    torch.nn.Sequential(
                        torch.nn.Linear(self.seq_encoder.embedding_size, 1),
                        torch.nn.Sigmoid(),
                        Squeeze(),
                    )
                ),
                FlattenHead(),
            )
        else:
            self._head = head

    @property
    def metric_name(self):
        return 'valid_auroc'

    @property
    def is_requires_reduced_sequence(self):
        return False

    def shared_step(self, x, y):
        y_h = self.seq_encoder(x)
        y_h = self._head(y_h)
        return y_h, y
