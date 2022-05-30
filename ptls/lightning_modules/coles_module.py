from hydra.utils import instantiate
from ptls.lightning_modules.AbsModule import ABSModule


class CoLESModule(ABSModule):
    def __init__(self, validation_metric=None,
                       seq_encoder=None,
                       head=None,
                       loss=None,
                       optimizer_partial=None,
                       lr_scheduler_partial=None):

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
