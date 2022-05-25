from hydra.utils import instantiate
from ptls.lightning_modules.AbsModule import ABSModule


class CoLESModule(ABSModule):
    def __init__(self, validation_metric=None,
                       seq_encoder=None,
                       head=None,
                       loss=None,
                       optimizer=None,
                       lr_scheduler_wrapper=None,
                       lr_scheduler=None):

        super().__init__(validation_metric,
                         seq_encoder,
                         loss,
                         optimizer,
                         lr_scheduler_wrapper,
                         lr_scheduler)

        self._head = head(input_size=self._seq_encoder.embedding_size)

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
