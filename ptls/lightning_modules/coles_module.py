from ptls.lightning_modules.AbsModule import ABSModule
from ptls.metric_learn.metric import BatchRecallTopPL
from ptls.models import create_head_layers
from ptls.metric_learn.losses import get_loss
from ptls.metric_learn.sampling_strategies import get_sampling_strategy


class CoLESModule(ABSModule):
    def __init__(self, params):
        super().__init__(params)

        self._head = create_head_layers(params, self.seq_encoder)

    @property
    def metric_name(self):
        return 'recall_top_k'

    @property
    def is_requires_reduced_sequence(self):
        return True

    def get_loss(self):
        sampling_strategy = get_sampling_strategy(self.hparams.params)
        loss = get_loss(self.hparams.params, sampling_strategy)
        return loss

    def get_validation_metric(self):
        return BatchRecallTopPL(**self.hparams.params['validation_metric_params'])

    def shared_step(self, x, y):
        y_h = self(x)
        if self._head is not None:
            y_h = self._head(y_h)
        return y_h, y
