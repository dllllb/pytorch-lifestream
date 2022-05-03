import torch
from torch.nn import BCELoss

from ptls.custom_layers import Squeeze
from ptls.lightning_modules.AbsModule import ABSModule
from ptls.seq_to_target import EpochAuroc
from ptls.seq_encoder.utils import AllStepsHead, FlattenHead


class RtdModule(ABSModule):
    def __init__(self, params):
        super().__init__(params)

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

    @property
    def metric_name(self):
        return 'valid_auroc'

    @property
    def is_requires_reduced_sequence(self):
        return False

    def get_loss(self):
        return BCELoss()

    def get_validation_metric(self):
        return EpochAuroc()

    def shared_step(self, x, y):
        y_h = self.seq_encoder(x)
        y_h = self._head(y_h)
        return y_h, y
