from functools import WRAPPER_ASSIGNMENTS
import torch

from ptls.nn.normalization import L2NormEncoder
from ptls.data_load.padded_batch import PaddedBatch


def _pb_shell(cls):
    class PBShell(cls):
        def __init__(self, *args, **kwargs):
            for attr in WRAPPER_ASSIGNMENTS:
                setattr(self, attr, getattr(cls, attr))
            super().__init__(*args, **kwargs)

        def forward(self, x: PaddedBatch):
            return PaddedBatch(super().forward(x.payload), x.seq_lens)

    return PBShell


PBLinear = _pb_shell(torch.nn.Linear)
PBLayerNorm = _pb_shell(torch.nn.LayerNorm)
PBReLU = _pb_shell(torch.nn.ReLU)
PBL2Norm = _pb_shell(L2NormEncoder)
