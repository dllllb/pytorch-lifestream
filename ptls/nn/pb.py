from functools import WRAPPER_ASSIGNMENTS
import torch

from ptls.nn.trx_encoder import PaddedBatch


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


class PBL2Norm(torch.nn.Module):
    def forward(self, x):
        return PaddedBatch(x.payload / (x.payload.pow(2).sum(dim=-1, keepdim=True) + 1e-9).pow(0.5),
                           x.seq_lens)
