import numpy as np
import torch
from torch import nn as nn

from ptls.seq_encoder.rnn_encoder import RnnEncoder
from ptls.trx_encoder import PaddedBatch


class SkipStepEncoder(nn.Module):
    def __init__(self, step_size):
        super().__init__()
        self.step_size = step_size

    def forward(self, x: PaddedBatch):
        max_len = x.payload.shape[1] - 1
        s = self.step_size

        first_dim_idx = []
        second_dim_idx = []
        for i, l in enumerate(x.seq_lens):
            idx_to_take = np.arange(min(l - 1, s - 1 + l % s), l, s)
            pad_idx = np.array([max_len - 1] * (max_len // s - len(idx_to_take)), dtype=np.int32)
            idx_to_take = np.concatenate([[-1], idx_to_take, pad_idx]) + 1
            first_dim_idx.append(np.ones(len(idx_to_take)) * i)
            second_dim_idx.append(idx_to_take)

        out = x.payload[first_dim_idx, second_dim_idx]
        out_lens = torch.tensor([min(1, l // self.step_size) for l in x.seq_lens])

        return PaddedBatch(out, out_lens)


def skip_rnn_encoder(input_size, params):
    rnn0 = RnnEncoder(input_size, params.rnn0)
    rnn1 = RnnEncoder(params.rnn0.hidden_size, params.rnn1)
    sse = SkipStepEncoder(params.skip_step_size)
    return nn.Sequential(rnn0, sse, rnn1)