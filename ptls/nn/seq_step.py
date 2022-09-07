import torch
import numpy as np
from torch import nn as nn

from ptls.data_load.padded_batch import PaddedBatch


class TimeStepShuffle(nn.Module):
    def forward(self, x: PaddedBatch):
        shuffled = []
        for seq, slen in zip(x.payload, x.seq_lens):
            idx = torch.randperm(slen) + 1
            pad_idx = torch.arange(slen + 1, len(seq))
            idx = torch.cat([torch.zeros(1, dtype=torch.long), idx, pad_idx])
            shuffled.append(seq[idx])

        shuffled = PaddedBatch(torch.stack(shuffled), x.seq_lens)
        return shuffled


class LastStepEncoder(nn.Module):
    """
    Class is used by ptls.nn.RnnSeqEncoder for reducing RNN output with shape (B, L, H), where
        B - batch size
        L - sequence length
        H - hidden RNN size
    to embeddings tensor with shape (B, H). The last hidden state is used for embedding.
    
    Example of usage: seq_encoder = RnnSeqEncoder(..., reducer='last_step')
    """
    def forward(self, x: PaddedBatch):
        h = x.payload[range(len(x.payload)), [l - 1 for l in x.seq_lens]]
        return h


class FirstStepEncoder(nn.Module):
    """
    Class is used by ptls.nn.RnnSeqEncoder class for reducing RNN output with shape (B, L, H)
    to embeddings tensor with shape (B, H). The first hidden state is used for embedding.
    
    where:
        B - batch size
        L - sequence length
        H - hidden RNN size
    
    Example of usage: seq_encoder = RnnSeqEncoder(..., reducer='first_step')
    """
    def forward(self, x: PaddedBatch):
        h = x.payload[:, 0, :]  # [B, L, H] -> [B, H]
        return h
    

class LastMaxAvgEncoder(nn.Module):
    """
    Class is used by ptls.nn.RnnSeqEncoder class for reducing RNN output with shape (B, L, H)
    to embeddings tensor with shape (B, 3 * H). Embeddings are created by concatenating:
        - last hidden state from RNN output,
        - max pool over all hidden states of RNN output,
        - average pool over all hidden states of RNN output.
        
    where:
        B - batch size
        L - sequence length
        H - hidden RNN size
        
    Example of usage: seq_encoder = RnnSeqEncoder(..., reducer='last_max_avg')
    """
    def forward(self, x: PaddedBatch):
        rnn_max_pool = x.payload.max(dim=1)[0]
        rnn_avg_pool = x.payload.sum(dim=1) / x.seq_lens.unsqueeze(-1)
        h = x.payload[range(len(x.payload)), [l - 1 for l in x.seq_lens]]
        h = torch.cat((h, rnn_max_pool, rnn_avg_pool), dim=-1)
        return h

    
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
