import torch

from ptls.data_load.padded_batch import PaddedBatch


class RBatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.bn = torch.nn.BatchNorm1d(1)

    def forward(self, x):
        B, T = x.size()  # B x T
        x = x.view(B * T, 1)
        x = self.bn(x)
        x = x.view(B, T, 1)
        return x


class RBatchNormWithLens(torch.nn.Module):
    """
    The same as RBatchNorm, but ...
    Drop padded symbols (zeros) from batch when batch stat update
    """
    def __init__(self):
        super().__init__()

        self.bn = torch.nn.BatchNorm1d(1)

    def forward(self, v: PaddedBatch):
        x = v.payload
        seq_lens = v.seq_lens
        B, T = x.size()  # B x T

        mask = v.seq_len_mask.bool()
        x_new = x
        x_new[mask] = self.bn(x[mask].view(-1, 1)).view(-1)
        return x_new.view(B, T, 1)
