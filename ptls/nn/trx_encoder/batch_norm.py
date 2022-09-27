import torch

from ptls.data_load.padded_batch import PaddedBatch


class RBatchNorm(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.bn = torch.nn.BatchNorm1d(num_features)

    def forward(self, v: PaddedBatch):
        x = v.payload
        B, T, H = x.size()  # B x T X H
        x = x.view(B * T, H)
        x = self.bn(x)
        x = x.view(B, T, H)
        return PaddedBatch(x, v.seq_lens)


class RBatchNormWithLens(torch.nn.Module):
    """
    The same as RBatchNorm, but ...
    Drop padded symbols (zeros) from batch when batch stat update
    """
    def __init__(self, num_features):
        super().__init__()

        self.bn = torch.nn.BatchNorm1d(num_features)

    def forward(self, v: PaddedBatch):
        x = v.payload
        B, T, H = x.size()  # B x T X H

        mask = v.seq_len_mask.bool()
        x_new = x.clone()
        x_new[mask] = self.bn(x[mask])
        return PaddedBatch(x_new, v.seq_lens)
