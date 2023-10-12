import torch
from torch import nn


class MergeProjectionHead(nn.Module):
    def __init__(self, h_dims=None):
        super().__init__()
        if h_dims is None:
            h_dims = list()
        else:
            assert type(h_dims) == list

        layers = list()
        for h in h_dims:
            layers.extend([nn.LazyLinear(h), nn.LazyInstanceNorm1d(), nn.LeakyReLU()])
        layers.extend([nn.LazyLinear(1), nn.Sigmoid(), nn.Flatten(0, -1)])
        self.model = nn.Sequential(*layers)

    def forward(self, long, short, n_repeats):
        long = torch.repeat_interleave(long, n_repeats, dim=0)
        preds = self.model(torch.cat([long, short], dim=-1))
        return preds
