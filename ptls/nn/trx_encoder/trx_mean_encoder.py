from collections import OrderedDict

import torch
from torch import nn as nn

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.trx_encoder.scalers import scaler_by_name


class TrxMeanEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scalers = OrderedDict(
            {name: scaler_by_name(scaler_name) for name, scaler_name in config['numeric_values'].items()}
        )

        self.embeddings = nn.ModuleDict()
        for name, dim in config['embeddings'].items():
            dict_len = dim['in']
            self.embeddings[name] = nn.EmbeddingBag(dict_len, dict_len, mode='mean')
            self.embeddings[name].weight = nn.Parameter(torch.diag(torch.ones(dict_len)).float())

    def forward(self, x: PaddedBatch):
        processed = []

        for field_name in self.embeddings.keys():
            processed.append(self.embeddings[field_name](x.payload[field_name]).detach())

        for value_name, scaler in self.scalers.items():
            var = scaler(x.payload[value_name].unsqueeze(-1).float())
            means = torch.tensor([e[:l].mean() for e, l in zip(var, x.seq_lens)]).unsqueeze(-1)
            processed.append(means)

        out = torch.cat(processed, -1)
        return out

    @staticmethod
    def output_size(config):
        sz = len(config['numeric_values'])
        sz += sum(dim['in'] for _, dim in config['embeddings'].items())
        return sz
