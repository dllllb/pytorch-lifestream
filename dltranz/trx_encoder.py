from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn


class PaddedBatch:
    def __init__(self, payload: Dict[str, torch.Tensor], length: torch.LongTensor):
        self._payload = payload
        self._length = length

    @property
    def payload(self):
        return self._payload

    @property
    def seq_lens(self):
        return self._length


class NoisyEmbedding(nn.Embedding):
    """
    Embeddings with additive gaussian noise with mean=0 and user-defined variance.
    *args and **kwargs defined by usual Embeddings
    Args:
        noise_scale (float): when > 0 applies additive noise to embeddings.
            When = 0, forward is equivalent to usual embeddings.
        dropout (float): probability of embedding axis to be dropped. 0 means no dropout at all.

    For other parameters defenition look at nn.Embedding help
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None,
                 noise_scale=0, dropout=0):
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                         norm_type, scale_grad_by_freq, sparse, _weight)
        self.noise = torch.distributions.Normal(0, noise_scale)
        self.scale = noise_scale
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(super().forward(x))
        if self.training and self.scale > 0:
            x += self.noise.sample((self.weight.shape[1], )).to(self.weight.device)
        return x


class IdentityScaler(nn.Module):
    def forward(self, x):
        return x


class LogScaler(nn.Module):
    def forward(self, x):
        return x.abs().log1p() * x.sign()


class YearScaler(nn.Module):
    def forward(self, x):
        return x/365


def scaler_by_name(name):
    scaler = {
        'identity': IdentityScaler,
        'sigmoid': torch.nn.Sigmoid,
        'log': LogScaler,
        'year': YearScaler,
    }.get(name, None)

    if scaler is None:
        raise Exception(f'unknown scaler name: {name}')
    else:
        return scaler()

    
class TrxEncoder(nn.Module):
    def __init__(self, config):

        super().__init__()
        self.scalers = OrderedDict(
            {name: scaler_by_name(scaler_name) for name, scaler_name in config['numeric_values'].items()}
        )

        self.embeddings = nn.ModuleDict()
        for emb_name, emb_props in config['embeddings'].items():
            if emb_props.get('disabled', False):
                continue
            self.embeddings[emb_name] = NoisyEmbedding(
                num_embeddings=emb_props['in'],
                embedding_dim=emb_props['out'],
                padding_idx=0,
                max_norm=1 if config['norm_embeddings'] else None,
                noise_scale=config['embeddings_noise'])

    def forward(self, x: PaddedBatch):
        processed = []
        for field_name, embed_layer in self.embeddings.items():
            processed.append(embed_layer(x.payload[field_name].long()))

        for value_name, scaler in self.scalers.items():
            res = scaler(x.payload[value_name].unsqueeze(-1).float())
            processed.append(res)

        out = torch.cat(processed, -1)
        return PaddedBatch(out, x.seq_lens)

    @staticmethod
    def output_size(config):
        nv = config.get('numeric_values', dict())
        sz = len(nv.keys())
        sz += sum(econf['out'] for econf in config.get('embeddings', dict()).values() if not econf.get('disabled', False))
        return sz


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
