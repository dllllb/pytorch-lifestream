import math
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

    @property
    def device(self):
        return self._length.device

    def __len__(self):
        return len(self._length)

    def to(self, device, non_blocking=False):
        length = self._length.to(device=device, non_blocking=non_blocking)
        payload = {
            k: v.to(device=device, non_blocking=non_blocking) for k, v in self._payload.items()
        }
        return PaddedBatch(payload, length)

    @property
    def seq_len_mask(self):
        """mask with B*T size for valid tokens in `payload`
        """
        if type(self._payload) is dict:
            B, T = next(iter(self._payload.values())).size()
        else:
            B, T = self._payload.size()[:2]
        return (torch.arange(T, device=self._length.device).unsqueeze(0).expand(B, T) < \
                self._length.unsqueeze(1)).long()


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
        if max_norm is not None:
            raise AttributeError("Please don't use embedding normalisation. "
                                 "https://github.com/pytorch/pytorch/issues/44792")

        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                         norm_type, scale_grad_by_freq, sparse, _weight)
        self.noise = torch.distributions.Normal(0, noise_scale) if noise_scale > 0 else None
        self.scale = noise_scale
        self.dropout = nn.Dropout(dropout)

        # This is workaround for https://github.com/pytorch/pytorch/issues/44792
        # Weight are normalized after forward pass
        _ = super().forward(torch.arange(num_embeddings))

    def forward(self, x):
        x = self.dropout(super().forward(x))
        if self.training and self.scale > 0:
            x += self.noise.sample((self.weight.shape[1], )).to(self.weight.device)
        return x


class RBatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.bn = torch.nn.BatchNorm1d(1)

    def forward(self, x):
        B, T, _ = x.size()  # B x T x 1
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

        mask = v.seq_len_mask
        x_new = x
        x_new[mask] = self.bn(x[mask].view(-1, 1)).view(-1)
        return x_new.view(B, T, 1)


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


class FloatPositionalEncoding(nn.Module):
    def __init__(self, out_size):
        super(FloatPositionalEncoding, self).__init__()

        self.out_size = out_size

    def forward(self, position):
        """

        :param position: B x T
        :return: B x T x H
        """
        div_term = torch.exp(torch.arange(0, self.out_size, 2).float() * (-math.log(10000.0) / self.out_size))
        div_term = div_term.unsqueeze(0).unsqueeze(0)
        div_term = div_term.to(device=position.device)

        position = position.unsqueeze(2)

        pe = torch.cat([torch.sin(position * div_term), torch.cos(position * div_term)], dim=2)
        self.register_buffer('pe', pe)

        return pe


class TrxEncoder(nn.Module):
    def __init__(self, config):

        super().__init__()
        self.scalers = nn.ModuleDict()
        self.use_batch_norm_with_lens = config.get('use_batch_norm_with_lens', False)
        self.clip_replace_value = config.get('clip_replace_value', 'max')

        for name, scaler_name in config['numeric_values'].items():
            self.scalers[name] = torch.nn.Sequential(
                RBatchNormWithLens() if self.use_batch_norm_with_lens else RBatchNorm(),
                scaler_by_name(scaler_name),
            )

        self.embeddings = nn.ModuleDict()
        for emb_name, emb_props in config['embeddings'].items():
            if emb_props.get('disabled', False):
                continue
            if emb_props['out'] == 0:
                continue
            self.embeddings[emb_name] = NoisyEmbedding(
                num_embeddings=emb_props['in'],
                embedding_dim=emb_props['out'],
                padding_idx=0,
                max_norm=1 if config.get('norm_embeddings', False) else None,
                noise_scale=config['embeddings_noise'],
            )

        self.pos = nn.ModuleDict()
        for pos_name, pos_params in config.get('positions', {}).items():
            self.pos[pos_name] = FloatPositionalEncoding(**pos_params)

    def forward(self, x: PaddedBatch):
        processed = []

        for field_name, embed_layer in self.embeddings.items():
            feature = self._smart_clip(x.payload[field_name].long(), embed_layer.num_embeddings)
            processed.append(embed_layer(feature))

        for value_name, scaler in self.scalers.items():
            if self.use_batch_norm_with_lens:
                res = scaler(PaddedBatch(x.payload[value_name].float(), x.seq_lens))
            else:
                res = scaler(x.payload[value_name].unsqueeze(-1).float())
            processed.append(res)

        for pos_name, pe in self.pos.items():
            processed.append(pe(x.payload[pos_name].float()))

        out = torch.cat(processed, -1)
        return PaddedBatch(out, x.seq_lens)

    def _smart_clip(self, values, max_size):
        if self.clip_replace_value == 'max':
            return values.clip(0, max_size - 1)
        else:
            res = values.clone()
            res[values >= max_size] = self.clip_replace_value
            return res

    @property
    def output_size(self):
        sz = len(self.scalers)
        sz += sum(econf.embedding_dim for econf in self.embeddings.values())
        sz += sum(pos_params.out_size for pos_params in self.pos.values())
        return sz

    @property
    def category_names(self):
        return set([field_name for field_name in self.embeddings.keys()] +
                   [value_name for value_name in self.scalers.keys()] +
                   [pos_name for pos_name in self.pos.keys()]
                   )

    @property
    def category_max_size(self):
        return {k: v.num_embeddings for k, v in self.embeddings.items()}


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
