from typing import List

import torch
from torch.nn import Linear, BatchNorm1d, Sigmoid, Sequential, ReLU, LogSoftmax

from ptls.custom_layers import Squeeze
from ptls.seq_encoder.rnn_encoder import RnnEncoder
from ptls.seq_encoder.skip_rnn_encoder import skip_rnn_encoder
from ptls.seq_encoder.utils import NormEncoder
from ptls.seq_encoder.utils import PerTransHead, TimeStepShuffle, scoring_head
from ptls.trx_encoder import TrxEncoder
from ptls.trx_encoder.trx_mean_encoder import TrxMeanEncoder


def trx_avg_model(params):
    p = TrxEncoder(params.trx_encoder)
    h = PerTransHead(p.output_size)
    m = torch.nn.Sequential(p, h, torch.nn.Sigmoid())
    return m


def trx_avg2_model(params):
    p = TrxMeanEncoder(params.trx_encoder)
    h = scoring_head(TrxMeanEncoder.output_size(params.trx_encoder), params.head)
    m = torch.nn.Sequential(p, h)
    return m


def rnn_model(params):
    p = TrxEncoder(**params.trx_encoder)
    e = RnnEncoder(p.output_size, **params.rnn)
    h = scoring_head(
        input_size=params.rnn.hidden_size * (2 if params.rnn.bidir else 1),
        params=params.head
    )

    m = torch.nn.Sequential(p, e, h)
    return m


def rnn_shuffle_model(params):
    p = TrxEncoder(params.trx_encoder)
    p_size = p.output_size
    p = torch.nn.Sequential(p, TimeStepShuffle())
    e = RnnEncoder(p_size, params.rnn)
    h = scoring_head(
        input_size=params.rnn.hidden_size * (2 if params.rnn.bidir else 1),
        params=params.head
    )

    m = torch.nn.Sequential(p, e, h)
    return m


def skip_rnn2_model(params):
    p = TrxEncoder(params.trx_encoder)
    e = skip_rnn_encoder(p.output_size, params.skip_rnn)
    h = scoring_head(
        input_size=params.skip_rnn.rnn1.hidden_size * (2 if params.skip_rnn.rnn1.bidir else 1),
        params=params.head
    )

    m = torch.nn.Sequential(p, e, h)
    return m


class Head(torch.nn.Module):
    r"""Head for the sequence encoder

    Parameters
    ----------
         input_size: int
            input size
         use_norm_encoder: bool. Default: False
            whether to use normalization layers before the head
         use_batch_norm: bool. Default: False.
            whether to use BatchNorm.
         hidden_layers_sizes: List[int]. Default: None.
            sizes of linear layers. If None without additional linear layers. Default = None,
         objective: str. Default: None.
            Options: None, 'classification', 'regression'. Default = None.
         num_classes: int. Default: 1.
            The number of classed in classification problem. Default correspond to binary classification.

     """
    def __init__(self,
                 input_size: int = None,
                 use_norm_encoder: bool = False,
                 use_batch_norm: bool = False,
                 hidden_layers_sizes: List[int] = None,
                 objective: str = None,
                 num_classes: int = 1):
        super().__init__()
        # TODO: check possibility to create empty head with do nothing

        layers = []

        if use_norm_encoder:
            layers.append(NormEncoder())

        if use_batch_norm:
            layers.append(BatchNorm1d(input_size))

        if hidden_layers_sizes is not None:
            layers_size = [input_size] + list(hidden_layers_sizes)
            for size_in, size_out in zip(layers_size[:-1], layers_size[1:]):
                layers.append(Linear(size_in, size_out))
                layers.append(ReLU())
                if use_batch_norm:
                    layers.append(BatchNorm1d(size_out))

        if objective == 'classification':
            if num_classes == 1:
                h = Sequential(Linear(input_size, num_classes), Sigmoid(), Squeeze())
            else:
                h = Sequential(Linear(input_size, num_classes), LogSoftmax(dim=1))
            layers.append(h)

        elif objective == 'regression':
            h = Sequential(Linear(input_size, 1), Squeeze())
            layers.append(h)

        elif objective is not None:
            raise AttributeError(f"unknown objective {objective}. Supported: classification and regression")

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
