import torch

from ptls.nn.seq_encoder.rnn_encoder import RnnEncoder
from ptls.nn.seq_encoder.utils import PerTransHead, scoring_head
from ptls.nn.trx_encoder import TrxEncoder
from ptls.nn.trx_encoder.trx_mean_encoder import TrxMeanEncoder


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
