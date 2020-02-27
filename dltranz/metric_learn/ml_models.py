# coding: utf-8
import os
import sys
import torch
import torch.nn as nn

from torch.autograd import Function

from dltranz.transf_seq_encoder import TransformerSeqEncoder

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '../..'))

from dltranz.seq_encoder import RnnEncoder, LastStepEncoder, PerTransTransf, FirstStepEncoder
from dltranz.trx_encoder import TrxEncoder

# TODO: is the same as dltranz.seq_encoder.NormEncoder
class L2Normalization(nn.Module):
    def __init__(self):
        super(L2Normalization, self).__init__()

    def forward(self, input):
        return input.div(torch.norm(input, dim=1).view(-1, 1))


class Binarization(Function):
    @staticmethod
    def forward(self, x):
        q = (x>0).float()
        return  (2*q - 1)

    @staticmethod
    def backward(self, gradOutput):
        return gradOutput

binary = Binarization.apply


class BinarizationLayer(nn.Module):
    def __init__(self, hs_from, hs_to):
        super(BinarizationLayer, self).__init__()
        self.linear = nn.Linear(hs_from, hs_to, bias = False)

    def forward(self, x):
        return binary(self.linear(x))


def rnn_model(params):
    layers = [
        TrxEncoder(params['trx_encoder']),
        RnnEncoder(TrxEncoder.output_size(params['trx_encoder']), params['rnn']),
        LastStepEncoder(),
    ]
    if params['use_normalization_layer']:
        layers.append(L2Normalization())
    m = torch.nn.Sequential(*layers)
    return m


def transformer_model(params):
    p = TrxEncoder(params['trx_encoder'])
    trx_size = TrxEncoder.output_size(params['trx_encoder'])
    enc_input_size = params['transf']['input_size']
    if enc_input_size != trx_size:
        inp_reshape = PerTransTransf(trx_size, enc_input_size)
        p = torch.nn.Sequential(p, inp_reshape)

    e = TransformerSeqEncoder(enc_input_size, params['transf'])
    l = FirstStepEncoder()
    layers = [p, e, l]

    if params['use_normalization_layer']:
        layers.append(L2Normalization())
    m = torch.nn.Sequential(*layers)
    return m


def ml_model_by_type(model_type):
    model = {
        'rnn': rnn_model,
        'transf': transformer_model,
    }[model_type]
    return model


class ModelEmbeddingEnsemble(nn.Module):
    def __init__(self, submodels):
        super(ModelEmbeddingEnsemble, self).__init__()
        self.models = nn.ModuleList(submodels)

    def forward(self, *args):
        out = torch.cat([m(*args) for m in self.models], dim=1)
        return out
