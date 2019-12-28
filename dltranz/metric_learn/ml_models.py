# coding: utf-8
import os
import sys
import torch
import torch.nn as nn

from torch.autograd import Function

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '../..'))

from dltranz.seq_encoder import RnnEncoder, LastStepEncoder, NormEncoder
from dltranz.trx_encoder import TrxEncoder

# TODO: копирует dltranz.seq_encoder.NormEncoder
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
    p = TrxEncoder(params['trx_encoder'])
    e = RnnEncoder(TrxEncoder.output_size(params['trx_encoder']), params['rnn'])
    l = LastStepEncoder()
    n = L2Normalization()
    m = torch.nn.Sequential(p, e, l, n)
    return m

class ModelEmbeddingEnsemble(nn.Module):
    def __init__(self, submodels):
        super(ModelEmbeddingEnsemble, self).__init__()
        self.models = nn.ModuleList(submodels)

    def forward(self, *args):
        out = torch.cat([m(*args) for m in self.models], dim=1)
        return out
