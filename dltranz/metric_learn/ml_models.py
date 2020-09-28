# coding: utf-8
import logging
import os
import sys
import torch
import torch.nn as nn

from torch.autograd import Function

from dltranz.agg_feature_model import AggFeatureModel
from dltranz.cpc import CPC_Ecoder
from dltranz.transf_seq_encoder import TransformerSeqEncoder

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '../..'))

from dltranz.seq_encoder import RnnEncoder, LastStepEncoder, PerTransTransf, FirstStepEncoder, PaddedBatch, \
    DropoutEncoder
from dltranz.trx_encoder import TrxEncoder

logger = logging.getLogger(__name__)


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


def projection_head(input_size, output_size):
    layers = [
        torch.nn.Linear(input_size, input_size),
        torch.nn.ReLU(),
        torch.nn.Linear(input_size, output_size),
    ]
    m = torch.nn.Sequential(*layers)
    return m


def rnn_model(params):
    encoder_layers = [
        TrxEncoder(params['trx_encoder']),
        RnnEncoder(TrxEncoder.output_size(params['trx_encoder']), params['rnn']),
        LastStepEncoder(),
    ]

    layers = [torch.nn.Sequential(*encoder_layers)]
    if 'projection_head' in params:
        logger.info('projection_head included')
        layers.extend(projection_head(params['rnn.hidden_size'], params['projection_head.output_size']))

    if params.get('embeddings_dropout', 0):
        layers.append(DropoutEncoder(params['embeddings_dropout']))
        logger.info('DropoutEncoder included')

    if params['use_normalization_layer']:
        layers.append(L2Normalization())
        logger.info('L2Normalization included')
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

    encoder = torch.nn.Sequential(*layers)
    layers = [encoder]
    if params['use_normalization_layer']:
        layers.append(L2Normalization())
    m = torch.nn.Sequential(*layers)
    return m


def agg_feature_model(params):
    layers = [
        torch.nn.Sequential(
            AggFeatureModel(params['trx_encoder']),
        ),
        torch.nn.BatchNorm1d(AggFeatureModel.output_size(params['trx_encoder'])),
    ]
    if params['use_normalization_layer']:
        layers.append(L2Normalization())
    m = torch.nn.Sequential(*layers)
    return m


def cpc_model(params):
    trx_e = TrxEncoder(params['trx_encoder'])
    trx_e_out_size = TrxEncoder.output_size(params['trx_encoder'])
    rnn_e = RnnEncoder(trx_e_out_size, params['rnn'])
    cpc_e = CPC_Ecoder(trx_e, rnn_e, trx_e_out_size, params['cpc'])
    return cpc_e


def ml_model_by_type(model_type):
    model = {
        'rnn': rnn_model,
        'transf': transformer_model,
        'agg_features': agg_feature_model,
        'cpc_model': cpc_model,
    }[model_type]
    return model


class MeLESModel(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.p = TrxEncoder(params['trx_encoder'])
        self.e = RnnEncoder(TrxEncoder.output_size(params['trx_encoder']), params['rnn'])
        self.l = torch.nn.Sequential(LastStepEncoder(), L2Normalization())

    def forward(self, padded_x, h_0=None):
        outputs = self.l(self.e(self.p(padded_x), h_0))
        return outputs


class ModelEmbeddingEnsemble(nn.Module):
    def __init__(self, submodels):
        super(ModelEmbeddingEnsemble, self).__init__()
        self.models = nn.ModuleList(submodels)

    def forward(self, x: PaddedBatch, h_0: torch.Tensor = None):
        """
        x - PaddedBatch of transactions sequences
        h_0 - previous state of embeddings (initial size for GRU). torch Tensor of shape (batch_size, embedding_size)
        """
        if h_0 is not None:
            h_0_splitted = torch.chunk(h_0, len(self.models), -1)
            out = torch.cat([m(x, h.contiguous()) for m, h in zip(self.models, h_0_splitted)], dim=-1)
        else:
            out = torch.cat([m(x) for i, m in enumerate(self.models)], dim=-1)
        return out


class ComplexModel(torch.nn.Module):
    def __init__(self, ml_model, params):
        super().__init__()
        self.ml_model = ml_model
        self.projection_ml_head = projection_head(params['rnn.hidden_size'], params['ml_projection_head.output_size'])
        self.projection_aug_head = torch.nn.Sequential(
            projection_head(params['rnn.hidden_size'], params['aug_projection_head.output_size']),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        encoder_output = self.ml_model(x)
        ml_head_output = self.projection_ml_head(encoder_output)
        aug_head_output = self.projection_aug_head(encoder_output)
        return aug_head_output, ml_head_output


def load_encoder_for_inference(conf):
    ext = os.path.splitext(conf['model_path.model'])[1]
    if ext in ('.pth', '.pt'):
        params = conf.get('params', conf)
        model_type = params['model_type']
        model_f = ml_model_by_type(model_type)
        model = model_f(params)
        model_d = torch.load(conf['model_path.model'], map_location=torch.device("cpu"))
        model.load_state_dict(model_d)

        if isinstance(model, CPC_Ecoder):
            trx_e, rnn_e = model.trx_encoder, model.seq_encoder
            l = LastStepEncoder()
            model = torch.nn.Sequential(trx_e, rnn_e, l)

    elif ext == '.p':
        model = torch.load(conf['model_path.model'], map_location=torch.device("cpu"))
    else:
        raise NotImplementedError(f'Unknown model file extension: "{ext}"')
    return model
