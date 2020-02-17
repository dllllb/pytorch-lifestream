import torch

from dltranz.seq_encoder import PerTransHead, scoring_head, TimeStepShuffle, RnnEncoder, skip_rnn_encoder, \
    PerTransTransf
from dltranz.transf_seq_encoder import TransformerSeqEncoder
from dltranz.trellisnet import TrellisNetEncoder
from dltranz.trx_encoder import TrxEncoder, TrxMeanEncoder


def trx_avg_model(params):
    p = TrxEncoder(params['trx_encoder'])
    h = PerTransHead(TrxEncoder.output_size(params['trx_encoder']))
    m = torch.nn.Sequential(p, h, torch.nn.Sigmoid())
    return m


def trx_avg2_model(params):
    p = TrxMeanEncoder(params['trx_encoder'])
    h = scoring_head(TrxMeanEncoder.output_size(params['trx_encoder']), params['head'])
    m = torch.nn.Sequential(p, h)
    return m


def rnn_model(params):
    p = TrxEncoder(params['trx_encoder'])
    e = RnnEncoder(TrxEncoder.output_size(params['trx_encoder']), params['rnn'])
    h = scoring_head(params['rnn.hidden_size'], params['head'])

    m = torch.nn.Sequential(p, e, h)
    return m


def rnn_shuffle_model(params):
    p = TrxEncoder(params['trx_encoder'])
    p = torch.nn.Sequential(p, TimeStepShuffle())
    e = RnnEncoder(TrxEncoder.output_size(params['trx_encoder']), params['rnn'])
    h = scoring_head(params['rnn.hidden_size'], params['head'])

    m = torch.nn.Sequential(p, e, h)
    return m


def skip_rnn2_model(params):
    p = TrxEncoder(params['trx_encoder'])
    e = skip_rnn_encoder(TrxEncoder.output_size(params['trx_encoder']), params['skip_rnn'])
    h = scoring_head(params['skip_rnn.rnn1.hidden_size'], params['head'])

    m = torch.nn.Sequential(p, e, h)
    return m


def transformer_model(params):
    p = TrxEncoder(params['trx_encoder'])
    trx_size = TrxEncoder.output_size(params['trx_encoder'])
    enc_input_size = params['transf']['input_size']
    if enc_input_size != trx_size:
        inp_reshape = PerTransTransf(trx_size, enc_input_size)
        p = torch.nn.Sequential(p, inp_reshape)

    e = TransformerSeqEncoder(enc_input_size, params['transf'])
    h = scoring_head(enc_input_size, params['head'])

    m = torch.nn.Sequential(p, e, h)
    return m


def trellisnet_model(params):
    p = TrxEncoder(params['trx_encoder'])
    trx_size = TrxEncoder.output_size(params['trx_encoder'])
    enc_input_size = params['trellisnet']['ninp']
    if enc_input_size != trx_size:
        inp_reshape = PerTransTransf(trx_size, enc_input_size)
        p = torch.nn.Sequential(p, inp_reshape)

    e = TrellisNetEncoder(enc_input_size, params['trellisnet'])

    h = scoring_head(params['trellisnet']['nout'], params['head'])

    m = torch.nn.Sequential(p, e, h)
    return m



def model_by_type(model_type):
    model = {
        'avg': trx_avg_model,
        'avg2': trx_avg2_model,
        'rnn': rnn_model,
        'rnn-shuffle': rnn_shuffle_model,
        'skip-rnn2': skip_rnn2_model,
        'transf': transformer_model,
        'trellisnet': trellisnet_model,
    }[model_type]
    return model


def freeze_layers(model):
    for p in model.parameters():
        p.requires_grad = False
