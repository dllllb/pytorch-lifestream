import torch

from dltranz.seq_encoder.rnn_encoder import RnnEncoder
from dltranz.seq_encoder.utils import PerTransHead, PerTransTransf, TimeStepShuffle, scoring_head
from dltranz.seq_encoder.rnn_encoder import skip_rnn_encoder
from dltranz.seq_encoder.transf_seq_encoder import TransformerSeqEncoder
from dltranz.trellisnet import TrellisNetEncoder
from dltranz.trx_encoder import TrxEncoder, TrxMeanEncoder


def trx_avg_model(params):
    p = TrxEncoder(params['trx_encoder'])
    h = PerTransHead(p.output_size)
    m = torch.nn.Sequential(p, h, torch.nn.Sigmoid())
    return m


def trx_avg2_model(params):
    p = TrxMeanEncoder(params['trx_encoder'])
    h = scoring_head(TrxMeanEncoder.output_size(params['trx_encoder']), params['head'])
    m = torch.nn.Sequential(p, h)
    return m


def rnn_model(params):
    p = TrxEncoder(params['trx_encoder'])
    e = RnnEncoder(p.output_size, params['rnn'])
    h = scoring_head(
        input_size=params['rnn.hidden_size'] * (2 if params['rnn.bidir'] else 1),
        params=params['head']
    )

    m = torch.nn.Sequential(p, e, h)
    return m


def rnn_shuffle_model(params):
    p = TrxEncoder(params['trx_encoder'])
    p_size = p.output_size
    p = torch.nn.Sequential(p, TimeStepShuffle())
    e = RnnEncoder(p_size, params['rnn'])
    h = scoring_head(
        input_size=params['rnn.hidden_size'] * (2 if params['rnn.bidir'] else 1),
        params=params['head']
    )

    m = torch.nn.Sequential(p, e, h)
    return m


def skip_rnn2_model(params):
    p = TrxEncoder(params['trx_encoder'])
    e = skip_rnn_encoder(p.output_size, params['skip_rnn'])
    h = scoring_head(
        input_size=params['skip_rnn.rnn1.hidden_size'] * (2 if params['skip_rnn.rnn1.bidir'] else 1),
        params=params['head']
    )

    m = torch.nn.Sequential(p, e, h)
    return m


def transformer_model(params):
    p = TrxEncoder(params['trx_encoder'])
    trx_size = p.output_size
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
    trx_size = p.output_size
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


class DistributionTargetsHead(torch.nn.Module):
    def __init__(self, seq_encoder, in_size=48, num_distr_classes=6):
        super().__init__()
        self.dense1 = torch.nn.Linear(seq_encoder.embedding_size, 512)

        self.dense2_distributions = torch.nn.Linear(512, 128)
        self.dense2_sums = torch.nn.Linear(512, 64)

        self.dense3_distr_neg = torch.nn.Linear(128, num_distr_classes)
        self.dense3_distr_pos = torch.nn.Linear(128, num_distr_classes)

        self.dense3_sums_neg = torch.nn.Linear(64, 1)
        self.dense3_sums_pos = torch.nn.Linear(64, 1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.dense1(x))

        out2_distr = self.relu(self.dense2_distributions(out1))
        out2_sums = self.relu(self.dense2_sums(out1))

        out3_distr_neg = self.dense3_distr_neg(out2_distr)
        out3_distr_pos = self.dense3_distr_pos(out2_distr)

        out3_sums_neg = self.dense3_sums_neg(out2_sums)
        out3_sums_pos = self.dense3_sums_pos(out2_sums)

        return out3_sums_neg, out3_distr_neg, out3_sums_pos, out3_distr_pos


def create_head_layers(params, seq_encoder):
    if not params.get('distribution_targets_task'):
        from torch.nn import Linear, BatchNorm1d, ReLU, Sigmoid, LogSoftmax
        from dltranz.custom_layers import Squeeze
        from dltranz.seq_encoder.utils import NormEncoder

        layers = []
        _locals = locals()
        for l_name, l_params in params['head_layers']:
            l_params = {k: int(v.format(**_locals)) if type(v) is str else v
                        for k, v in l_params.items()}

            cls = _locals.get(l_name, None)
            layers.append(cls(**l_params))
        return torch.nn.Sequential(*layers)
    else:
        return DistributionTargetsHead(seq_encoder)
