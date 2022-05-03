import numpy as np
import torch
from torch import nn as nn

from ptls.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.seq_encoder.utils import LastStepEncoder
from ptls.trx_encoder import PaddedBatch, TrxEncoder


class RnnEncoder(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()

        self.hidden_size = config['hidden_size']
        self.rnn_type = config['type']
        self.bidirectional = config['bidir']
        if self.bidirectional:
            raise AttributeError('bidirectional RNN is not supported yet')
        self.trainable_starter = config['trainable_starter']

        # initialize RNN
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size,
                self.hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=self.bidirectional)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size,
                self.hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=self.bidirectional)
        else:
            raise Exception(f'wrong rnn type "{self.rnn_type}"')

        self.full_hidden_size = self.hidden_size if not self.bidirectional else self.hidden_size * 2

        # initialize starter position if needed
        if self.trainable_starter == 'static':
            num_dir = 2 if self.bidirectional else 1
            self.starter_h = nn.Parameter(torch.randn(num_dir, 1, self.hidden_size))

    def forward(self, x: PaddedBatch, h_0: torch.Tensor = None):
        """

        :param x:
        :param h_0: None or [1, B, H] float tensor
                    0.0 values in all components of hidden state of specific client means no-previous state and
                    use starter for this client
                    h_0 = None means no-previous state for all clients in batch
        :return:
        """
        shape = x.payload.size()
        assert shape[1] > 0, "Batch can'not have 0 transactions"

        # prepare initial state
        if self.trainable_starter == 'static':
            starter_h = self.starter_h.expand(-1, shape[0], -1).contiguous()
            if h_0 is None:
                h_0 = starter_h
            elif h_0 is not None and not self.training:
                h_0 = torch.where(
                    (h_0.squeeze(0).abs().sum(dim=1) == 0.0).unsqueeze(0).unsqueeze(2).expand(*starter_h.size()),
                    starter_h,
                    h_0,
                )
            else:
                raise NotImplementedError('Unsupported mode: cannot mix fixed X and learning Starter')

        # pass-through rnn
        if self.rnn_type == 'lstm':
            out, _ = self.rnn(x.payload)
        elif self.rnn_type == 'gru':
            out, _ = self.rnn(x.payload, h_0)
        else:
            raise Exception(f'wrong rnn type "{self.rnn_type}"')

        return PaddedBatch(out, x.seq_lens)


class RnnSeqEncoder(AbsSeqEncoder):
    def __init__(self, params, is_reduce_sequence):
        super().__init__(params, is_reduce_sequence)

        p = TrxEncoder(params['trx_encoder'])
        e = RnnEncoder(p.output_size, params['rnn'])
        layers = [p, e]
        self.reducer = LastStepEncoder()
        self.model = torch.nn.Sequential(*layers)

    @property
    def category_max_size(self):
        return self.model[0].category_max_size

    @property
    def category_names(self):
        return self.model[0].category_names

    @property
    def embedding_size(self):
        return self.params['rnn']['hidden_size']

    def forward(self, x):
        x = self.model(x)
        if self.is_reduce_sequence:
            x = self.reducer(x)
        return x


class RnnInference(torch.nn.Module):
    def __init__(self, model: RnnSeqEncoder):
        super().__init__()

        self.hidden_size = hidden_size
        self.is_reduce_sequence = model.seq_encoder.is_reduce_sequence
        self.p = model.seq_encoder.model[0]
        self.e = model.seq_encoder.model[1]
        self.reducer = model.seq_encoder.reducer

    def forward(self, x, h_0=None):
        x_ = self.p(x)
        x = self.e(x_, h_0)
        if self.is_reduce_sequence:
            x = self.reducer(x)
        return x


class RnnSeqEncoderDistributionTarget(RnnSeqEncoder):
    def transform(self, x):
        return np.sign(x) * np.log(np.abs(x) + 1)

    def transform_inv(self, x):
        return np.sign(x) * (np.exp(np.abs(x)) - 1)

    def __init__(self, params, is_reduce_sequence):
        super().__init__(params, is_reduce_sequence)
        head_params = dict(params['head_layers']).get('CombinedTargetHeadFromRnn', None)
        self.pass_samples = head_params.get('pass_samples', True)
        self.numeric_name = list(params['trx_encoder']['numeric_values'].keys())[0]
        self.collect_pos, self.collect_neg = (head_params.get('pos', True), head_params.get('neg', True)) if head_params else (0, 0)
        self.eps = 1e-7

    def forward(self, x):
        amount_col = []
        for i, row in enumerate(x.payload[self.numeric_name]):
            amount_col += [list(row[:x.seq_lens[i].item()].cpu().numpy())]
        amount_col = np.array(amount_col, dtype=object)
        neg_sums = []
        pos_sums = []
        for list_row in amount_col:
            np_row = np.array(list_row)
            if self.collect_neg:
                neg_sums += [np.sum(self.transform_inv(np_row[np.where(np_row < 0)]))]
            if self.collect_pos:
                pos_sums += [np.sum(self.transform_inv(np_row[np.where(np_row >= 0)]))]
        neg_sum_logs = np.log(np.abs(np.array(neg_sums)) + self.eps)
        pos_sum_logs = np.log(np.array(pos_sums) + self.eps)

        x = super().forward(x)
        if (not self.pass_samples):
            return x
        if self.collect_neg and self.collect_pos:
            return x, neg_sum_logs, pos_sum_logs
        elif self.collect_neg:
            return x, neg_sum_logs
        elif self.collect_pos:
            return x, pos_sum_logs


class SkipStepEncoder(nn.Module):
    def __init__(self, step_size):
        super().__init__()
        self.step_size = step_size

    def forward(self, x: PaddedBatch):
        max_len = x.payload.shape[1] - 1
        s = self.step_size

        first_dim_idx = []
        second_dim_idx = []
        for i, l in enumerate(x.seq_lens):
            idx_to_take = np.arange(min(l - 1, s - 1 + l % s), l, s)
            pad_idx = np.array([max_len - 1] * (max_len // s - len(idx_to_take)), dtype=np.int32)
            idx_to_take = np.concatenate([[-1], idx_to_take, pad_idx]) + 1
            first_dim_idx.append(np.ones(len(idx_to_take)) * i)
            second_dim_idx.append(idx_to_take)

        out = x.payload[first_dim_idx, second_dim_idx]
        out_lens = torch.tensor([min(1, l // self.step_size) for l in x.seq_lens])

        return PaddedBatch(out, out_lens)


def skip_rnn_encoder(input_size, params):
    rnn0 = RnnEncoder(input_size, params['rnn0'])
    rnn1 = RnnEncoder(params['rnn0']['hidden_size'], params['rnn1'])
    sse = SkipStepEncoder(params['skip_step_size'])
    return nn.Sequential(rnn0, sse, rnn1)
