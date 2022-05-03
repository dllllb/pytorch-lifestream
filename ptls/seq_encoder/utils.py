import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn as nn
from torch.nn import functional as tf

from ptls.custom_layers import Squeeze
from ptls.trx_encoder import PaddedBatch


class PerTransHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.cnn = torch.nn.Conv1d(input_size, 1, 1)

    def forward(self, x):
        seq_len = x.payload.shape[1]
        feature_vec = torch.transpose(x.payload, 1, 2)

        tx = self.cnn(feature_vec)
        x = tf.avg_pool1d(tx, seq_len)
        x = torch.transpose(x, 1, 2)

        return x.squeeze()


class PerTransTransf(nn.Module):
    def __init__(self, input_size, out_size):
        super().__init__()
        self.cnn = torch.nn.Conv1d(input_size, out_size, 1)

    def forward(self, x):
        feature_vec = torch.transpose(x.payload, 1, 2)

        tx = self.cnn(feature_vec)
        out = torch.transpose(tx, 1, 2)

        return PaddedBatch(out, x.seq_lens)


class ConcatLenEncoder(nn.Module):
    def forward(self, x: PaddedBatch):
        lens = x.seq_lens.unsqueeze(-1).float().to(x.payload.device)
        lens_normed = lens / 200

        h = x.payload[range(len(x.payload)), [l - 1 for l in x.seq_lens]]

        embeddings = torch.cat([h, lens_normed, -torch.log(lens_normed)], -1)
        return embeddings


class TimeStepShuffle(nn.Module):
    def forward(self, x: PaddedBatch):
        shuffled = []
        for seq, slen in zip(x.payload, x.seq_lens):
            idx = torch.randperm(slen) + 1
            pad_idx = torch.arange(slen + 1, len(seq))
            idx = torch.cat([torch.zeros(1, dtype=torch.long), idx, pad_idx])
            shuffled.append(seq[idx])

        shuffled = PaddedBatch(torch.stack(shuffled), x.seq_lens)
        return shuffled


class LastStepEncoder(nn.Module):
    def forward(self, x: PaddedBatch):
        h = x.payload[range(len(x.payload)), [l - 1 for l in x.seq_lens]]
        return h


class FirstStepEncoder(nn.Module):
    def forward(self, x: PaddedBatch):
        h = x.payload[:, 0, :]  # [B, T, H] -> [B, H]
        return h


class MeanStepEncoder(nn.Module):
    def forward(self, x: PaddedBatch):
        means = torch.stack([e[0:l].mean(dim=0) for e, l in zip(x.payload, x.seq_lens)])
        return means


class NormEncoder(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return x / (x.pow(2).sum(dim=1) + self.eps).pow(0.5).unsqueeze(-1).expand(*x.size())


class PayloadEncoder(nn.Module):
    def forward(self, x: PaddedBatch):
        return x.payload


class AllStepsHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x: PaddedBatch):
        out = self.head(x.payload)
        return PaddedBatch(out, x.seq_lens)


class AllStepsMeanHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x: PaddedBatch):
        out = self.head(x.payload)
        means = torch.tensor([e[0:l].mean() for e, l in zip(out, x.seq_lens)])
        return means


class FlattenHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: PaddedBatch):
        mask = torch.zeros(x.payload.shape, dtype=bool)
        for i, length in enumerate(x.seq_lens):
            mask[i, :length] = True

        return x.payload.flatten()[mask.flatten()]


def scoring_head(input_size, params):
    if params['pred_all_states'] and params['explicit_lengths']:
        raise AttributeError('Can not use `pred_all_states` and `explicit_lengths` together for now')

    layers = []

    if not params['pred_all_states']:
        if params['explicit_lengths']:
            e0 = ConcatLenEncoder()
            input_size += 2
            layers.append(e0)
        else:
            e0 = LastStepEncoder()
            layers.append(e0)

    if params['norm_input']:
        layers.append(NormEncoder())

    if params['use_batch_norm']:
        layers.append(nn.BatchNorm1d(input_size))

    if params.get('neural_automl', False):
        if params['pred_all_states']:
            raise AttributeError('neural_automl not supported with `pred_all_states` for now')
        layers.append(nn.BatchNorm1d(input_size))
        head = node.get_model(model_config=params['neural_automl'], input_size=input_size)
        layers.append(head)
        return nn.Sequential(*layers)

    if "head_layers" not in params:
        head_output_size = params.get('num_classes', 1)
        if params.get('objective', 'classification') == 'classification':
            if head_output_size == 1:
                h = nn.Sequential(nn.Linear(input_size, head_output_size), nn.Sigmoid(), Squeeze())
            else:
                h = nn.Sequential(nn.Linear(input_size, head_output_size), nn.LogSoftmax(dim=1))
        else:
            h = nn.Sequential(nn.Linear(input_size, head_output_size), Squeeze())
        if params['pred_all_states']:
            if params['pred_all_states_mean']:
                h = AllStepsMeanHead(h)
                layers.append(h)
            else:
                h = AllStepsHead(h)
                layers.append(h)
                if params.get('pred_flatten', False):
                    layers.append(FlattenHead())

        else:
            layers.append(h)
    else:
        head_layers = [input_size] + params["head_layers"] + [1]
        for size_in, size_out in zip(head_layers[:-1], head_layers[1:]):
            layers.append(nn.Linear(size_in, size_out))
            if size_out != 1:
                layers.append(nn.BatchNorm1d(size_out))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
            else:
                layers.append(nn.Sigmoid())
        layers.append(Squeeze())

    return nn.Sequential(*layers)


def transform(x):
    return np.sign(x) * np.log(np.abs(x) + 1)


def transform_inv(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)


def load_data_pq(file_name_in):
    table = pq.read_table(file_name_in)
    df = table.to_pandas()
    return df.to_numpy(), list(df.columns)


def top_tr_types(np_data, tr_types_col, tr_amounts_col, f):
    tr_amounts_by_value = {}
    for i in range(len(np_data)):
        tr_types_ = np_data[i][tr_types_col]
        tr_amounts_ = f(np_data[i][tr_amounts_col])
        for ix_, j in enumerate(tr_types_):
            if j in tr_amounts_by_value:
                tr_amounts_by_value[j] += tr_amounts_[ix_]
            else:
                tr_amounts_by_value[j] = tr_amounts_[ix_]

    amounts_by_val_sorted = sorted(tr_amounts_by_value.items(),
                                   key = lambda x: abs(x[1]), reverse = True)

    negative_items = []
    positive_items = []
    for pair in amounts_by_val_sorted:
        if pair[1] >= 0:
            positive_items += [pair[0]]
        else:
            negative_items += [pair[0]]
    return negative_items, positive_items


def get_distributions(np_data, tr_amounts_col, tr_types_col=None,
                      negative_items=None, positive_items=None, top_thr=None,
                      take_first_fraction=0, f=lambda x: x):
    set_top_neg_types = set(negative_items[:top_thr])
    set_top_pos_types = set(positive_items[:top_thr])

    sums_of_negative_target = [0 for _ in range(len(np_data))]
    sums_of_positive_target = [0 for _ in range(len(np_data))]

    neg_distribution = [[] for _ in range(len(np_data))]
    pos_distribution = [[] for _ in range(len(np_data))]

    for i in range(len(np_data)):
        num = len(np_data[i][tr_amounts_col])
        thr_target_ix = int(num * take_first_fraction)
        amount_target = f(np_data[i][tr_amounts_col][thr_target_ix:])

        tr_type_target = np_data[i][tr_types_col][thr_target_ix:]
        neg_tr_amounts_target = {}
        others_neg_tr_amounts_target = 0
        pos_tr_amounts_target = {}
        others_pos_tr_amounts_target = 0

        # create value counts for each top transaction type and create neg and pos sum for id

        for ixx, el in enumerate(tr_type_target):
            if amount_target[ixx] < 0:
                sums_of_negative_target[i] += amount_target[ixx]
            else:
                sums_of_positive_target[i] += amount_target[ixx]

            if el in set_top_neg_types:
                neg_tr_amounts_target[el] = neg_tr_amounts_target.get(el, 0) + amount_target[ixx]
            elif el in set_top_pos_types:
                pos_tr_amounts_target[el] = pos_tr_amounts_target.get(el, 0) + amount_target[ixx]
            elif amount_target[ixx] < 0:
                others_neg_tr_amounts_target += amount_target[ixx]
            elif amount_target[ixx] >= 0:
                others_pos_tr_amounts_target += amount_target[ixx]
            else:
                assert False, "Should not be!"

        # collect neg and pos distribution for id
        for j in negative_items[:top_thr]:
            if j in neg_tr_amounts_target:
                p_neg = neg_tr_amounts_target[j] / sums_of_negative_target[i]
            else:
                p_neg = 0.
            neg_distribution[i] += [p_neg]
        if sums_of_negative_target[i] != 0:
            neg_distribution[i] += [others_neg_tr_amounts_target / sums_of_negative_target[i]]
        else:
            neg_distribution[i] += [0.]

        for j in positive_items[:top_thr]:
            if j in pos_tr_amounts_target:
                p_pos = pos_tr_amounts_target[j] / sums_of_positive_target[i]
            else:
                p_pos = 0.
            pos_distribution[i] += [p_pos]
        if sums_of_positive_target[i] != 0:
            pos_distribution[i] += [others_pos_tr_amounts_target / sums_of_positive_target[i]]
        else:
            pos_distribution[i] += [0.]

    return sums_of_negative_target, sums_of_positive_target, neg_distribution, pos_distribution

