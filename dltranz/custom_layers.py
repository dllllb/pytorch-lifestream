import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl


class DropoutEncoder(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = torch.FloatTensor(x.shape[1]).uniform_(0, 1) <= self.p
            x = x.masked_fill(mask.to(x.device), 0)
        return x


class Squeeze(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.squeeze()


class CatLayer(nn.Module):
    def __init__(self, left_tail, right_tail):
        super().__init__()
        self.left_tail = left_tail
        self.right_tail = right_tail

    def forward(self, x):
        l, r = x
        t = torch.cat([self.left_tail(l), self.right_tail(r)], axis=1)
        return t


class MLP(nn.Module):
    def __init__(self, input_size, params):
        super().__init__()
        self.input_size = input_size
        self.use_batch_norm = params.get('use_batch_norm', True)

        layers = []
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(input_size))
        layers_size = [input_size] + params['hidden_layers_size']
        for size_in, size_out in zip(layers_size[:-1], layers_size[1:]):
            layers.append(nn.Linear(size_in, size_out))
            layers.append(nn.ReLU())
            if params['drop_p']:
                layers.append(nn.Dropout(params['drop_p']))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(size_out))
            self.output_size = layers_size[-1]

        if params.get('objective', None) == 'classification':
            head_output_size = params.get('num_classes', 1)
            if head_output_size == 1:
                h = nn.Sequential(nn.Linear(layers_size[-1], head_output_size), nn.Sigmoid(), Squeeze())
            else:
                h = nn.Sequential(nn.Linear(layers_size[-1], head_output_size), nn.LogSoftmax(dim=1))
            layers.append(h)
            self.output_size = head_output_size

        elif params.get('objective', None) == 'multilabel_classification':
            head_output_size = params['num_classes']
            h = nn.Sequential(nn.Linear(layers_size[-1], head_output_size), nn.Sigmoid())
            layers.append(h)
            self.output_size = head_output_size

        elif params.get('objective', None) == 'regression':
            h = nn.Sequential(nn.Linear(layers_size[-1], 1), Squeeze())
            layers.append(h)
            self.output_size = 1

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class TabularRowEncoder(torch.nn.Module):
    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dim):
        """ This is an embedding module for an entier set of features

        Parameters
        ----------
        input_dim : int
            Number of features coming as input (number of columns)
        cat_dims : list of int
            Number of modalities for each categorial features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs
        cat_emb_dim : int or list of int
            Embedding dimension for each categorical features
            If int, the same embdeding dimension will be used for all categorical features
        """
        super().__init__()
        if cat_dims == [] or cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            return

        self.skip_embedding = False
        self.cat_idxs = cat_idxs
        if isinstance(cat_emb_dim, int):
            self.cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
        else:
            self.cat_emb_dims = cat_emb_dim

        # check that all embeddings are provided
        if len(self.cat_emb_dims) != len(cat_dims):
            msg = """ cat_emb_dim and cat_dims must be lists of same length, got {len(self.cat_emb_dims)}
                      and {len(cat_dims)}"""
            raise ValueError(msg)
        self.post_embed_dim = int(input_dim + np.sum(self.cat_emb_dims) - len(self.cat_emb_dims))

        self.embeddings = torch.nn.ModuleList()

        # Sort dims by cat_idx
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        self.cat_emb_dims = [self.cat_emb_dims[i] for i in sorted_idxs]

        for cat_dim, emb_dim in zip(cat_dims, self.cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))
            # TODO: NoisyEmbeddings

    def forward(self, x):
        """
        Apply embdeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x

        cols = []
        prev_cat_idx = -1
        for cat_feat_counter, cat_idx in enumerate(self.cat_idxs):
            if cat_idx > prev_cat_idx + 1:
                cols.append(x[:, prev_cat_idx + 1:cat_idx].float())
            cols.append(self.embeddings[cat_feat_counter](x[:, cat_idx].long()))
            prev_cat_idx = cat_idx

        if prev_cat_idx + 1 < x.shape[1]:
            cols.append(x[:, prev_cat_idx + 1:].float())

        # concat
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings

    @property
    def output_size(self):
        return self.post_embed_dim


class EmbedderNetwork(nn.Module):
    def __init__(self, embedder, network):
        super().__init__()
        self.embedder = embedder
        self.network = network

    def forward(self, x):
        embedded_x = self.embedder(x)
        return embedded_x, self.network(embedded_x)


class DistributionTargetHead(torch.nn.Module):
    def __init__(self, in_size=256, num_distr_classes_pos=4, num_distr_classes_neg=14, pos=True, neg=True):
        super().__init__()
        self.pos, self.neg = pos, neg
        self.dense1 = torch.nn.Linear(in_size, 128)

        self.dense2_neg = torch.nn.Linear(128, num_distr_classes_neg) if self.neg else None
        self.dense2_pos = torch.nn.Linear(128, num_distr_classes_pos) if self.pos else None

        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.dense1(x))
        out2_pos = out2_neg = 0
        if self.pos:
            out2_pos = self.dense2_pos(out1)
        if self.neg:
            out2_neg = self.dense2_neg(out1)
        return out2_neg, out2_pos


class RegressionTargetHead(torch.nn.Module):
    def __init__(self, in_size=256, gates=True, pos=True, neg=True, pass_samples=True):
        super().__init__()
        self.pos, self.neg = pos, neg
        self.pass_samples = pass_samples
        self.dense1 = torch.nn.Linear(in_size, 64)
        if pass_samples:
            self.dense2_neg = torch.nn.Linear(64, 15) if self.neg else None
            self.dense2_pos = torch.nn.Linear(64, 15) if self.pos else None
        else:
            self.dense2_neg = torch.nn.Linear(64, 16) if self.neg else None
            self.dense2_pos = torch.nn.Linear(64, 16) if self.pos else None
        self.dense3_neg = torch.nn.Linear(16, 1) if self.neg else None
        self.dense3_pos = torch.nn.Linear(16, 1) if self.pos else None
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.gates = gates

    def forward(self, x, neg_sum_logs=None, pos_sum_logs=None):
        out1 = self.relu(self.dense1(x))
        if self.gates:
            out1 = out1.detach()

        out3_neg = out3_pos = 0
        if self.pos:
            out2_pos = self.relu(self.dense2_pos(out1))
            if self.pass_samples:
                out2_pos = torch.cat((out2_pos, pos_sum_logs), dim=1).float()
            out3_pos = self.dense3_pos(out2_pos)
            out3_pos = self.sigmoid(out3_pos) if self.gates else out3_pos
        if self.neg:
            out2_neg = self.relu(self.dense2_neg(out1))
            if self.pass_samples:
                out2_neg = torch.cat((out2_neg, neg_sum_logs), dim=1).float()
            out3_neg = self.dense3_neg(out2_neg)
            out3_neg = self.sigmoid(out3_neg) if self.gates else out3_neg

        return out3_neg, out3_pos


class CombinedTargetHeadFromRnn(torch.nn.Module):
    def __init__(self, in_size=48, num_distr_classes_pos=4, num_distr_classes_neg=14, pos=True, neg=True, use_gates=True, pass_samples=True):
        super().__init__()

        self.pos, self.neg = pos, neg
        self.use_gates = use_gates
        self.dense = torch.nn.Linear(in_size, 256)
        self.distribution = DistributionTargetHead(256, num_distr_classes_pos, num_distr_classes_neg, pos, neg)
        self.regr_sums = RegressionTargetHead(256, False, pos, neg, pass_samples)
        self.regr_gates = RegressionTargetHead(256, True, pos, neg, pass_samples) if self.use_gates else None

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        device = x[0].device if isinstance(x, tuple) else x.device
        sum_logs2 = torch.tensor(x[2][:, None], device=device) if isinstance(x, tuple) and len(x) > 2 else 0
        x, sum_logs1 = (x[0], torch.tensor(x[1][:, None], device=device)) if isinstance(x, tuple) else (x, 0)  # negative sum logs if len(x) == 3
        if not self.neg and self.pos:
            sum_logs1, sum_logs2 = sum_logs2, sum_logs1


        out1 = self.relu(self.dense(x))
        distr_neg, distr_pos = self.distribution(out1)

        sums_neg, sums_pos = self.regr_sums(out1, sum_logs1, sum_logs2)
        gate_neg, gate_pos = self.regr_gates(out1, sum_logs1, sum_logs2) if self.use_gates else (0, 0)

        return {'neg_sum': sum_logs1 * gate_neg + sums_neg * (1 - gate_neg),
                'neg_distribution': distr_neg,
                'pos_sum': sum_logs2 * gate_pos + sums_pos * (1 - gate_pos),
                'pos_distribution': distr_pos}


class TargetHeadFromAggFeatures(torch.nn.Module):
    def __init__(self, in_size=48, num_distr_classes=6):
        super().__init__()
        self.dense1 = torch.nn.Linear(in_size, 512)

        self.distribution = DistributionTargetHead(512, num_distr_classes)

        self.dense2_sums = torch.nn.Linear(512, 64)
        self.dense3_sums_neg = torch.nn.Linear(64, 1)
        self.dense3_sums_pos = torch.nn.Linear(64, 1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.dense1(x))

        distr_neg, distr_pos = self.distribution(out1)

        out2_sums = self.relu(self.dense2_sums(out1))
        out3_sums_neg = self.dense3_sums_neg(out2_sums)
        out3_sums_pos = self.dense3_sums_pos(out2_sums)

        return {'neg_sum': out3_sums_neg,
                'neg_distribution': distr_neg,
                'pos_sum': out3_sums_pos,
                'pos_distribution': distr_pos}


class DummyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return {'neg_sum': x[0],
                'neg_distribution': x[1],
                'pos_sum': x[2],
                'pos_distribution': x[3]}

