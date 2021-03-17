import numpy as np
import torch
import torch.nn as nn


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


class DistributionTargetsHeadFromRnn(torch.nn.Module):
    def __init__(self, in_size=48, num_distr_classes=6):
        super().__init__()
        self.dense1 = torch.nn.Linear(in_size, 256)

        self.dense2_distributions = torch.nn.Linear(256, 128)
        self.dense2_sums = torch.nn.Linear(256, 64)
        self.dense2_gate = torch.nn.Linear(256, 64)

        self.dense3_distr_neg = torch.nn.Linear(128, num_distr_classes)
        self.dense3_distr_pos = torch.nn.Linear(128, num_distr_classes)

        self.dense3_sums_neg = torch.nn.Linear(64, 15)
        self.dense3_sums_pos = torch.nn.Linear(64, 15)
        self.dense3_gate_neg = torch.nn.Linear(64, 15)
        self.dense3_gate_pos = torch.nn.Linear(64, 15)

        self.dense4_sums_neg = torch.nn.Linear(16, 1)
        self.dense4_sums_pos = torch.nn.Linear(16, 1)
        self.dense4_gate_neg = torch.nn.Linear(16, 1)
        self.dense4_gate_pos = torch.nn.Linear(16, 1)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        device = x[0].device
        x, neg_sum_logs, pos_sum_logs = x[0], torch.tensor(x[1][:, None], device=device), torch.tensor(x[2][:, None], device=device)

        out1 = self.relu(self.dense1(x))

        out2_distr = self.relu(self.dense2_distributions(out1))
        out2_sums = self.relu(self.dense2_sums(out1))
        out2_gate = self.relu(self.dense2_gate(out1))
        out2_gate = out2_gate.detach()

        out3_distr_neg = self.dense3_distr_neg(out2_distr)
        out3_distr_pos = self.dense3_distr_pos(out2_distr)

        out3_sums_neg = self.relu(self.dense3_sums_neg(out2_sums))
        out3_sums_pos = self.relu(self.dense3_sums_pos(out2_sums))
        out3_gate_neg = self.relu(self.dense3_gate_neg(out2_gate))
        out3_gate_pos = self.relu(self.dense3_gate_pos(out2_gate))

        out3_sums_neg = torch.cat((out3_sums_neg, neg_sum_logs), dim=1).float()
        out3_sums_pos = torch.cat((out3_sums_pos, pos_sum_logs), dim=1).float()
        out3_gate_neg = torch.cat((out3_gate_neg, neg_sum_logs), dim=1).float()
        out3_gate_pos = torch.cat((out3_gate_pos, pos_sum_logs), dim=1).float()

        out4_sums_neg = self.dense4_sums_neg(out3_sums_neg)
        out4_sums_pos = self.dense4_sums_pos(out3_sums_pos)
        out4_gate_neg = self.sigmoid(self.dense4_gate_neg(out3_gate_neg))
        out4_gate_pos = self.sigmoid(self.dense4_gate_pos(out3_gate_pos))

        return neg_sum_logs * out4_gate_neg + out4_sums_neg * (1 - out4_gate_neg), out3_distr_neg, pos_sum_logs * out4_gate_pos + out4_sums_pos * (1 - out4_gate_pos), out3_distr_pos


class DistributionTargetsHeadFromAggFeatures(torch.nn.Module):
    def __init__(self, in_size=48, num_distr_classes=6):
        super().__init__()
        self.dense1 = torch.nn.Linear(in_size, 512)

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


class DummyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0], x[1], x[2], x[3]
