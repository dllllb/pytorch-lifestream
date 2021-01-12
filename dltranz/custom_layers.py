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


class EmbeddingGenerator(torch.nn.Module):
    """
        Classical embeddings generator
    """

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
