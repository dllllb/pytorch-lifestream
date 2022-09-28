from typing import Union
from typing import Dict

import torch
from torch import nn as nn

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.trx_encoder.scalers import scaler_by_name, BaseScaler


class TrxEncoderBase(nn.Module):
    """Base class for TrxEncoders.

    This provides a parts of embeddings for feature fields.
    This doesn't provide a full embedding of transaction.

    Parameters
    ----------
    embeddings:
        dict with categorical feature names.
        Values must be like this `{'in': dictionary_size, 'out': embedding_size}`
        These features will be encoded with lookup embedding table of shape (dictionary_size, embedding_size)
        Values can be a `torch.nn.Embedding` implementation
    numeric_values:
        dict with numerical feature names.
        Values must be a string with scaler_name.
        Possible values are: 'identity', 'sigmoid', 'log', 'year'.
        These features will be scaled with selected scaler.
        Values can be `ptls.nn.trx_encoder.scalers.BaseScaler` implementatoin

        One field can have many scalers. In this case key become alias and col name should be in scaler.
        Example:
        >>> TrxEncoderBase(
        >>>     numeric_values={
        >>>         'amount_orig': IdentityScaler(col_name='amount'),
        >>>         'amount_log': LogScaler(col_name='amount'),
        >>>     },
        >>> )

    out_of_index:
        How to process a categorical indexes which are greater than dictionary size.
        'clip' - values will be collapsed to maximum index. This works well for frequency encoded categories.
            We join infrequent categories to one.
        'assert' - raise an error of invalid index appear.
    """
    def __init__(self,
                 embeddings: Dict[str, Union[Dict, torch.nn.Embedding]] = None,
                 numeric_values: Dict[str, Union[str, BaseScaler]] = None,
                 out_of_index: str = 'clip',
                 ):
        super().__init__()

        if embeddings is None:
            embeddings = {}
        if numeric_values is None:
            numeric_values = {}

        self.embeddings = torch.nn.ModuleDict()
        for col_name, emb_props in embeddings.items():
            if type(emb_props) is dict:
                if emb_props.get('disabled', False):
                    continue
                if emb_props['in'] == 0 or emb_props['out'] == 0:
                    continue
                if emb_props['in'] < 3:
                    raise AttributeError(f'At least 3 should be in `embeddings.{col_name}.in`. '
                                         f'0-padding and at least two different embedding indexes')
                self.embeddings[col_name] = torch.nn.Embedding(
                    num_embeddings=emb_props['in'],
                    embedding_dim=emb_props['out'],
                    padding_idx=0,
                )
            elif isinstance(emb_props, torch.nn.Embedding):
                self.embeddings[col_name] = emb_props
            else:
                raise AttributeError(f'Wrong type of embeddings, found {type(col_name)} for "{col_name}"')
        self.out_of_index = out_of_index
        assert out_of_index in ('clip', 'assert')

        self.numeric_values = torch.nn.ModuleDict()
        for col_name, scaler_name in numeric_values.items():
            if type(scaler_name) is str:
                if scaler_name == 'none':
                    continue
                self.numeric_values[col_name] = scaler_by_name(scaler_name)
            elif isinstance(scaler_name, BaseScaler):
                self.numeric_values[col_name] = scaler_name
            else:
                raise AttributeError(f'Wrong type of numeric_values, found {type(scaler_name)} for "{col_name}"')

    def get_category_indexes(self, x: PaddedBatch, col_name: str):
        """Returns category feature values clipped to dictionary size.

        Parameters
        ----------
        x: PaddedBatch with feature dict. Each value is `(B, T)` size
        col_name: required feature name
        """
        v = x.payload[col_name].long()
        max_size = self.embeddings[col_name].num_embeddings

        if self.out_of_index == 'clip':
            return v.clip(0, max_size - 1)
        if self.out_of_index == 'assert':
            out_of_index_cnt = (v >= max_size).sum()
            if out_of_index_cnt > 0:
                raise IndexError(f'Found indexes greater than dictionary size for "{col_name}"')
            return v
        raise AssertionError(f'Unknown out_of_index value: {self.out_of_index}')

    def get_category_embeddings(self, x: PaddedBatch, col_name: str):
        indexes = self.get_category_indexes(x, col_name)
        return self.embeddings[col_name](indexes)

    def get_numeric_scaled(self, x: PaddedBatch, col_name):
        """Returns numerical feature values transformed with selected scaler.

        Parameters
        ----------
        x: PaddedBatch with feature dict. Each value is `(B, T)` size
        col_name: required feature name
        """
        scaler = self.numeric_values[col_name]
        if scaler.col_name is None:
            v = x.payload[col_name].unsqueeze(2).float()
        else:
            v = x.payload[scaler.col_name].unsqueeze(2).float()
        return scaler(v)

    @property
    def numerical_size(self):
        return sum(n.output_size for n in self.numeric_values.values())

    @property
    def embedding_size(self):
        return sum(e.embedding_dim for e in self.embeddings.values())

    @property
    def output_size(self):
        s = self.numerical_size + self.embedding_size
        return s

    @property
    def category_names(self):
        """Returns set of used feature names
        """
        return set([field_name for field_name in self.embeddings.keys()] +
                   [value_name for value_name in self.scalers.keys()]
                   )

    @property
    def category_max_size(self):
        """Returns dict with categorical feature names. Value is dictionary size
        """
        return {k: v['in'] for k, v in self.embeddings.items()}
