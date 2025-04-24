from typing import Dict, Union

import numpy as np
import torch
import warnings
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.trx_encoder.encoders import BaseEncoder
from ptls.nn.trx_encoder.scalers import IdentityScaler, scaler_by_name
from torch import nn as nn


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
        Values can be `ptls.nn.trx_encoder.scalers.IdentityScaler` implementation

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
                 numeric_values: Dict[str, Union[str, BaseEncoder]] = None,
                 custom_embeddings: Dict[str, BaseEncoder] = None,
                 out_of_index: str = 'clip',
                 fill_numeric_nan_by: float = 0,
                 ):
        super().__init__()

        if embeddings is None:
            embeddings = {}
        if custom_embeddings is None:
            custom_embeddings = {}

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

        self.custom_embeddings = torch.nn.ModuleDict(custom_embeddings)

        if numeric_values is not None:
            for col_name, scaler in numeric_values.items():
                if type(scaler) is str:
                    if scaler == 'none':
                        continue
                    self.custom_embeddings[col_name] = scaler_by_name(scaler)
                elif isinstance(scaler, BaseEncoder):
                    self.custom_embeddings[col_name] = scaler
                else:
                    raise AttributeError(f'Wrong type of numeric_values, found {type(scaler)} for "{col_name}"')
        self.fill_numeric_nan_by = fill_numeric_nan_by

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

    def get_custom_embeddings(self, x: PaddedBatch, col_name: str):
        """Returns embeddings given by custom embedder
        
        Parameters
        ----------
        x: PaddedBatch with feature dict. Each value is `(B, T)` size
        col_name: required feature name
        """
        embedder = self.custom_embeddings[col_name]
        col_name = col_name if embedder.col_name is None else embedder.col_name
        data = x.payload[col_name]

        if torch.is_tensor(data):
            if torch.isnan(data).any():
                warnings.warn(f"NaN detected in input tensor for feature '{col_name}'. Replacing NaNs with {self.fill_numeric_nan_by}.")
                data = torch.nan_to_num(data, nan=self.fill_numeric_nan_by)

        if isinstance(embedder, IdentityScaler):
            return embedder(data)
        
        embeddings = torch.nn.utils.rnn.pad_sequence([
            embedder(trx_value)
            for trx_value in data], batch_first=True
        )
        return embeddings

    def _get_custom_embeddings(self, x: PaddedBatch, col_name: str):
        """Returns embeddings given by custom embedder

        Parameters
        ----------
        x: PaddedBatch with feature dict. Each value is `(B, T)` size
        col_name: required feature name
        """
        embedder = self.custom_embeddings[col_name]
        col_name = col_name if embedder.col_name is None else embedder.col_name
        if isinstance(embedder, IdentityScaler):
            return embedder(x.payload[col_name])

        # B, T, L
        custom_col = x.payload[col_name]
        assert type(custom_col) is list
        assert type(custom_col[1]) is np.ndarray

        custom_col_flat = np.concatenate(custom_col, axis=0)  # T1+Ti+Tn, L
        custom_embeddings = torch.cat([
            embedder(custom_col_flat[i:i+embedder.batch_size])
            for i in range(0, len(custom_col_flat), embedder.batch_size)
        ], dim=0) # T1+Ti+Tn, H
        custom_embeddings = torch.split(custom_embeddings, x.seq_lens.tolist())  # B, T, H

        embeddings = torch.nn.utils.rnn.pad_sequence(custom_embeddings, batch_first=True)
        return embeddings

    @property
    def numerical_size(self):
        # :TODO: this property is only for backward compability
        return self.custom_embedding_size

    @property
    def embedding_size(self):
        return sum(e.embedding_dim for e in self.embeddings.values())

    @property
    def custom_embedding_size(self):
        return sum(e.output_size for e in self.custom_embeddings.values())

    @property
    def output_size(self):
        return self.embedding_size + self.custom_embedding_size

    @property
    def category_names(self):
        """Returns set of used feature names
        """
        return set(list(self.embeddings.keys()) +
                   list(self.numeric_values.keys()) +
                   list(self.custom_embeddings.keys()) +
                   list(self.identity_embeddings.keys())
                   )

    @property
    def category_max_size(self):
        """Returns dict with categorical feature names. Value is dictionary size"""
        return {k: v.num_embeddings for k, v in self.embeddings.items()}
