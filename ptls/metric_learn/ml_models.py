# coding: utf-8
import logging

import torch
from torch import nn as nn

from ptls.data_load.padded_batch import PaddedBatch

logger = logging.getLogger(__name__)


def projection_head(input_size, output_size):
    layers = [
        torch.nn.Linear(input_size, input_size),
        torch.nn.ReLU(),
        torch.nn.Linear(input_size, output_size),
    ]
    m = torch.nn.Sequential(*layers)
    return m


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
        self.projection_ml_head = projection_head(params.rnn.hidden_size, params.ml_projection_head.output_size)
        self.projection_aug_head = torch.nn.Sequential(
            projection_head(params.rnn.hidden_size, params.aug_projection_head.output_size),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        encoder_output = self.ml_model(x)
        ml_head_output = self.projection_ml_head(encoder_output)
        aug_head_output = self.projection_aug_head(encoder_output)
        return aug_head_output, ml_head_output
