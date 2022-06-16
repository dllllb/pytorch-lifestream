from typing import List

import torch
from torch.nn import Linear, BatchNorm1d, Sigmoid, Sequential, ReLU, LogSoftmax

from ptls.custom_layers import Squeeze
from ptls.nn.seq_encoder.utils import NormEncoder


class Head(torch.nn.Module):
    r"""Head for the sequence encoder

    Parameters
    ----------
         input_size: int
            input size
         use_norm_encoder: bool. Default: False
            whether to use normalization layers before the head
         use_batch_norm: bool. Default: False.
            whether to use BatchNorm.
         hidden_layers_sizes: List[int]. Default: None.
            sizes of linear layers. If None without additional linear layers. Default = None,
         objective: str. Default: None.
            Options: None, 'classification', 'regression'. Default = None.
         num_classes: int. Default: 1.
            The number of classed in classification problem. Default correspond to binary classification.

     """
    def __init__(self,
                 input_size: int = None,
                 use_norm_encoder: bool = False,
                 use_batch_norm: bool = False,
                 hidden_layers_sizes: List[int] = None,
                 objective: str = None,
                 num_classes: int = 1):
        super().__init__()
        # TODO: check possibility to create empty head with do nothing

        layers = []

        if use_norm_encoder:
            layers.append(NormEncoder())

        if use_batch_norm:
            layers.append(BatchNorm1d(input_size))

        if hidden_layers_sizes is not None:
            layers_size = [input_size] + list(hidden_layers_sizes)
            for size_in, size_out in zip(layers_size[:-1], layers_size[1:]):
                layers.append(Linear(size_in, size_out))
                layers.append(ReLU())
                if use_batch_norm:
                    layers.append(BatchNorm1d(size_out))
                input_size = size_out

        if objective == 'classification':
            if num_classes == 1:
                h = Sequential(Linear(input_size, num_classes), Sigmoid(), Squeeze())
            else:
                h = Sequential(Linear(input_size, num_classes), LogSoftmax(dim=1))
            layers.append(h)

        elif objective == 'regression':
            h = Sequential(Linear(input_size, 1), Squeeze())
            layers.append(h)

        elif objective is not None:
            raise AttributeError(f"unknown objective {objective}. Supported: classification and regression")

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
