from typing import List

import torch
from torch.nn import Linear, BatchNorm1d, Sigmoid, Sequential, ReLU, LogSoftmax, Flatten, Softplus
from ptls.nn.normalization import L2NormEncoder


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
            sizes of linear layers. If None without additional linear layers.
         objective: str. Options:
            None (default) - corresponds to linear output with relu
            classification - linear output with sigmoid or logsoftmax (num_classes > 1)
            regression - pure linear output
            softplus - linear output with softplus
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
            layers.append(L2NormEncoder())

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
                h = Sequential(Linear(input_size, num_classes), Sigmoid(), Flatten(0))
            else:
                h = Sequential(Linear(input_size, num_classes), LogSoftmax(dim=1))
            layers.append(h)

        elif objective == 'regression':
            if num_classes == 1:
                layers.append(Sequential(Linear(input_size, 1), Flatten(0)))
            else:
                layers.append(Linear(input_size, num_classes))

        elif objective == 'softplus':
            if num_classes == 1:
                layers.append(Sequential(Linear(input_size, num_classes), Softplus(), Flatten(0)))
            else:
                layers.append(Sequential(Linear(input_size, num_classes), Softplus()))

        elif objective is not None:
            raise AttributeError(f"Unknown objective {objective}. Supported: classification, regression and softplus.")

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
