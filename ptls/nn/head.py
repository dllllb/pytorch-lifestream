from typing import List

import torch
from torch.nn import Linear, BatchNorm1d, Sigmoid, Sequential, ReLU, LogSoftmax, Flatten, Softplus, Dropout
import torch.nn.functional as F
from ptls.nn.normalization import L2NormEncoder
from ptls.nn.seq_encoder.utils import reset_parameters


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
                 drop_probs: List[float] = None,
                 objective: str = None,
                 num_classes: int = 1,
                 n_copies=1):
        super().__init__()
        # TODO: check possibility to create empty head with do nothing

        layers = []

        if use_norm_encoder:
            layers.append(L2NormEncoder())

        if use_batch_norm:
            layers.append(BatchNorm1d(input_size))

        if drop_probs: assert len(drop_probs) == len(hidden_layers_sizes),\
            'dimensions of `drop_probs` and `hidden_layers_sizes` should be equal'

        if hidden_layers_sizes is not None:
            layers_size = [input_size] + list(hidden_layers_sizes)
            for ix, (size_in, size_out) in enumerate(zip(layers_size[:-1], layers_size[1:])):
                layers.append(Linear(size_in, size_out))
                if use_batch_norm:
                    layers.append(BatchNorm1d(size_out))
                layers.append(ReLU())
                if drop_probs:
                    layers.append(Dropout(drop_probs[ix]))
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

        self.n_copies = n_copies
        if self.n_copies == 1:
            self.model = torch.nn.Sequential(*layers)
        else:
            self.model = torch.nn.ModuleList([torch.nn.Sequential(*layers) for _ in range(self.n_copies)])
            for m in self.model:
                reset_parameters(m)

    def forward(self, x):
        if type(x) is list:
            return [self.model[i](x[i]) for i in range(len(x))]
        return self.model(x)


class SphereHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("one", torch.tensor(1).reshape(1, 1))

    def forward(self, phi):
        if type(phi) is list:
            return [self.model[i](phi[i]) for i in range(len(phi))]
        phi = F.tanh(phi)
        bs, _ = phi.shape
        phi = phi + 1
        phi[:, :-1] = phi[:, :-1] / 2
        phi = phi * torch.pi
        sin = torch.cat([torch.tile(self.one, [bs, 1]), torch.sin(phi)], dim=1)
        cos = torch.cat([torch.cos(phi), torch.tile(self.one, [bs, 1])], dim=1)
        x = torch.cumprod(sin, dim=1) * cos
        return x
