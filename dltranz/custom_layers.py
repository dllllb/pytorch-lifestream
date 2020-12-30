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
