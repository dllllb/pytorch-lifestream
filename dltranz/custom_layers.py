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


class MixedTrxFeaturesHead(nn.Module):
    def __init__(self, trx_tail, features_tail, head):
        super().__init__()
        self.trx_tail = trx_tail
        self.features_tail = features_tail
        self.head = head

    def forward(self, x):
        padded_batch, features = x
        return self.head(self.features_tail(features))
        t = torch.cat([self.trx_tail(padded_batch), self.features_tail(features)], axis=1)

        return self.head(t)


def MLP(input_size, params):
    layers = [nn.BatchNorm1d(input_size)]
    layers_size = [input_size] + params['hidden_layers_size']
    for size_in, size_out in zip(layers_size[:-1], layers_size[1:]):
        layers.append(nn.Linear(size_in, size_out))
        layers.append(nn.ReLU())
        if params['drop_p']:
            layers.append(nn.Dropout(params['drop_p']))
        layers.append(nn.BatchNorm1d(size_out))

    if params.get('objective', None) == 'classification':
        head_output_size = params.get('num_classes', 1)
        if head_output_size == 1:
            h = nn.Sequential(nn.Linear(layers_size[-1], head_output_size), nn.Sigmoid(), Squeeze())
        else:
            h = nn.Sequential(nn.Linear(layers_size[-1], head_output_size), nn.LogSoftmax(dim=1))
        layers.append(h)

    elif params.get('objective', None) == 'multilabel_classification':
        head_output_size = params['num_classes']
        h = nn.Sequential(nn.Linear(layers_size[-1], head_output_size), nn.Sigmoid())
        layers.append(h)

    elif params.get('objective', None) == 'regression':
        h = nn.Sequential(nn.Linear(layers_size[-1], 1), Squeeze())
        layers.append(h)

    return nn.Sequential(*layers)
