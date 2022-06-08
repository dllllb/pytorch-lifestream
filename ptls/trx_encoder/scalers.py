import torch
from torch import nn as nn


class IdentityScaler(nn.Module):
    def forward(self, x):
        return x


class LogScaler(nn.Module):
    def forward(self, x):
        return x.abs().log1p() * x.sign()


class YearScaler(nn.Module):
    def forward(self, x):
        return x/365


def scaler_by_name(name):
    scaler = {
        'identity': IdentityScaler,
        'sigmoid': torch.nn.Sigmoid,
        'log': LogScaler,
        'year': YearScaler,
    }.get(name, None)

    if scaler is None:
        raise Exception(f'unknown scaler name: {name}')
    else:
        return scaler()
