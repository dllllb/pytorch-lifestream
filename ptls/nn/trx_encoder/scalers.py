import torch
from torch import nn


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


class PoissonScaler(nn.Module):
    """
    Explicit estimator for poissonian target with standard pytorch sampler extrapolation.
    """
    def __init__(self, kmax=33):
        super().__init__()
        self.kmax = 0.7 * kmax
        self.arange = torch.nn.Parameter(torch.arange(kmax), requires_grad=False)
        self.factor = torch.nn.Parameter(torch.special.gammaln(1 + self.arange), requires_grad=False)

    def forward(self, x):
        if self.kmax == 0:
            return torch.poisson(x)
        res = self.arange * torch.log(x).unsqueeze(-1) - self.factor * torch.ones_like(x).unsqueeze(-1)
        return res.argmax(dim=-1).float().where(x < self.kmax, torch.poisson(x))


class ExpScaler(nn.Module):
    def __init__(self, column=0):
        super().__init__()
        self.column = column

    def forward(self, x):
        if self.column is not None:
            return torch.exp(x if x.dim() == 1 else x[:, self.column].unsqueeze(-1))
        else:
            return torch.exp(x)
