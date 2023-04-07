import torch

from .encoders import BaseEncoder


class IdentityScaler(BaseEncoder):
    def forward(self, x):
        return x.unsqueeze(2).float()

    @property
    def output_size(self):
        return 1


class LogScaler(IdentityScaler):
    def forward(self, x):
        x = super().forward(x)
        return x.abs().log1p() * x.sign()

    @property
    def output_size(self):
        return 1


class YearScaler(IdentityScaler):
    def forward(self, x):
        x = super().forward(x)
        return x/365

    @property
    def output_size(self):
        return 1


class NumToVector(IdentityScaler):
    def __init__(self, embeddings_size):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(1, 1, embeddings_size), requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(1, 1, embeddings_size), requires_grad=True)

    def forward(self, x):
        x = super().forward(x)
        return x * self.w + self.b

    @property
    def output_size(self):
        return self.w.size(2)


class LogNumToVector(IdentityScaler):
    def __init__(self, embeddings_size):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(1, 1, embeddings_size), requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(1, 1, embeddings_size), requires_grad=True)

    def forward(self, x):
        x = super().forward(x)
        return x.abs().log1p() * x.sign() * self.w + self.b

    @property
    def output_size(self):
        return self.w.size(2)


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


class PoissonScaler(IdentityScaler):
    """
    Explicit estimator for poissonian target with standard pytorch sampler extrapolation.
    """
    def __init__(self, kmax=33):
        super().__init__()
        self.kmax = 0.7 * kmax
        self.arange = torch.nn.Parameter(torch.arange(kmax), requires_grad=False)
        self.factor = torch.nn.Parameter(torch.special.gammaln(1 + self.arange), requires_grad=False)

    def forward(self, x):
        x = super().forward(x)
        if self.kmax == 0:
            return torch.poisson(x)
        res = self.arange * torch.log(x).unsqueeze(-1) - self.factor * torch.ones_like(x).unsqueeze(-1)
        return res.argmax(dim=-1).float().where(x < self.kmax, torch.poisson(x))

    @property
    def output_size(self):
        return 1


class ExpScaler(IdentityScaler):
    def __init__(self, column=0):
        super().__init__()
        self.column = column

    def forward(self, x):
        x = super().forward(x)
        if self.column is not None:
            return torch.exp(x if x.dim() == 1 else x[:, self.column].unsqueeze(-1))
        else:
            return torch.exp(x)

    @property
    def output_size(self):
        return 1
