import torch
from torch import nn as nn


class L2NormEncoder(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        norm = torch.linalg.norm(x, dim=-1, keepdim=True) + self.eps
        return x / norm
