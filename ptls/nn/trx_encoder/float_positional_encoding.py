import math

import torch
from torch import nn as nn


class FloatPositionalEncoding(nn.Module):
    def __init__(self, out_size):
        super(FloatPositionalEncoding, self).__init__()

        self.out_size = out_size

    def forward(self, position):
        """

        :param position: B x T
        :return: B x T x H
        """
        div_term = torch.exp(torch.arange(0, self.out_size, 2).float() * (-math.log(10000.0) / self.out_size))
        div_term = div_term.unsqueeze(0).unsqueeze(0)
        div_term = div_term.to(device=position.device)

        position = position.unsqueeze(2)

        pe = torch.cat([torch.sin(position * div_term), torch.cos(position * div_term)], dim=2)
        self.register_buffer('pe', pe)

        return pe
