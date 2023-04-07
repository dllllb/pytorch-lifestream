import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(self, col_name: str = None):
        super().__init__()
        self.col_name = col_name

    @property
    def output_size(self) -> int:
        raise NotImplementedError()


class IdentityEncoder(BaseEncoder):
    def __init__(self, output_size: int, col_name: str = None):
        super().__init__(col_name)
        self.__output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @property
    def output_size(self) -> int:
        return self.__output_size
