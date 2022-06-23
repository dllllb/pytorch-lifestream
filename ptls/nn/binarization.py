from torch import nn as nn
from torch.autograd import Function


class Binarization(Function):
    @staticmethod
    def forward(self, x):
        q = (x > 0).float()
        return 2*q - 1

    @staticmethod
    def backward(self, grad_outputs):
        return grad_outputs


binary = Binarization.apply


class BinarizationLayer(nn.Module):
    def __init__(self, hs_from, hs_to):
        super(BinarizationLayer, self).__init__()
        self.linear = nn.Linear(hs_from, hs_to, bias=False)

    def forward(self, x):
        return binary(self.linear(x))