import torch

from ptls.loss import ZILNLoss


def test_ziln_loss():
    ziln, B = ZILNLoss(), 10
    assert ziln(torch.randn(B, 3), torch.randn(B, 1)) >= 0
    assert ziln(torch.randn(B, 3 + 3), torch.randn(B, 3)) >= 0