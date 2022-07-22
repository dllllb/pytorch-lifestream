import torch
from ptls.frames.supervised.metrics import BucketAccuracy, JSDiv, UnivMeanError, LogAccuracy
from ptls.loss import ZILNLoss


def test_bucket_accuracy():
    m, B = BucketAccuracy(), 100
    assert m(torch.randn(B), torch.randn(B)) >= 0
    assert m(torch.randn(B, 3), torch.randn(B)) >= 0
    assert m(torch.randn(B), torch.randn(B, 3)) >= 0
    assert m(torch.randn(B, 3), torch.randn(B, 3)) >= 0


def test_jsdiv():
    m = JSDiv()
    y = torch.zeros(10, 5)
    assert m(torch.randn(y.shape[0], y.shape[1] + 3), y) >= 0


def test_univ_mean_error():
    m, B = UnivMeanError(), 10
    assert m(torch.randn(B), torch.randn(B)) >= 0
    assert m(torch.randn(B, 3), torch.randn(B)) >= 0
    assert m(torch.randn(B, 3), torch.randn(B, 3)) >= 0


def test_accuracy_bin():
    acc = LogAccuracy()
    y_hat = torch.tensor([0.1, 0.4, 0.6, 0.8, 0.9])
    y = torch.tensor([0, 1, 0, 1, 0])
    acc(y_hat, y)
    assert acc.compute().mul(100).int() == 40


def test_accuracy_mul():
    acc = LogAccuracy()
    y_hat = torch.log_softmax(torch.tensor([
        [-1, 2, 1],
        [1, 2, -1],
        [1, -2, 0],
        [1, 1, 2],
        [1, 1, 2],
    ]).float(), dim=1)
    y = torch.tensor([0, 1, 1, 2, 2])
    acc(y_hat, y)
    assert acc.compute().mul(100).int() == 60


def test_ziln_loss():
    loss = ZILNLoss()
    min_loss = 0.5 * torch.log(torch.tensor(loss.eps))
    y = torch.zeros(10, 5)
    assert loss(torch.zeros(y.shape[0]), y) == 0
    assert loss(torch.zeros(y.shape[0], 2), y) >= min_loss
    assert loss(torch.zeros(y.shape[0], 3), y) >= min_loss
    assert loss(torch.zeros(y.shape[0], 3 + y.shape[1]), y) >= min_loss
