import torch

from ptls.frames.supervised.metrics import RMSE, BucketAccuracy, LogAccuracy


def test_rmse():
    rmse, B = RMSE(), 10
    assert rmse(torch.randn(B), torch.randn(B)) >= 0
    assert rmse(torch.randn(B, 3), torch.randn(B)) >= 0
    assert rmse(torch.randn(B, 3), torch.randn(B, 3)) >= 0


def test_bucket_accuracy():
    bacc, B = BucketAccuracy(), 100
    assert bacc(torch.randn(B), torch.randn(B)) >= 0
    assert bacc(torch.randn(B, 3), torch.randn(B)) >= 0
    assert bacc(torch.randn(B), torch.randn(B, 3)) >= 0
    assert bacc(torch.randn(B, 3), torch.randn(B, 3)) >= 0


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