import torch

from ptls.loss import PairwiseMarginRankingLoss


def test_best_loss():
    prediction = torch.tensor([0.4, 0.5, 0.6, 0.2, 0.3])
    label = torch.tensor([1, 1, 1, 0, 0])

    loss = PairwiseMarginRankingLoss()
    out = loss(prediction, label)
    assert 0. == out
    assert type(out) is torch.Tensor


def test_usual_loss():
    prediction = torch.tensor([0.2, 0.5, 0.6, 0.7, 0.3])
    label = torch.tensor([1, 1, 1, 0, 0])

    loss = PairwiseMarginRankingLoss()
    out = loss(prediction, label)
    assert 0. < out
    assert type(out) is torch.Tensor


def test_one_class():
    prediction = torch.tensor([0.2, 0.5, 0.6, 0.7, 0.3])
    label = torch.tensor([0, 0, 0, 0, 0])

    loss = PairwiseMarginRankingLoss()
    out = loss(prediction, label)

    assert 0. == out
    assert type(out) is torch.Tensor


def test_minimal_example():
    prediction = torch.tensor([0.2, 0.5])
    label = torch.tensor([0, 1])

    loss = PairwiseMarginRankingLoss()
    out = loss(prediction, label)

    assert 0. == out
    assert type(out) is torch.Tensor
