import torch

from dltranz.metric_learn.losses import ContrastiveLoss, HistogramLoss, BinomialDevianceLoss, TripletLoss, MarginLoss
from dltranz.metric_learn.sampling_strategies import AllPositivePairSelector
from dltranz.metric_learn.sampling_strategies import AllTripletSelector


def get_data():
    B, C, H = 2, 2, 2  # Batch, num Classes, Hidden size
    x = torch.tensor([
        [0.9611,  0.2761],
        [0.6419,  0.7668],
        [-0.7595,  0.6505],
        [-0.3806,  0.9247],
    ])
    # x = torch.randn(B * C, H)
    x = x.div(x.pow(2).sum(dim=1, keepdim=True).pow(0.5))
    y = torch.arange(C).view(-1, 1).expand(C, B).reshape(-1)
    return x, y


def test_contrastive_loss():
    x, y = get_data()
    sampling_strategy = AllPositivePairSelector()
    loss_fn = ContrastiveLoss(0.5, sampling_strategy)

    print(x)
    print(y)
    loss = loss_fn(x, y)
    print(loss)
    assert 1 == 1


def test_binomial_deviance_loss():
    x, y = get_data()
    sampling_strategy = AllPositivePairSelector()
    loss_fn = BinomialDevianceLoss(sampling_strategy)

    print(x)
    print(y)
    loss = loss_fn(x, y)
    print(loss)
    assert 1 == 1


def test_margin_loss():
    x, y = get_data()
    sampling_strategy = AllPositivePairSelector()
    loss_fn = MarginLoss(sampling_strategy)

    print(x)
    print(y)
    loss = loss_fn(x, y)
    print(loss)
    assert 1 == 1


def test_triplet_loss():
    x, y = get_data()
    sampling_strategy = AllTripletSelector()
    loss_fn = TripletLoss(0.5, sampling_strategy)

    print(x)
    print(y)
    loss = loss_fn(x, y)
    print(loss)
    assert 1 == 1


def test_histogram_loss1():
    x, y = get_data()
    loss_fn = HistogramLoss(num_steps=5, device='cpu')

    print(x)
    print(y)
    loss = loss_fn(x, y)
    print(loss)
    assert 1 == 1


def test_histogram_loss2():
    x, y = get_data()
    loss_fn = HistogramLoss(num_steps=51, device='cpu')

    print(x)
    print(y)
    loss = loss_fn(x, y)
    print(loss)
    assert 1 == 1

