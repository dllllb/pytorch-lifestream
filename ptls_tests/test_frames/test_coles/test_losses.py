import torch

from ptls.frames.coles.losses.complex_loss import ComplexLoss
from ptls.frames.coles.losses import (
    MarginLoss, HistogramLoss, TripletLoss, BinomialDevianceLoss, ContrastiveLoss,
    CentroidLoss, CentroidSoftmaxLoss,
)
from ptls.frames.coles.sampling_strategies import AllPositivePairSelector, AllTripletSelector


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
    loss_fn = HistogramLoss(num_steps=5)

    print(x)
    print(y)
    loss = loss_fn(x, y)
    print(loss)
    assert 1 == 1


def test_histogram_loss2():
    x, y = get_data()
    loss_fn = HistogramLoss(num_steps=51)

    print(x)
    print(y)
    loss = loss_fn(x, y)
    print(loss)
    assert 1 == 1


def test_complex_loss():
    B, C, H = 2, 2, 2  # Batch, num Classes, Hidden size
    x_ml = torch.randn(B * C, H)
    x_ml = x_ml.div(x_ml.pow(2).sum(dim=1, keepdim=True).pow(0.5))
    y_ml = torch.arange(C).view(-1, 1).expand(C, B).reshape(-1, 1)

    sampling_strategy = AllPositivePairSelector()
    ml_loss = MarginLoss(sampling_strategy)

    x_aug = torch.randn(B * C, C)
    x_aug = x_aug.div(x_aug.sum(dim=1, keepdim=True))
    y_aug = torch.arange(B).view(1, -1).expand(C, B).reshape(-1, 1)
    aug_loss_fn = torch.nn.NLLLoss()

    y = torch.cat([y_ml, y_aug], axis=1)
    loss_fn = ComplexLoss(ml_loss, aug_loss_fn, 0.5)
    loss = loss_fn((x_aug, x_ml), y)
    print(loss)
    assert 1 == 1


def test_centroid_loss():
    x, y = get_data()
    loss_fn = CentroidLoss(class_num=None)

    loss = loss_fn(x, y)
    assert loss > 0


def test_centroid_loss_class_num():
    x = torch.tensor([
        [0, 0],  # 1, 1
        [2, 2],  # 1, 1
        [-1, 0],  # 0, -1
        [1, -2],  # 0, -1
    ])
    y = torch.tensor([0, 0, 1, 1])
    loss_fn = CentroidLoss(class_num=2)

    loss = loss_fn(x, y)
    assert loss == 2


def test_centroid_softmax_loss():
    x, y = get_data()
    loss_fn = CentroidSoftmaxLoss(class_num=None, temperature=1.0)

    loss = loss_fn(x, y)
    assert loss > 0
