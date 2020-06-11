import torch

from dltranz.metric_learn.losses import HistogramLoss


def test_histogram_loss1():
    loss_fn = HistogramLoss(num_steps=5, device='cpu')

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
    print(x)
    print(y)
    loss = loss_fn(x, y)
    print(loss)
    assert 1 == 1

def test_histogram_loss2():
    loss_fn = HistogramLoss(num_steps=51, device='cpu')

    B, C, H = 2, 2, 2  # Batch, num Classes, Hidden size
    x = torch.tensor([
        [0.9611,  0.2761],
        [0.9611,  0.2761],
        [-0.7595,  0.6505],
        [-0.3806,  -0.7595],
    ])

    # x = torch.randn(B * C, H)
    x = x.div(x.pow(2).sum(dim=1, keepdim=True).pow(0.5))
    y = torch.arange(C).view(-1, 1).expand(C, B).reshape(-1)
    print(x)
    print(y)
    loss = loss_fn(x, y)
    print(loss)
    assert 1 == 1

