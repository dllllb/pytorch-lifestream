import torch
import numpy as np

from dltranz.loss import DistributionTargetsLoss


def test_best_loss():
    eps = 1e-7

    prediction = (torch.tensor([[np.log(10 + 1)]], device='cuda'), 
                  torch.tensor([[100., 0., 0., 0., 0., 0.]], device='cuda'),
                  torch.tensor([[0]], device='cuda'),
                  torch.tensor([[0., 100., 0., 0., 0., 0.]], device='cuda'))

    label = np.array([[10,
                       list([1., 0., 0., 0., 0., 0.]),
                       0,
                       list([0., 1., 0., 0., 0., 0.])]])

    loss = DistributionTargetsLoss()
    out = loss(prediction, label)
    assert abs(out.item() - 0.) < eps
    assert type(out) is torch.Tensor


def test_loss_300():
    eps = 1e-7

    prediction = (torch.tensor([[10]], device='cuda'), 
                  torch.tensor([[100., 0., 0., 0., 0., 0.]], device='cuda'),
                  torch.tensor([[0]], device='cuda'),
                  torch.tensor([[0., 100., 0., 0., 0., 0.]], device='cuda'))

    label = np.array([[0,
                       list([1., 0., 0., 0., 0., 0.]),
                       0,
                       list([0., 1., 0., 0., 0., 0.])]])

    loss = DistributionTargetsLoss()
    out = loss(prediction, label)
    assert abs(out.item() - 300.) < eps
    assert type(out) is torch.Tensor

    
def test_usual_loss_first():
    eps = 1e-7

    prediction = (torch.tensor([[-1.]], device='cuda'), 
                  torch.tensor([[0.1, 0.2, 0.1, 0.1, 0.3, 0.2]], device='cuda'),
                  torch.tensor([[ 1.]], device='cuda'),
                  torch.tensor([[0.1, 0.2, 0.1, 0.1, 0.3, 0.2]], device='cuda'))

    label = np.array([[-1.,
                       list([0.1, 0.2, 0.1, 0.1, 0.3, 0.2]),
                       1.,
                       list([0.1, 0.2, 0.1, 0.1, 0.3, 0.2])]])

    loss = DistributionTargetsLoss()
    out = loss(prediction, label)

    assert abs(out.item() - 12.138427734375) < eps
    assert type(out) is torch.Tensor


def test_usual_loss_second():
    eps = 1e-7

    prediction = (torch.tensor([[-1.]], device='cuda'), 
                  torch.tensor([[0.1, 0.2, 0.1, 0.1, 0.3, 0.2]], device='cuda'),
                  torch.tensor([[ 1.]], device='cuda'),
                  torch.tensor([[0.3, 0.5, 0., 0.1, 0.1, 0.0]], device='cuda'))

    label = np.array([[-10.,
                       list([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
                       8.,
                       list([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])]])

    loss = DistributionTargetsLoss()
    out = loss(prediction, label)

    assert abs(out.item() - 38.56253433227539) < eps
    assert type(out) is torch.Tensor


def test_one_class():
    eps = 1e-7

    prediction = (torch.tensor([[-1.]], device='cuda'), 
                  torch.tensor([[1., 0., 0., 0., 0., 0.]], device='cuda'),
                  torch.tensor([[ 1.]], device='cuda'),
                  torch.tensor([[0., 1., 0., 0., 0., 0.]], device='cuda'))

    label = np.array([[-1.,
                       list([1., 0., 0., 0., 0., 0.]),
                       1.,
                       list([0., 1., 0., 0., 0., 0.])]])

    loss = DistributionTargetsLoss()
    out = loss(prediction, label)

    assert abs(out.item() - 10.703119277954102) < eps
    assert type(out) is torch.Tensor
