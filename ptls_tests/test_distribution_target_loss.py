import torch
import numpy as np

from ptls.loss import DistributionTargetsLoss


def test_best_loss():
    eps = 1e-7

    prediction = {'neg_sum': torch.tensor([[np.log(10 + 1)]]), 
                  'neg_distribution': torch.tensor([[100., 0., 0., 0., 0., 0.]]),
                  'pos_sum': torch.tensor([[0]]),
                  'pos_distribution': torch.tensor([[0., 100., 0., 0., 0., 0.]])}

    label = {'neg_sum': np.array([[10]]),
             'neg_distribution': np.array([[1., 0., 0., 0., 0., 0.]]),
             'pos_sum': np.array([[0]]),
             'pos_distribution': np.array([[0., 1., 0., 0., 0., 0.]])}

    loss = DistributionTargetsLoss()
    out = loss(prediction, label)
    assert abs(out.item() - 0.) < eps
    assert type(out) is torch.Tensor


def test_loss_300():
    eps = 1e-7
    
    prediction = {'neg_sum': torch.tensor([[10]]), 
                  'neg_distribution': torch.tensor([[100., 0., 0., 0., 0., 0.]]),
                  'pos_sum': torch.tensor([[0]]),
                  'pos_distribution': torch.tensor([[0., 100., 0., 0., 0., 0.]])}

    label = {'neg_sum': np.array([[0]]),
             'neg_distribution': np.array([[1., 0., 0., 0., 0., 0.]]),
             'pos_sum': np.array([[0]]),
             'pos_distribution': np.array([[0., 1., 0., 0., 0., 0.]])}

    loss = DistributionTargetsLoss()
    out = loss(prediction, label)
    assert abs(out.item() - 300.) < eps
    assert type(out) is torch.Tensor

    
def test_usual_loss_first():
    eps = 1e-7

    prediction = {'neg_sum': torch.tensor([[-1.]]), 
                  'neg_distribution': torch.tensor([[0.1, 0.2, 0.1, 0.1, 0.3, 0.2]]),
                  'pos_sum': torch.tensor([[ 1.]]),
                  'pos_distribution': torch.tensor([[0.1, 0.2, 0.1, 0.1, 0.3, 0.2]])}

    label = {'neg_sum': np.array([[-1.]]),
             'neg_distribution': np.array([[0.1, 0.2, 0.1, 0.1, 0.3, 0.2]]),
             'pos_sum': np.array([[1.]]),
             'pos_distribution': np.array([[0.1, 0.2, 0.1, 0.1, 0.3, 0.2]])}

    loss = DistributionTargetsLoss()
    out = loss(prediction, label)

    assert abs(out.item() - 12.138458251953125) < eps
    assert type(out) is torch.Tensor


def test_usual_loss_second():
    eps = 1e-7
    
    prediction = {'neg_sum': torch.tensor([[-1.]]), 
                  'neg_distribution': torch.tensor([[0.1, 0.2, 0.1, 0.1, 0.3, 0.2]]),
                  'pos_sum': torch.tensor([[ 1.]]),
                  'pos_distribution': torch.tensor([[0.3, 0.5, 0., 0.1, 0.1, 0.0]])}

    label = {'neg_sum': np.array([[-10.]]),
             'neg_distribution': np.array([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]]),
             'pos_sum': np.array([[8.]]),
             'pos_distribution': np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])}

    loss = DistributionTargetsLoss()
    out = loss(prediction, label)

    assert abs(out.item() - 38.563011169433594) < eps
    assert type(out) is torch.Tensor

    
def test_one_class():
    eps = 1e-7

    prediction = {'neg_sum': torch.tensor([[-1.]]), 
                  'neg_distribution': torch.tensor([[1., 0., 0., 0., 0., 0.]]),
                  'pos_sum': torch.tensor([[ 1.]]),
                  'pos_distribution': torch.tensor([[0., 1., 0., 0., 0., 0.]])}

    label = {'neg_sum': np.array([[-1.]]),
             'neg_distribution': np.array([[1., 0., 0., 0., 0., 0.]]),
             'pos_sum': np.array([[1.]]),
             'pos_distribution': np.array([[0., 1., 0., 0., 0., 0.]])}

    loss = DistributionTargetsLoss()
    out = loss(prediction, label)

    assert abs(out.item() - 10.703149795532227) < eps
    assert type(out) is torch.Tensor
