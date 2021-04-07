import torch
import numpy as np

from dltranz.loss import DistributionTargetsLoss


def test_best_loss():
    eps = 1e-7
    params = {'head_layers':
                {'CombinedTargetHeadFromRnn':
                    {'in_size': 48,
                    'num_distr_classes': 6,
                    'pos': True,
                    'neg': True,
                    'use_gates': True,
                    'pass_samples': True
                    }
                }
            }
    prediction = (torch.tensor([[np.log(10 + 1)]]), 
                  torch.tensor([[100., 0., 0., 0., 0., 0.]]),
                  torch.tensor([[0]]),
                  torch.tensor([[0., 100., 0., 0., 0., 0.]]))

    label = np.array([[10,
                       list([1., 0., 0., 0., 0., 0.]),
                       0,
                       list([0., 1., 0., 0., 0., 0.])]])

    loss = DistributionTargetsLoss(params)
    out = loss(prediction, label)
    assert abs(out.item() - 0.) < eps
    assert type(out) is torch.Tensor


def test_loss_300():
    eps = 1e-7
    params = {'head_layers':
                {'CombinedTargetHeadFromRnn':
                    {'in_size': 48,
                    'num_distr_classes': 6,
                    'pos': True,
                    'neg': True,
                    'use_gates': True,
                    'pass_samples': True
                    }
                }
            }
    prediction = (torch.tensor([[10]]), 
                  torch.tensor([[100., 0., 0., 0., 0., 0.]]),
                  torch.tensor([[0]]),
                  torch.tensor([[0., 100., 0., 0., 0., 0.]]))

    label = np.array([[0,
                       list([1., 0., 0., 0., 0., 0.]),
                       0,
                       list([0., 1., 0., 0., 0., 0.])]])

    loss = DistributionTargetsLoss(params)
    out = loss(prediction, label)
    assert abs(out.item() - 300.) < eps
    assert type(out) is torch.Tensor

    
def test_usual_loss_first():
    eps = 1e-7
    params = {'head_layers':
                {'CombinedTargetHeadFromRnn':
                    {'in_size': 48,
                    'num_distr_classes': 6,
                    'pos': True,
                    'neg': True,
                    'use_gates': True,
                    'pass_samples': True
                    }
                }
            }
    prediction = (torch.tensor([[-1.]]), 
                  torch.tensor([[0.1, 0.2, 0.1, 0.1, 0.3, 0.2]]),
                  torch.tensor([[ 1.]]),
                  torch.tensor([[0.1, 0.2, 0.1, 0.1, 0.3, 0.2]]))

    label = np.array([[-1.,
                       list([0.1, 0.2, 0.1, 0.1, 0.3, 0.2]),
                       1.,
                       list([0.1, 0.2, 0.1, 0.1, 0.3, 0.2])]])

    loss = DistributionTargetsLoss(params)
    out = loss(prediction, label)

    assert abs(out.item() - 12.138458251953125) < eps
    assert type(out) is torch.Tensor


def test_usual_loss_second():
    eps = 1e-7
    params = {'head_layers':
                {'CombinedTargetHeadFromRnn':
                    {'in_size': 48,
                    'num_distr_classes': 6,
                    'pos': True,
                    'neg': True,
                    'use_gates': True,
                    'pass_samples': True
                    }
                }
            }
    prediction = (torch.tensor([[-1.]]), 
                  torch.tensor([[0.1, 0.2, 0.1, 0.1, 0.3, 0.2]]),
                  torch.tensor([[ 1.]]),
                  torch.tensor([[0.3, 0.5, 0., 0.1, 0.1, 0.0]]))

    label = np.array([[-10.,
                       list([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
                       8.,
                       list([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])]])

    loss = DistributionTargetsLoss(params)
    out = loss(prediction, label)

    assert abs(out.item() - 38.563011169433594) < eps
    assert type(out) is torch.Tensor

    
def test_one_class():
    eps = 1e-7
    params = {'head_layers':
                {'CombinedTargetHeadFromRnn':
                    {'in_size': 48,
                    'num_distr_classes': 6,
                    'pos': True,
                    'neg': True,
                    'use_gates': True,
                    'pass_samples': True
                    }
                }
            }
    prediction = (torch.tensor([[-1.]]), 
                  torch.tensor([[1., 0., 0., 0., 0., 0.]]),
                  torch.tensor([[ 1.]]),
                  torch.tensor([[0., 1., 0., 0., 0., 0.]]))

    label = np.array([[-1.,
                       list([1., 0., 0., 0., 0., 0.]),
                       1.,
                       list([0., 1., 0., 0., 0., 0.])]])

    loss = DistributionTargetsLoss(params)
    out = loss(prediction, label)

    assert abs(out.item() - 10.703149795532227) < eps
    assert type(out) is torch.Tensor
