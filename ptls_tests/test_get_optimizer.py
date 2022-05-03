import torch

from ptls.train import get_optimizer


def test_get_optimizer():
    model = torch.nn.Linear(10, 1)
    params = {
        'train.lr': 0.01,
        'train.weight_decay': 0.01,
    }
    optim = get_optimizer(model, params)


def test_get_optimizer_params():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.Sigmoid(),
        torch.nn.Linear(5, 1),
        torch.nn.Sigmoid(),
    )
    print([k for k, v in model.named_parameters()])
    params = {
        'train.lr': 0.01,
        'train.weight_decay': 0.01,
        'train.optimiser_params': {
            '0.weight': {'lr': 0.001},
        },
    }

    optim = get_optimizer(model, params)
    for grp in optim.param_groups:
        if len(grp['params']) == 1:
            assert grp['lr'] == 0.001
        elif len(grp['params']) == 3:
            assert grp['lr'] == 0.01
        else:
            raise AssertionError('Error in test. Only 4 parameters in total')


def test_get_optimizer_2params():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.Sigmoid(),
        torch.nn.Linear(10, 5),
        torch.nn.Sigmoid(),
        torch.nn.Linear(5, 1),
        torch.nn.Sigmoid(),
    )
    print([k for k, v in model.named_parameters()])
    params = {
        'train.lr': 0.01,
        'train.weight_decay': 0.01,
        'train.optimiser_params': {
            '0.weight': {'lr': 0.001},
            '2.weight': {'lr': 0.001},
        },
    }

    optim = get_optimizer(model, params)
    for grp in optim.param_groups:
        if len(grp['params']) == 1:
            assert grp['lr'] == 0.001
        elif len(grp['params']) == 4:
            assert grp['lr'] == 0.01
        else:
            raise AssertionError('Error in test. Only 6 parameters in total')
