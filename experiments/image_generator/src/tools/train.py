import datetime
import os
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from torch import multiprocessing as mp


def fit_model(seed, num_processes,
              max_epochs, device, model,
              train_loader, valid_loader,
              lr, step_size, gamma,
              loss_fn, metrics_fn,
              ):
    model.to(device)
    model.share_memory()
    torch.manual_seed(seed)
    np.random.seed(seed)

    _start = datetime.datetime.now()
    print(f'Start fit model in {num_processes} processes on {max_epochs} epoch '
          f'({datetime.datetime.now():%Y-%m-%dT%H:%M:%S})')

    test_fn = partial(valid, seed=seed, test_loader=valid_loader, model=model, device=device,
                      loss_fn=loss_fn, metrics_fn=metrics_fn)
    processes = []
    for rank in range(num_processes):
        kwargs = {
            'seed': seed + rank,
            'device': device,
            'model': model,
            'max_epochs': max_epochs,
            'lr': lr,
            'step_size': step_size,
            'gamma': gamma,
            'train_loader': train_loader,
            'loss_fn': loss_fn,
            'test_fn': test_fn,
        }

        if num_processes > 1:
            p = mp.Process(target=train, kwargs=kwargs)
            p.start()
            processes.append(p)
        else:
            train(**kwargs)

    for p in processes:
        p.join()
    print(f'Fit model complete '
          f'({datetime.datetime.now():%Y-%m-%dT%H:%M:%S})')
    _end = datetime.datetime.now()

    # Once training is complete, we can test the model
    result = test_fn()

    result['fit_time'] = str(_end - _start)
    return result


def train(seed, device, model, max_epochs, lr, step_size, gamma, train_loader, loss_fn, test_fn):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    torch.manual_seed(seed)
    np.random.seed(seed)

    for epoch in range(1, max_epochs + 1):
        train_epoch(epoch, model, device, train_loader, optimizer, loss_fn)
        scheduler.step()
        if epoch % 8 == 0:
            test_fn()


def _to_device(x, device):
    if type(x) is dict:
        return {k: _to_device(v, device) for k, v in x.items()}
    return x.to(device)


def _multi_head_loss(output, target, loss_fn):
    if type(output) is dict:
        part_loss = {f'loss_{k}': loss_fn(v, target[k]) for k, v in output.items()}
    else:
        part_loss = loss_fn(output, target)
    return part_loss


def _loss_collect(part_loss, cum_loss):
    if type(part_loss) is dict:
        for k, v in part_loss.items():
            cum_loss[k] += v.item()
        loss = sum((v for k, v in part_loss.items() if k.find('loss') >= 0))
    else:
        loss = part_loss

    cum_loss['loss'] += loss.item()
    return loss, cum_loss


def train_epoch(epoch, model, device, data_loader, optimizer, loss_fn):
    model.train()
    pid = os.getpid()

    cum_loss = defaultdict(float)
    for data, target in data_loader:
        optimizer.zero_grad()
        output = model(_to_device(data, device))
        part_loss = _multi_head_loss(output, _to_device(target, device), loss_fn)
        loss, cum_loss = _loss_collect(part_loss, cum_loss)
        loss.backward()
        optimizer.step()

    for k, v in cum_loss.items():
        cum_loss[k] = v / len(data_loader)

    print(f'{pid}\t{datetime.datetime.now():%Y-%m-%dT%H:%M:%S}\tTrain Epoch: {epoch} \t' + ' '.join(
        [f'{k}: {v:.5f}' for k, v in sorted(cum_loss.items())]))


def valid(seed, loss_fn, test_loader, model, device, metrics_fn):
    return valid_epoch(model, device, test_loader, loss_fn, metrics_fn)


def valid_epoch(model, device, data_loader, loss_fn, metrics_fn):
    model.eval()
    pid = os.getpid()

    cum_loss = defaultdict(float)
    with torch.no_grad():
        for data, target in data_loader:
            target = _to_device(target, device)
            output = model(_to_device(data, device))
            part_loss = _multi_head_loss(output, target, loss_fn)
            _, cum_loss = _loss_collect(part_loss, cum_loss)

            for k, v in metrics_fn.items():
                cum_loss[k] += v(output, target).item()

    for k, v in cum_loss.items():
        cum_loss[k] = v / len(data_loader)

    print(f'{pid}\t{datetime.datetime.now():%Y-%m-%dT%H:%M:%S}\tValid:\t' + ' '.join(
        [f'{k}: {v:.5f}' for k, v in sorted(cum_loss.items())]))
    return cum_loss
