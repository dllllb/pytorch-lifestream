import logging
import warnings

import torch
from ignite.contrib.handlers import ProgressBar, LRScheduler
from ignite.metrics import RunningAverage
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', module='tensorboard.compat.tensorflow_stub.dtypes')
from torch.utils.tensorboard import SummaryWriter

from dltranz.seq_encoder import PaddedBatch

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
import ignite
from bisect import bisect_right

from dltranz.train import batch_to_device, SchedulerWrapper, PrepareEpoch


def fit_model(model, train_loader, valid_loader, loss, optimizer, scheduler, params, valid_metrics, train_handlers):
    device = torch.device(params.get('device', 'cpu'))
    model.to(device)

    trainer = create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss,
        device=device,
        prepare_batch=batch_to_device,
        output_transform=lambda x, y, y_pred, loss: \
                (loss.item(), x[next(iter(x.keys()))].seq_lens if isinstance(x, dict) else x.seq_lens),
    )

    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x[1].float().mean()).attach(trainer, 'seq_len')
    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(trainer, ['loss', 'seq_len'])

    validation_evaluator = create_supervised_evaluator(
        model=model,
        device=device,
        prepare_batch=batch_to_device,
        metrics=valid_metrics
    )

    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(validation_evaluator)

    # valid_metric_name = valid_metric.__class__.__name__
    # valid_metric.attach(validation_evaluator, valid_metric_name)

    trainer.add_event_handler(Events.EPOCH_STARTED, PrepareEpoch(train_loader))

    trainer.add_event_handler(Events.EPOCH_COMPLETED, SchedulerWrapper(scheduler))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        validation_evaluator.run(valid_loader)
        metrics = validation_evaluator.state.metrics
        msgs = []
        for metric in valid_metrics:
            msgs.append(f'{metric}: {metrics[metric]:.3f}')
        pbar.log_message(
            f'Epoch: {engine.state.epoch},  {", ".join(msgs)}')

    for handler in train_handlers:
        handler(trainer, validation_evaluator, optimizer)

    trainer.run(train_loader, max_epochs=params['train.n_epoch'])

    return validation_evaluator.state.metrics
