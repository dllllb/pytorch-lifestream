import logging
import warnings

import torch
from copy import deepcopy
from ignite.contrib.handlers import ProgressBar, LRScheduler, create_lr_scheduler_with_warmup
from ignite.contrib.handlers.param_scheduler import ParamScheduler
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage
import numpy as np
import pandas as pd
from math import sqrt

warnings.filterwarnings('ignore', module='tensorboard.compat.tensorflow_stub.dtypes')
from torch.utils.tensorboard import SummaryWriter
from dltranz.trx_encoder import PaddedBatch
from dltranz.swa import SWA

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
import ignite
from bisect import bisect_right

logger = logging.getLogger(__name__)


def batch_to_device(batch, device, non_blocking):
    x, y = batch

    if isinstance(x, dict):
        batches = {}
        for key, sx in x.items():
            batches[key] = sx.to(device=device, non_blocking=non_blocking)
        new_y = y.to(device=device, non_blocking=non_blocking)
        return batches, new_y

    elif isinstance(x, tuple):
        batches = []
        for sx in x:
            batches.append(sx.to(device=device, non_blocking=non_blocking))
        new_y = y.to(device=device, non_blocking=non_blocking)
        return tuple(batches), new_y

    else:
        new_x = x.to(device=device, non_blocking=non_blocking)
        new_y = y.to(device=device, non_blocking=non_blocking)
        return new_x, new_y


def get_optimizer(model, params):
    """Returns optimizer

    :param model: model with his `model.named_parameters()`
    :param params: dict with options:
        ['train.lr']: `lr` for Adam optimizer
        ['train.weight_decay']: `weight_decay` for Adam optimizer
        ['train.optimiser_params']: (optional) list of tuples (par_name, options),
            each tuple define new parameter group.
            `par_name` is end of parameter name from `model.named_parameters()` for this parameter group
            'options' is dict with options for this parameter group
    :return:
    """
    optimiser_params = params.get('train.optimiser_params', None)
    if optimiser_params is None:
        parameters = model.parameters()
    else:
        parameters = []
        for par_name, options in optimiser_params.items():
            options = options.copy()
            options['params'] = [v for k, v in model.named_parameters() if k.startswith(par_name)]
            parameters.append(options)
        default_options = {
            'params': [v for k, v in model.named_parameters() if all(
                (not k.startswith(par_name) for par_name, options in optimiser_params.items())
            )]}
        parameters.append(default_options)
    optimizer = torch.optim.Adam(parameters, lr=params['train.lr'], weight_decay=params['train.weight_decay'])

    if params.get('train.swa', None):
        optimizer = SWA(optimizer)

    return optimizer


class SchedulerWrapper(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __call__(self, *args, **kwargs):
        self.scheduler.step()

    def step(self, epoch=None):
        self.scheduler.step(epoch)

    @property
    def optimizer(self):
        return self.scheduler.optimizer


class ReduceLROnPlateauWrapper(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __call__(self, metric_value, *args, **kwargs):
        self.scheduler.step(metric_value)

    def step(self, metric, epoch=None):
        self.scheduler.step(metric, epoch)

    @property
    def optimizer(self):
        return self.scheduler.optimizer


class MultiGammaScheduler(torch.optim.lr_scheduler.MultiStepLR):

    def __init__(self, optimizer, milestones, gammas, gamma=0.1, last_epoch=-1):
        self.gammas = gammas
        super(MultiGammaScheduler, self).__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self):
        idx = bisect_right(self.milestones, self.last_epoch)
        gammas = self.gammas[:idx]
        gamma = np.prod(gammas)
        return [base_lr * gamma for base_lr in self.base_lrs]


def get_lr_scheduler(optimizer, params):
    if 'scheduler' in params:
        # TODO: check the this code branch
        if params['scheduler.current'] != '':
            scheduler_type = params['scheduler.current']

            scheduler_params = params[f'scheduler.{scheduler_type}']

            if scheduler_type == 'MultiGammaScheduler':
                scheduler = MultiGammaScheduler(optimizer,
                                         milestones=scheduler_params['milestones'],
                                         gammas=scheduler_params['gammas'],
                                         gamma=scheduler_params['gamma'],
                                         last_epoch=scheduler_params['last_epoch'])

            logger.info('MultiGammaScheduler used')

    elif params['lr_scheduler'].get('CosineAnnealing', False):
        T_max = params['train'].get('n_epoch', params['train.lr_scheduler.n_epoch'])
        eta_min = params['lr_scheduler'].get('eta_min', 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        logger.info('CosineAnnealingLR lr_scheduler used')
        wrapper = SchedulerWrapper

    elif params['lr_scheduler'].get('ReduceLROnPlateau', False):
        mode = params['lr_scheduler'].get('mode', 'max')
        factor = params['lr_scheduler'].get('factor', 0.1)
        patience = params['lr_scheduler'].get('patience', 10)
        threshold = params['lr_scheduler'].get('threshold', 0.001)
        min_lr = params['lr_scheduler'].get('min_lr', 1e-6)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode='rel',
            min_lr=min_lr,
            verbose=True
        )
        logger.info('ReduceLROnPlateau lr_scheduler used')
        wrapper = ReduceLROnPlateauWrapper

    else:
        lr_step_size = params['lr_scheduler']['step_size']
        lr_step_gamma = params['lr_scheduler']['step_gamma']
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_step_gamma)
        logger.info('StepLR lr_scheduler used')
        wrapper = SchedulerWrapper

    # TODO: ReduceLROnPlateau + warmup
    if 'warmup' in params['lr_scheduler']:
        wrapper = LRScheduler
        # optimiser param groups are not supported with LRScheduler
        # create_lr_scheduler_with_warmup don't works with SchedulerWrapper

    scheduler = wrapper(scheduler)

    if 'warmup' in params['lr_scheduler']:
        scheduler = create_lr_scheduler_with_warmup(scheduler, **params['lr_scheduler.warmup'])
        logger.info('LR warmup used')

    return scheduler


class MlflowHandler:
    def __init__(self, logger):
        self.logger = logger

    def __call__(self, train_engine, valid_engine, optimizer):

        def global_state_transform(*args, **kwargs):
            return train_engine.state.iteration

        self.logger.attach(
            train_engine,
            log_handler=ignite.contrib.handlers.mlflow_logger.OutputHandler(
                tag='train',
                metric_names='all'
            ),
            event_name=Events.ITERATION_STARTED
        )

        self.logger.attach(
            valid_engine,
            log_handler=ignite.contrib.handlers.mlflow_logger.OutputHandler(
                tag='validation',
                metric_names='all',
                global_step_transform=global_state_transform
            ),
            event_name=Events.EPOCH_COMPLETED
        )

        self.logger.attach(
            train_engine,
            log_handler=ignite.contrib.handlers.mlflow_logger.OptimizerParamsHandler(optimizer),
            event_name=Events.ITERATION_STARTED
        )


class TensorboardHandler:
    def __init__(self, log_dir):
        self.logger = SummaryWriter(log_dir)

    def __call__(self, train_engine, valid_engine, optimizer):
        @train_engine.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            self.logger.add_scalar('train/loss', engine.state.metrics['loss'], engine.state.iteration)


class PrepareEpoch:
    def __init__(self, train_loader):
        self.train_loader = train_loader

    def __call__(self, *args, **kwargs):
        if hasattr(self.train_loader, 'prepare_epoch'):
            self.train_loader.prepare_epoch()


def output_transform(x, y, y_pred, loss):
    if isinstance(x, PaddedBatch):
        seq_lens_mean = x.seq_lens.float().mean()

    elif isinstance(x, dict) and isinstance(x[next(iter(x.keys()))], PaddedBatch):
        seq_lens_mean = x[next(iter(x.keys()))].seq_lens.float().mean()

    elif (isinstance(x, tuple) or isinstance(x, list)) and isinstance(x[0], PaddedBatch):
        seq_lens_mean = x[0].seq_lens.float().mean()

    else:
        seq_lens_mean = 0

    return loss.item(), seq_lens_mean


def fit_model(model, train_loader, valid_loader, loss, optimizer, scheduler, params, valid_metrics, train_handlers):
    device = torch.device(params.get('device', 'cpu'))
    model.to(device)

    trainer = create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss,
        device=device,
        prepare_batch=batch_to_device,
        output_transform=output_transform,
    )

    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'seq_len')
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

    trainer.add_event_handler(Events.EPOCH_STARTED, PrepareEpoch(train_loader))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        if valid_loader is not None:
            validation_evaluator.run(valid_loader)
            metrics = validation_evaluator.state.metrics
            msgs = []
            for metric in metrics.keys():
                msgs.append(f'{metric}: {metrics[metric]:.3f}')
            pbar.log_message(
                f'Epoch: {engine.state.epoch},  {", ".join(msgs)}')
        else:
            pbar.log_message(
                f'Epoch: {engine.state.epoch},  without validation')


    @validation_evaluator.on(Events.COMPLETED)
    def reduce_step(engine):
        # will be executed every time when validation_evaluator finish run
        # engine is validation_evaluator
        if isinstance(scheduler, ReduceLROnPlateauWrapper):
            metric_value = next(iter(engine.state.metrics.values()))
            scheduler(metric_value)
        else:
            scheduler()

    if params.get('train.use_best_epoch', False):
        best_metric_value = float('-inf')
        best_parameters = {}

        @validation_evaluator.on(Events.COMPLETED)
        def save_best_parameters(engine):
            metric_value = next(iter(engine.state.metrics.values()))
            nonlocal best_metric_value
            nonlocal best_parameters
            if best_metric_value < metric_value:
                best_metric_value = metric_value
                best_parameters = deepcopy(model.state_dict())

    # Stochastic Weight Averaging
    if params.get('train.swa', False):
        @trainer.on(Events.EPOCH_COMPLETED)
        def update_swa(engine):
            if engine.state.epoch >= params['train.swa'].get('swa_start'):
                optimizer.update_swa()

    for handler in train_handlers:
        handler(trainer, validation_evaluator, optimizer)

    trainer.run(train_loader, max_epochs=params['train.n_epoch'])

    if params.get('train.use_best_epoch', False):
        model.load_state_dict(best_parameters)

    elif params.get('train.swa', None):
        optimizer.swap_swa_sgd()
        optimizer.bn_update(train_loader, model, device, prepare_batch=batch_to_device)

    return validation_evaluator.state.metrics


def score_model(model, valid_loader, params):
    if torch.cuda.is_available():
        device = torch.device(params.get('device', 'cuda'))
    else:
        device = torch.device(params.get('device', 'cpu'))
    model.to(device)

    pred = []
    true = []

    def process_valid(_, batch):
        x, y = batch_to_device(batch, device, False)

        model.eval()
        with torch.no_grad():
            outputs = model(x)
            pred.append(outputs.cpu().numpy())
            true.append(y.cpu().numpy())

        return outputs, y

    validation_evaluator = Engine(process_valid)

    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(validation_evaluator)

    validation_evaluator.run(valid_loader)

    true = np.concatenate(true)
    pred = np.concatenate(pred)

    if params['norm_scores']:
        pred = pred / np.max(pred)
        pred = pred - np.min(pred)

    return true, pred


def block_iterator(iterator, size):
    bucket = list()
    for e in iterator:
        bucket.append(e)
        if len(bucket) >= size:
            yield bucket
            bucket = list()
    if bucket:
        yield bucket


def predict_proba_path(model, path_wc, create_loader, files_per_batch=100):
    params = model.params

    from glob import glob
    import sparkpickle

    data_files = [path for path in glob(path_wc)]

    scores = []
    for fl in block_iterator(data_files, files_per_batch):
        score_data = list()

        for path in fl:
            with open(path, 'rb') as f:
                score_data.extend(dict(e) for e in sparkpickle.load(f))

        loader = create_loader(score_data, params)
        if len(loader) == 0:  # no valid samples in block
            continue

        pred = score_model(model, loader, params)
        scores.append(pred)

    return pd.concat(scores)


class CheckpointHandler:
    def __init__(self, model, **model_checkpoint_params):
        self.ignite_version = ignite.__version__

        self.save_interval = model_checkpoint_params.get('save_interval', 1)
        if self.ignite_version >= '0.3.0':
            del model_checkpoint_params['save_interval']
            model_checkpoint_params['global_step_transform'] = lambda engine, event_name: engine.state.epoch
        self.handler = ModelCheckpoint(**model_checkpoint_params)
        self.model = model

    def __call__(self, train_engine, valid_engine, optimizer):
        if self.ignite_version >= '0.3.0':
            train_engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.save_interval),
                                           self.handler, {'model': self.model})
        else:
            train_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                           self.handler, {'model': self.model})
