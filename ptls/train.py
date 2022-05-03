import logging
from bisect import bisect_right

import numpy as np
import torch

from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import Adafactor, AdafactorSchedule

from ptls.swa import SWA


logger = logging.getLogger(__name__)


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
    elif optimiser_params.get('type', None) == 'Adafactor':
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=params['train.lr'])
        logger.info("Ada factor optimizer is used")
    elif optimiser_params.get('type', None) == 'SWA':
        optimizer = SWA(optimizer)
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
        T_max = params['train'].get('n_epoch', params['lr_scheduler.n_epoch'])
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
        scheduler = get_linear_schedule_with_warmup(optimizer, 
           num_warmup_steps=params['lr_scheduler']["warmup"]['num_warmup_steps'], 
           num_training_steps=params['lr_scheduler']["warmup"]['num_training_steps'])
        logger.info('LR warmup used')
        # optimiser param groups are not supported with LRScheduler
        # create_lr_scheduler_with_warmup don't works with SchedulerWrapper
    else:
        scheduler = wrapper(scheduler)

    return scheduler


def score_model(model, valid_loader, params=None):
    """
      - extended valid_loader. input format: x, * in batch:
      - output: pred(x), * in score_model

    Returns:

    """
    if params is None:
        params = {}

    if torch.cuda.is_available():
        device = torch.device(params.get('device', 'cuda'))
    else:
        device = torch.device(params.get('device', 'cpu'))
    model.to(device)
    model.eval()

    outputs = []
    with torch.no_grad():
        for batch in tqdm(valid_loader, leave=False):
            x, *others = batch
            x = x.to(device)
            out = model(x)

            batch_output = [out.cpu().numpy(), *others]
            outputs.append(batch_output)

    outputs = zip(*outputs)
    outputs = (np.concatenate(l) for l in outputs)
    return outputs
