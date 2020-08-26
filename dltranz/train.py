import logging
import warnings

import torch
from ignite.contrib.handlers import ProgressBar, LRScheduler, create_lr_scheduler_with_warmup
from ignite.contrib.handlers.param_scheduler import ParamScheduler
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage
import numpy as np
import pandas as pd
from math import sqrt

warnings.filterwarnings('ignore', module='tensorboard.compat.tensorflow_stub.dtypes')
from torch.utils.tensorboard import SummaryWriter

from dltranz.seq_encoder import PaddedBatch

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
import ignite
from bisect import bisect_right

logger = logging.getLogger(__name__)


def batch_to_device(batch, device, non_blocking):
    x, y = batch
    y = y.type(torch.FloatTensor)
    #print('shape',y.shape)
    #print('y',torch.sum(y[:,1:], axis=1) )
    #quit()
    if not isinstance(x, dict):
        new_x = {k: v.to(device=device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in x.payload.items()}
        new_y = y.to(device=device, non_blocking=non_blocking)
        return PaddedBatch(new_x, x.seq_lens), new_y
    else:
        batches = {}
        for key, sx in x.items():
            new_x = {k: v.to(device=device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in
                     sx.payload.items()}
            batches[key] = PaddedBatch(new_x, sx.seq_lens)
        new_y = y.to(device=device, non_blocking=non_blocking)
        return batches, new_y


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
    return optimizer


class SchedulerWrapper:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __call__(self, *args, **kwargs):
        self.scheduler.step()


class ReduceLROnPlateauWrapper:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __call__(self, metric_value, *args, **kwargs):
        self.scheduler.step(metric_value)


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
        T_max = params['train']['n_epoch']
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

#custom class
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
class SpendPredictMetric(ignite.metrics.Metric):

    def __init__(self, ignored_class=None, output_transform=lambda x: x):
        self._relative_error = None
        super(SpendPredictMetric, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._relative_error = []
        super(SpendPredictMetric, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        delta = torch.abs(y_pred[:,0] - y[:,0])
        rel_delta = 100*delta / torch.max(y[:,0], torch.exp(y[:,0]-y[:,0]) )
        self._relative_error += [torch.mean(rel_delta).item()]
        
    @sync_all_reduce("_relative_error")
    def compute(self):
        if self._relative_error == 0:
           raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return sum(self._relative_error)/len(self._relative_error)

#custom class
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
class PercentPredictMetric(ignite.metrics.Metric):

    def __init__(self, ignored_class=None, output_transform=lambda x: x):
        self._relative_error = None
        self.softmax = torch.nn.Softmax(dim=1)
        super(PercentPredictMetric, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._relative_error = []
        super(PercentPredictMetric, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        soft_pred = self.softmax(y_pred[:,1:53])
        #print('---------------------')
        #print(y)
        #print(soft_pred)
        delta = torch.norm(soft_pred - y[:,1:53], dim=1)
        #print(delta)
        #print(delta.shape)
        #print(soft_pred.shape)
        rel_delta = 100*torch.mean(delta)/sqrt(2) 
        self._relative_error += [rel_delta.item()]
        
    @sync_all_reduce("_relative_error")
    def compute(self):
        if self._relative_error == 0:
           raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return sum(self._relative_error)/len(self._relative_error)

#custom class
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
class MeanSumPredictMetric(ignite.metrics.Metric):

    def __init__(self, ignored_class=None, output_transform=lambda x: x):
        self._relative_error = None
        self.apriori_mean_list = [104.50246442561681, 24.13884404918004, 23.987986130468652, 25.99619133450599, 31.85280996717411, 48.45940328726219, 55.47242164432059, 30.274800540801422, 78.54245364057059, 33.584090641015216, 188.10486081552096, 36.28945151578021, 81.29908838191119, 13.36464317351852, 84.84840735088136, 151.9366333535613, 74.19875206712715, 33.440992543674014, 264.2680258351682, 37.409850273598, 202.8606972950071, 191.2214637202601, 40.240267318700866, 88.53218639904627, 126.69188207321002, 98.40358576861554, 133.64843240208026, 195.70574982471038, 190.9354436258683, 229.74390047798462, 167.13503214421857, 263.6275381853625, 61.9833114297043, 829.9443242145692, 70.43075570091183, 26.589740581794942, 120.50270399815835, 158.77900298555343, 42.00745367058008, 87.87754873721126, 88.79054102150519, 19.639214062447582, 517.5821513304905, 146.5522789130751, 149.49401859950868, 113.19210670119924, 61.46378188880782, 31.345534958521874, 78.6743993311771, 323.44568014813973, 346.0130976935031, 42.04684542519712]
        self.apriori_mean_list = np.array(self.apriori_mean_list)
        self.softmax = torch.nn.Softmax(dim=1)
        super(MeanSumPredictMetric, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._relative_error = []
        super(MeanSumPredictMetric, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        soft_pred = self.softmax(y_pred[:,1:53])
        #print('-----------------------')
        #print(soft_pred)
        rss = (torch.exp(y_pred[:,53:]) - torch.exp(y[:,53:]) )**2
        rss = soft_pred*rss # torch.max(y[:,53:], torch.exp(y[:,53:]-y[:,53:]) )i
        rss = rss.sum(axis=1)
        mean_apriori = np.tile(self.apriori_mean_list,(y_pred.shape[0],1))
        mean_apriori = torch.FloatTensor(mean_apriori)
        if y_pred.is_cuda:
            mean_apriori = mean_apriori.to(y_pred.get_device())
        tss = (torch.exp(y_pred[:,53:]) - mean_apriori)**2
        tss = soft_pred*tss
        tss = tss.sum(axis=1)
        #print(tss)
        #print(rss)
        r2 = 1 - rss/tss
        #print(r2)
        #quit()
        self._relative_error += [torch.mean(r2).item()]
        
    @sync_all_reduce("_relative_error")
    def compute(self):
        if self._relative_error == 0:
           raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return sum(self._relative_error)/len(self._relative_error)

def fit_model(model, train_loader, valid_loader, loss, optimizer, scheduler, params, valid_metrics, train_handlers):
        #quit()
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
        #metrics=valid_metrics 
        metrics={'total_number':SpendPredictMetric(), 'type_transac':PercentPredictMetric(), 'mean_rur':MeanSumPredictMetric()}
    )

    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(validation_evaluator)

    # valid_metric_name = valid_metric.__class__.__name__
    # valid_metric.attach(validation_evaluator, valid_metric_name)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, PrepareEpoch(train_loader))

    # @trainer.on(Events.GET_BATCH_COMPLETED)
    #def log_training_first_iterations(trainer):
    #    print(trainer.state.batch)
    #    x,y = batch_to_device(trainer.state.batch, device, True)
    #    y_output = model(x)
    #    print(y_output)
    #    print("------------------------------")
    #    quit()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        validation_evaluator.run(valid_loader)
        metrics = validation_evaluator.state.metrics
        msgs = []
        for metric in ['total_number', 'type_transac', 'mean_rur']: #valid_metrics:
            msgs.append(f'{metric}: {metrics[metric]:.3f}')
        pbar.log_message(
            f'Epoch: {engine.state.epoch},  {", ".join(msgs)}')

    @validation_evaluator.on(Events.COMPLETED)
    def reduce_step(engine):
        # will be executed every time when validation_evaluator finish run
        # engine is validation_evaluator
        if isinstance(scheduler, ReduceLROnPlateauWrapper):
            metric_value = next(iter(engine.state.metrics.values()))
            scheduler(metric_value)
        else:
            scheduler()

    for handler in train_handlers:
        handler(trainer, validation_evaluator, optimizer)

    trainer.run(train_loader, max_epochs=params['train.n_epoch'])

    return validation_evaluator.state.metrics


def score_model(model, valid_loader, params):
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
