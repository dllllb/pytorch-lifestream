import datetime
import logging

import numpy as np
from time import strftime

import torch
from ignite.contrib.metrics import ROC_AUC
from ignite.metrics import Accuracy, Metric
from sklearn.metrics import roc_auc_score

from dltranz.ensemble import ModelEnsemble
from dltranz.data_load import create_train_loader, create_validation_loader, ZeroDownSampler, \
    LastKTrxDataset
from dltranz.loss import get_loss
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model, score_model, TensorboardHandler


logger = logging.getLogger(__name__)


def update_model_stats(stats_file, params, results):
    import json
    import os.path

    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = []

    stats.append({'results': results, 'params': params})

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)


def get_epoch_score_metric(metric_name):
    m = {
        'auroc': ROC_AUC,
        'accuracy': Accuracy,
        'accuracy_labeled' : Accuracy_labeled
    }.get(metric_name)
    if m is not None:
        return m
    else:
        raise AttributeError(f'unknown metric "{metric_name}')


def get_score_metric(metric_name):
    m = {'auroc': roc_auc_score}.get(metric_name)
    if m is not None:
        return m
    else:
        raise AttributeError(f'unknown metric "{metric_name}')


def run_experiment(train_ds, valid_ds, conf, model):
    import time
    start = time.time()

    handlers = []
    log_path = conf.get('tensorboard.log.path', None)
    if log_path is not None:
        handlers.append(TensorboardHandler(f'{log_path}/{strftime("%Y-%m-%d+%H-%M")}'))

    params = conf['params']

    if params['ensemble_size'] > 1:
        if params['ensemble_last_k']:
            m, scores = fit_model_ensemble_last_k(model, train_ds, valid_ds, params, handlers)
        else:
            m, scores = fit_model_ensemble(model, train_ds, valid_ds, params, handlers)
    else:
        m, scores = fit_model_on_data(model, train_ds, valid_ds, params, handlers)

    exec_sec = time.time() - start

    if conf.get('save_model', False):
        torch.save(m, conf['model_path.model'])
        logger.info(f'Model saved to "{conf["model_path.model"]}"')
        if 'state_dict' in conf['model_path']:
            torch.save(m.state_dict(), conf['model_path.state_dict'])
            logger.info(f'State_dict saved to "{conf["model_path.state_dict"]}"')

    results = {
        **scores,
        'exec-sec': exec_sec,
        'finish_time': datetime.datetime.now().isoformat(),
    }
    if conf.get('log_ds_stat', True):
        train_targets = [y for x, y in train_ds]
        results['n_ones'] = int(sum(train_targets))
        results['n_samples'] = len(train_ds)

    if conf.get('log_model_info', False):
        results['model_info'] = str(m)

    stats_file = conf.get('stats.path', None)

    if stats_file is not None:
        update_model_stats(stats_file, params, results)
    else:
        return results


def fit_model_on_data(model, train_ds, valid_ds, params, handlers):
    m = model(params)

    if params['train']['random_neg']:
        targets = [y for x, y in train_ds]
        sampler = ZeroDownSampler(targets)
    else:
        sampler = None

    train_loader = create_train_loader(train_ds, params['train'], sampler)
    valid_loader = create_validation_loader(valid_ds, params['valid'])

    loss = get_loss(params)
    opt = get_optimizer(m, params)
    scheduler = get_lr_scheduler(opt, params)

    metric_name = params['score_metric']
    metric = get_epoch_score_metric(metric_name)()

    return m, fit_model(m, train_loader, valid_loader, loss, opt, scheduler, params, {metric_name: metric}, handlers)


def fit_model_ensemble(model, train_ds, valid_ds, params, handlers):
    ensemble_size = params['ensemble_size']

    preds = []
    true = None
    submodels = []
    for model_no in range(ensemble_size):
        model_n, score_n = fit_model_on_data(model, train_ds, valid_ds, params, handlers)

        valid_loader = create_validation_loader(valid_ds, params['valid'])
        true_n, pred_n = score_model(model_n, valid_loader, params)

        preds.append(pred_n)
        true = true_n
        submodels.append(model_n)

    pred = np.array(preds).mean(axis=0)

    score = get_score_metric(params['score_metric'])(true, pred)

    return ModelEnsemble(submodels=submodels), score


def fit_model_ensemble_last_k(model, train_ds, valid_ds, params, handlers):
    ensemble_size = params['ensemble_size']

    preds = []
    true = None
    submodels = []
    shares = [2**-i for i in range(ensemble_size)]
    for model_no, share in enumerate(shares):
        train_ds = LastKTrxDataset(train_ds, share)
        valid_ds = LastKTrxDataset(valid_ds, share)
        model_n, score_n = fit_model_on_data(model, train_ds, valid_ds, params, handlers)

        valid_loader = create_validation_loader(valid_ds, params['valid'])
        true_n, pred_n = score_model(model_n, valid_loader, params)

        preds.append(pred_n)
        true = true_n
        submodels.append(model_n)

    pred = np.array(preds).mean(axis=0)

    score = get_score_metric(params['score_metric'])(true, pred)

    return ModelEnsemble(submodels=submodels), score


class CustomMetric(Metric):
    def __init__(self, func):
        super().__init__(output_transform=lambda x: x)
        self.func = func
        self.num_value = 0.0
        self.denum_value = 0

    def reset(self):
        self.num_value = 0.0
        self.denum_value = 0

        super().reset()

    def update(self, output):
        x, y = output
        value = self.func(x, y)

        self.num_value += value
        self.denum_value += 1

    def compute(self):
        if self.denum_value == 0:
            return 0.0
        return self.num_value / self.denum_value

class Accuracy_labeled(CustomMetric):
    def __init__(self):
        super().__init__(func = lambda x,y: (torch.argmax(x['labeled'],1) == y).float().mean())