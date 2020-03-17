import logging
import os

import numpy as np
import torch

from dltranz.experiment import update_model_stats
from dltranz.metric_learn.losses import get_loss
from dltranz.metric_learn.metric import BatchRecallTop
from dltranz.metric_learn.ml_models import ml_model_by_type
from dltranz.metric_learn.sampling_strategies import get_sampling_strategy
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model
from dltranz.util import init_logger, get_conf

from dltranz.cpc import CPCShellV2
from metric_learning import create_data_loaders

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # reproducibility
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def run_experiment(model, conf):
    import time
    start = time.time()

    stats_file = conf['stats.path']
    params = conf['params']

    train_loader, valid_loader = create_data_loaders(conf)

    loss = get_loss(params, sampling_strategy=None, kw_params={'linear_predictor': model.linear_predictor})

    valid_metric = {'BatchRecallTop': BatchRecallTop(k=params['valid.split_strategy.split_count'] - 1)}
    optimizer = get_optimizer(model, params)
    scheduler = get_lr_scheduler(optimizer, params)

    metric_values = fit_model(model, train_loader, valid_loader, loss, optimizer, scheduler, params, valid_metric,
                              train_handlers=[])

    exec_sec = time.time() - start

    if conf.get('save_model', False):
        save_dir = os.path.dirname(conf['model_path.model'])
        os.makedirs(save_dir, exist_ok=True)

        torch.save(model.encoder, conf['model_path.model'])
        logger.info(f'Model saved to "{conf["model_path.model"]}"')

    results = {
        'exec-sec': exec_sec,
        'Recall_top_K': metric_values,
    }

    if conf.get('log_results', True):
        update_model_stats(stats_file, params, results)


def main(args=None):
    conf = get_conf(args)

    model_f = ml_model_by_type(conf['params.model_type'])
    model = model_f(conf['params'])

    model = CPCShellV2(model, conf['params.cpc.embedding_size'], conf['params.cpc.k_pos_samples'])

    return run_experiment(model, conf)


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('dataset_preparation')
    init_logger('metric_learning')

    main()
