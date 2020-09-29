import logging
import os
import numpy as np
import torch

from torch.utils.data import DataLoader

from dltranz.data_load import ConvertingTrxDataset
from dltranz.experiment import update_model_stats, get_epoch_score_metric
from dltranz.loss import get_loss
from dltranz.metric_learn.dataset import SplittingDataset, split_strategy
from dltranz.metric_learn.ml_models import ml_model_by_type
from dltranz.baselines.sop import SentencePairsHead, collate_sop_pairs
from dltranz.baselines.nsp import SequencePairsDataset
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model
from dltranz.util import init_logger, get_conf
from metric_learning import prepare_data

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # reproducibility
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def create_data_loaders(conf):
    train_data, valid_data = prepare_data(conf)

    train_dataset = SplittingDataset(
        train_data,
        split_strategy.create(**conf['params.train.split_strategy'])
    )
    train_dataset = ConvertingTrxDataset(train_dataset, with_target=False)
    train_dataset = SequencePairsDataset(train_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        collate_fn=collate_sop_pairs,
        num_workers=conf['params.train'].get('num_workers', 0),
        batch_size=conf['params.train.batch_size'],
    )

    valid_dataset = SplittingDataset(
        valid_data,
        split_strategy.create(**conf['params.valid.split_strategy'])
    )
    valid_dataset = ConvertingTrxDataset(valid_dataset, with_target=False)
    valid_dataset = SequencePairsDataset(valid_dataset)

    valid_loader = DataLoader(
        dataset=valid_dataset,
        shuffle=False,
        collate_fn=collate_sop_pairs,
        num_workers=conf['params.valid'].get('num_workers', 0),
        batch_size=conf['params.valid.batch_size'],
    )

    return train_loader, valid_loader


def run_experiment(model, conf):
    import time
    start = time.time()

    stats_file = conf['stats.path']
    params = conf['params']

    train_loader, valid_loader = create_data_loaders(conf)
    loss = get_loss(params)

    metric_name = params['score_metric']
    valid_metrics = {metric_name: get_epoch_score_metric(metric_name)()}

    optimizer = get_optimizer(model, params)
    scheduler = get_lr_scheduler(optimizer, params)

    train_handlers = []
    metric_values = fit_model(model, train_loader, valid_loader, loss, optimizer, scheduler, params, valid_metrics,
                              train_handlers=train_handlers)

    exec_sec = time.time() - start

    if conf.get('save_model', False):
        save_dir = os.path.dirname(conf['model_path.model'])
        os.makedirs(save_dir, exist_ok=True)

        m_encoder = model.base_model[0] if conf['model_path.only_encoder'] else model.base_model

        torch.save(m_encoder, conf['model_path.model'])
        logger.info(f'Model saved to "{conf["model_path.model"]}"')

    results = {
        'exec-sec': exec_sec,
        metric_name: metric_values,
    }

    if conf.get('log_results', True):
        update_model_stats(stats_file, params, results)


def main(args=None):
    conf = get_conf(args)

    model_f = ml_model_by_type(conf['params.model_type'])
    base_model = model_f(conf['params'])

    if 'rnn' in conf['params']:
        embeddings_size = conf['params.rnn.hidden_size']
    elif 'transf' in conf['params']:
        embeddings_size = conf['params.transf.input_size']
    else:
        raise AttributeError

    model = SentencePairsHead(base_model, embeddings_size, conf['params.head'])

    return run_experiment(model, conf)


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('dataset_preparation')

    main()
