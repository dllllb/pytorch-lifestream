import logging
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from dltranz.data_load import ConvertingTrxDataset, DropoutTrxDataset
from dltranz.experiment import update_model_stats
from dltranz.metric_learn.dataset import SplittingDataset, split_strategy
from dltranz.metric_learn.dataset import TargetEnumeratorDataset, collate_splitted_rows
from dltranz.metric_learn.losses import get_loss
from dltranz.metric_learn.metric import BatchRecallTop
from dltranz.metric_learn.ml_models import rnn_model
from dltranz.metric_learn.sampling_strategies import get_sampling_strategy
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model
from dltranz.util import init_logger, get_conf

logger = logging.getLogger(__name__)


def prepare_embeddings(seq, conf):
    embeddings = list(conf['params.trx_encoder.embeddings'].keys())

    for rec in seq:
        feature_arrays = rec['feature_arrays']
        feature_arrays = {k: v + (1 if k in embeddings else 0) for k, v in feature_arrays.items()}

        for e_name, e_params in conf['params.trx_encoder.embeddings'].items():
            feature_arrays[e_name] = feature_arrays[e_name].clip(0, e_params['in'] - 1)

        rec['feature_arrays'] = feature_arrays
        yield rec


def create_data_loaders(conf):
    with open(conf['dataset.path'], 'rb') as f:
        data = pickle.load(f)
    data = list(prepare_embeddings(data, conf))

    valid_ix = np.arange(len(data))
    valid_ix = np.random.choice(valid_ix, size=int(len(data) * conf['dataset.valid_size']), replace=False)

    train_data = [rec for i, rec in enumerate(data) if i not in valid_ix]
    valid_data = [rec for i, rec in enumerate(data) if i in valid_ix]

    logger.info(f'Train data len: {len(train_data)}, Valid data len: {len(valid_data)}')

    train_dataset = SplittingDataset(
        train_data,
        split_strategy.create(**conf['params.train.split_strategy'])
    )
    train_dataset = TargetEnumeratorDataset(train_dataset)
    train_dataset = ConvertingTrxDataset(train_dataset)
    train_dataset = DropoutTrxDataset(train_dataset, trx_dropout=conf['params.train.trx_dropout'],
                                      seq_len=conf['params.train.max_seq_len'])
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        collate_fn=collate_splitted_rows,
        num_workers=conf['params.train'].get('num_workers', 0),
        batch_size=conf['params.train.batch_size'],
    )

    valid_dataset = SplittingDataset(
        valid_data,
        split_strategy.create(**conf['params.valid.split_strategy'])
    )
    valid_dataset = TargetEnumeratorDataset(valid_dataset)
    valid_dataset = ConvertingTrxDataset(valid_dataset)
    valid_dataset = DropoutTrxDataset(valid_dataset, trx_dropout=0.0,
                                      seq_len=conf['params.valid.max_seq_len'])
    valid_loader = DataLoader(
        dataset=valid_dataset,
        shuffle=False,
        collate_fn=collate_splitted_rows,
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

    sampling_strategy = get_sampling_strategy(params)
    loss = get_loss(params, sampling_strategy)

    valid_metric = {'BatchRecallTop': BatchRecallTop(k=params['valid.split_strategy.split_count'] - 1)}
    optimizer = get_optimizer(model, params)
    scheduler = get_lr_scheduler(optimizer, params)

    metric_values = fit_model(model, train_loader, valid_loader, loss, optimizer, scheduler, params, valid_metric,
                              train_handlers=[])

    exec_sec = time.time() - start

    if conf.get('save_model', False):
        torch.save(model, conf['model_path.model'])
        logger.info(f'Model saved to "{conf["model_path.model"]}"')

    results = {
        'exec-sec': exec_sec,
        'Recall_top_K': metric_values,
    }

    if conf.get('log_results', True):
        update_model_stats(stats_file, params, results)


def main(args=None):
    conf = get_conf(args)

    model = rnn_model(conf['params'])

    return run_experiment(model, conf)


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('dataset_preparation')

    main(None)
