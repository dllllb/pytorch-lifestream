import logging
import os
from itertools import islice

import numpy as np
import torch
from torch.utils.data import DataLoader

from metric_learning import prepare_embeddings, shuffle_client_list_reproducible
from dltranz.data_load import ConvertingTrxDataset, DropoutTrxDataset, read_data_gen, AllTimeShuffleMLDataset
from dltranz.experiment import update_model_stats
from dltranz.metric_learn.dataset import SeveralSplittingsDataset, split_strategy
from dltranz.metric_learn.dataset import ComplexTargetDataset, collate_splitted_rows
from dltranz.metric_learn.losses import get_loss, ComplexLoss
from dltranz.metric_learn.metric import metric_Recall_top_K
from dltranz.metric_learn.ml_models import ml_model_by_type, ComplexModel
from dltranz.metric_learn.sampling_strategies import get_sampling_strategy
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model, CheckpointHandler
from dltranz.util import init_logger, get_conf, CustomMetric, switch_reproducibility_on

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    switch_reproducibility_on()


def create_data_loaders(conf):
    data = read_data_gen(conf['dataset.train_path'])
    if 'max_rows' in conf['dataset']:
        data = islice(data, conf['dataset.max_rows'])
    data = prepare_embeddings(data, conf, is_train=True)
    data = shuffle_client_list_reproducible(conf, data)
    data = list(data)
    if 'client_list_keep_count' in conf['dataset']:
        data = data[:conf['dataset.client_list_keep_count']]

    valid_ix = np.arange(len(data))
    valid_ix = np.random.choice(valid_ix, size=int(len(data) * conf['dataset.valid_size']), replace=False)

    train_data = [rec for i, rec in enumerate(data) if i not in valid_ix]
    valid_data = [rec for i, rec in enumerate(data) if i in valid_ix]

    logger.info(f'Train data len: {len(train_data)}, Valid data len: {len(valid_data)}')

    split_strategies = [
        split_strategy.create(**split_strategy_params) for split_strategy_params in
        conf['params.train.split_strategies']
    ]
    split_counts = [
        split_strategy_params['split_count'] for split_strategy_params in conf['params.train.split_strategies']
    ]
    train_dataset = SeveralSplittingsDataset(
        train_data,
        split_strategies
    )
    train_dataset = ComplexTargetDataset(train_dataset, split_counts)

    train_dataset = ConvertingTrxDataset(train_dataset)

    train_dataset = DropoutTrxDataset(train_dataset, trx_dropout=conf['params.train.trx_dropout'],
                                      seq_len=conf['params.train.max_seq_len'])

    if conf['params.train'].get('all_time_shuffle', False):
        train_dataset = AllTimeShuffleMLDataset(train_dataset)
        logger.info('AllTimeShuffle used')

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        collate_fn=collate_splitted_rows,
        num_workers=conf['params.train'].get('num_workers', 0),
        batch_size=conf['params.train.batch_size'],
    )

    split_strategies = [
        split_strategy.create(**split_strategy_params) for split_strategy_params in
        conf['params.valid.split_strategies']
    ]
    split_counts = [
        split_strategy_params['split_count'] for split_strategy_params in conf['params.valid.split_strategies']
    ]
    valid_dataset = SeveralSplittingsDataset(
        valid_data,
        split_strategies
    )
    valid_dataset = ComplexTargetDataset(valid_dataset, split_counts)
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


def run_experiment(model, conf, train_loader, valid_loader):
    import time
    start = time.time()

    stats_file = conf['stats.path']
    params = conf['params']

    sampling_strategy = get_sampling_strategy(params)
    ml_loss = get_loss(params, sampling_strategy)
    aug_loss = torch.nn.NLLLoss()
    loss = ComplexLoss(ml_loss, aug_loss, params['train'].get('ml_loss_weight', 1.0))

    split_counts = [
        split_strategy_params['split_count'] for split_strategy_params in conf['params.valid.split_strategies']
    ]
    valid_metric = {
        'Accuracy': CustomMetric(func=lambda x, y: (torch.argmax(x[0], 1) == y[:, 0]).float().mean()),
        'BatchRecallTop': CustomMetric(func=lambda x, y:
        metric_Recall_top_K(x[1], y[:, 1], sum(split_counts) - 1, 'cosine'))
    }
    optimizer = get_optimizer(model, params)
    scheduler = get_lr_scheduler(optimizer, params)

    train_handlers = []
    if 'checkpoints' in conf['params.train']:
        checkpoint = CheckpointHandler(
            model=model,
            **conf['params.train.checkpoints']
        )
        train_handlers.append(checkpoint)

    metric_values = fit_model(model, train_loader, valid_loader, loss, optimizer, scheduler, params, valid_metric,
                              train_handlers=train_handlers)

    exec_sec = time.time() - start

    if conf.get('save_model', False):
        save_dir = os.path.dirname(conf['model_path.model'])
        os.makedirs(save_dir, exist_ok=True)

        ml_model = model.ml_model
        m_encoder = ml_model[0] if conf['model_path.only_encoder'] else ml_model

        torch.save(m_encoder, conf['model_path.model'])
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
    ml_model = model_f(conf['params'])
    model = ComplexModel(ml_model, conf['params'])
    train_loader, valid_loader = create_data_loaders(conf)

    return run_experiment(model, conf, train_loader, valid_loader)


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('dataset_preparation')

    main()
