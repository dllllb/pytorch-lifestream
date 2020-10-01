import logging
import os
import numpy as np
import torch

from functools import partial
from torch.utils.data import DataLoader

from dltranz.data_load import TrxDataset, ConvertingTrxDataset, DropoutTrxDataset
from dltranz.baselines.rtd import collate_rtd_batch
from dltranz.experiment import update_model_stats, get_epoch_score_metric
from dltranz.loss import get_loss
from dltranz.models import model_by_type
from dltranz.seq_encoder import LastStepEncoder, MeanStepEncoder
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model
from dltranz.util import init_logger, get_conf, switch_reproducibility_on
from metric_learning import prepare_data

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # reproducibility
    switch_reproducibility_on()


def create_data_loaders(conf):
    train_data, valid_data = prepare_data(conf)

    collate_fn = partial(
        collate_rtd_batch,
        replace_prob=conf['params.train.replace_token.replace_prob'],
        skip_first=conf['params.train.replace_token.skip_first']
    )

    train_dataset = ConvertingTrxDataset(TrxDataset(train_data, with_target=False), with_target=False)
    train_dataset = DropoutTrxDataset(
        train_dataset,
        conf['params.train.trx_dropout'],
        conf['params.train.max_seq_len'],
        with_target=False
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=conf['params.train'].get('num_workers', 0),
        batch_size=conf['params.train.batch_size'],
    )

    valid_dataset = ConvertingTrxDataset(TrxDataset(valid_data, with_target=False), with_target=False)
    valid_dataset = DropoutTrxDataset(valid_dataset, 0, conf['params.valid.max_seq_len'], with_target=False)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        shuffle=False,
        collate_fn=collate_fn,
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

        if 'rnn' in conf['params']:
            m_encoder = torch.nn.Sequential(*model[:-1], LastStepEncoder())
        elif 'transf' in conf['params']:
            m_encoder = torch.nn.Sequential(*model[:-1], MeanStepEncoder())

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

    model_f = model_by_type(conf['params.model_type'])
    model = model_f(conf['params'])

    return run_experiment(model, conf)


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('dataset_preparation')

    main()
