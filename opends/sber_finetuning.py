import datetime
import logging
import os
import pickle
import sys
import random
from copy import deepcopy

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from dltranz.data_load import TrxDataset, ConvertingTrxDataset, DropoutTrxDataset, padded_collate, \
    create_validation_loader
from dltranz.loss import get_loss
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model
from dltranz.util import init_logger, get_conf
from dltranz.experiment import get_epoch_score_metric, update_model_stats
from dltranz.metric_learn.inference_tools import score_part_of_data

logger = logging.getLogger(__name__)


class SubsamplingDataset(Dataset):
    def __init__(self,
                 dataset,
                 min_seq_len_s, min_seq_len_e,
                 max_seq_len_s, max_seq_len_e,
                 crop_proba_init, crop_proba_gamma,
                 total_n_epoch,
                 ):
        self.dataset = dataset

        self.min_seq_len_s = min_seq_len_s
        self.min_seq_len_e = min_seq_len_e
        self.max_seq_len_s = max_seq_len_s
        self.max_seq_len_e = max_seq_len_e
        self.crop_proba_init = crop_proba_init
        self.crop_proba_gamma = crop_proba_gamma
        self.total_n_epoch = total_n_epoch

        self.n_epoch = -1

    def prepare_epoch(self):
        self.n_epoch += 1

    @property
    def _progress(self):
        return min(self.n_epoch / (self.total_n_epoch - 1), 1.0)

    @property
    def min_seq_len(self):
        return int(self.min_seq_len_s + (self.min_seq_len_e - self.min_seq_len_s) * self._progress ** 2)

    @property
    def max_seq_len(self):
        return int(self.max_seq_len_s + (self.max_seq_len_e - self.max_seq_len_s) * self._progress ** 2)

    @property
    def crop_proba(self):
        return self.crop_proba_init * self.crop_proba_gamma ** self.n_epoch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        rec = self.dataset[item]

        if random.random() >= self.crop_proba:
            return rec

        seq_len = len(rec['event_time'])
        if seq_len <= self.min_seq_len:
            return rec

        r_len = random.randint(self.min_seq_len, min(self.max_seq_len, seq_len))
        r_pos = seq_len - r_len
        if r_pos <= 0:
            return rec

        rec = deepcopy(rec)
        r_pos = random.randint(0, r_pos)

        rec['feature_arrays'] = {k: v[r_pos: r_pos + r_len] for k, v in rec['feature_arrays'].items()}
        rec['event_time'] = rec['event_time'][r_pos: r_pos + r_len]
        return rec


class EpochTrackingDataLoader(DataLoader):
    def prepare_epoch(self):
        self.preparing_dataset.prepare_epoch()


def read_consumer_data(conf):
    logger.info(f'Data loading...')

    with open(os.path.join(conf['data_path'], conf['dataset.train_path']), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(conf['data_path'], conf['dataset.valid_path']), 'rb') as f:
        valid_data = pickle.load(f)

    logger.info(f'Loaded train: {len(train_data)}, valid: {len(valid_data)}')

    return train_data, valid_data


def create_ds(train_data, valid_data, conf):
    if 'SubsamplingDataset' in conf['params.train']:
        train_data = SubsamplingDataset(train_data, **conf['params.train.SubsamplingDataset'])

    train_ds = ConvertingTrxDataset(TrxDataset(train_data, y_dtype=np.int64))
    valid_ds = ConvertingTrxDataset(TrxDataset(valid_data, y_dtype=np.int64))

    return train_ds, valid_ds


def run_experiment(train_ds, valid_ds, params, model_f):
    model = model_f(params)

    train_ds = DropoutTrxDataset(train_ds, params['train.trx_dropout'], params['train.max_seq_len'])
    train_loader = EpochTrackingDataLoader(
        train_ds,
        batch_size=params['train.batch_size'],
        shuffle=True,
        num_workers=params['train.num_workers'],
        collate_fn=padded_collate)

    train_loader.preparing_dataset = train_ds.core_dataset.delegate.data
    valid_loader = create_validation_loader(valid_ds, params['valid'])

    loss = get_loss(params)
    opt = get_optimizer(model, params)
    scheduler = get_lr_scheduler(opt, params)

    metric_name = params['score_metric']
    metric = get_epoch_score_metric(metric_name)()
    handlers = []

    scores = fit_model(model, train_loader, valid_loader, loss, opt, scheduler, params, {metric_name: metric}, handlers)

    return model, {
        **scores,
        'finish_time': datetime.datetime.now().isoformat(),
    }


def load_model(conf):
    pretrained_model_path = conf['pretrained_model_path']

    pre_model = torch.load(pretrained_model_path)

    input_size = conf['rnn.hidden_size']
    head_output_size = conf['head.num_classes']

    model = torch.nn.Sequential(
        pre_model[:-1],
        torch.nn.Linear(input_size, head_output_size),
        torch.nn.LogSoftmax(dim=1),
    )
    return model


def main(args=None):
    conf = get_conf(args)

    model_f = load_model
    train_data, valid_data = read_consumer_data(conf)

    # train
    results = []
    all_data = train_data + valid_data

    skf = StratifiedKFold(5)
    target_values = [rec['target'] for rec in all_data]
    for i, (i_train, i_valid) in enumerate(skf.split(all_data, target_values)):
        logger.info(f'Train fold: {i}')
        i_train_data = [rec for i, rec in enumerate(all_data) if i in i_train]
        i_valid_data = [rec for i, rec in enumerate(all_data) if i in i_valid]

        train_ds, valid_ds = create_ds(i_train_data, i_valid_data, conf)
        model, result = run_experiment(train_ds, valid_ds, conf['params'], model_f)

        # inference
        columns = conf['output.columns']
        score_part_of_data(i, i_valid_data, columns, model, conf)

        results.append(result)

    # results
    stats_file = conf.get('stats.path', None)
    if stats_file is not None:
        update_model_stats(stats_file, conf, results)


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('retail_embeddings_projects.embedding_tools')

    if len(sys.argv) == 1:
        args = [
            'data_path=""',
            '--conf',
            'conf/sber_target_dataset.hocon',
            'conf/sber_targetft_params_train.json',
        ]
    else:
        args = None
    main(args)
