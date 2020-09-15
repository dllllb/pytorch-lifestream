import datetime
import logging
import pickle
import sys
import random
from copy import deepcopy

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from dltranz.data_load import TrxDataset, ConvertingTrxDataset, DropoutTrxDataset, padded_collate, \
    create_validation_loader, read_data_gen
from dltranz.loss import get_loss
from dltranz.models import model_by_type
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model
from dltranz.util import init_logger, get_conf
from dltranz.experiment import get_epoch_score_metric, update_model_stats
from dltranz.metric_learn.inference_tools import infer_part_of_data, save_scores
from metric_learning import prepare_embeddings

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


def read_consumer_data(path, conf):
    logger.info(f'Data loading...')

    data = read_data_gen(path)
    #for rec in data:
     #print(rec['target'])
     #print(type(rec['target']))
     #print(rec['target'][0])
     #quit()
    
    #data = (rec for rec in data if rec['target'] is not None and not np.isnan(rec['target']))
    data = (rec for rec in data if rec['target'] is not None )

    data = prepare_embeddings(data, conf, is_train=False)
    data = list(data)

    logger.info(f'Loaded data with target: {len(data)}')


    return data


def create_ds(train_data, valid_data, conf):
    if 'SubsamplingDataset' in conf['params.train']:
        train_data = SubsamplingDataset(train_data, **conf['params.train.SubsamplingDataset'])

    train_ds = ConvertingTrxDataset(TrxDataset(train_data, y_dtype=np.float32))
    valid_ds = ConvertingTrxDataset(TrxDataset(valid_data, y_dtype=np.float32))

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


def prepare_parser(parser):
    pass


def main(_):
    init_logger(__name__)
    init_logger('dltranz')
    init_logger('metric_learning')

    conf = get_conf(sys.argv[2:])

    model_f = model_by_type(conf['params.model_type'])
    train_data = read_consumer_data(conf['dataset.train_path'], conf)
    test_data = read_consumer_data(conf['dataset.test_path'], conf)

    # train
    results = []

    skf = StratifiedKFold(conf['cv_n_split'])
    nrows = conf['params'].get('labeled_amount',-1) # semi-supervised setup. default = supervised

    target_values = [rec['target'] for rec in train_data]
    for i, (i_train, i_valid) in enumerate(skf.split(train_data, target_values)):
        logger.info(f'Train fold: {i}')
        i_train_data = [rec for i, rec in enumerate(train_data) if i in i_train]
        i_valid_data = [rec for i, rec in enumerate(train_data) if i in i_valid]

        if nrows > 0: i_train_data = i_train_data[:nrows]

        train_ds, valid_ds = create_ds(i_train_data, i_valid_data, conf)
        model, result = run_experiment(train_ds, valid_ds, conf['params'], model_f)

        # inference
        columns = conf['output.columns']
        train_scores = infer_part_of_data(i, i_valid_data, columns, model, conf)
        save_scores(train_scores, i, conf['output.valid'])

        test_scores = infer_part_of_data(i, test_data, columns, model, conf)
        save_scores(test_scores, i, conf['output.test'])

        results.append(result)

    # results
    stats_file = conf.get('stats.path', None)
    if stats_file is not None:
        update_model_stats(stats_file, conf, results)
