import datetime
import json
import logging
import math
import random
import sys
from copy import deepcopy
from itertools import islice

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from dltranz.data_load import TrxDataset, ConvertingTrxDataset, DropoutTrxDataset, padded_collate, \
    create_validation_loader, read_data_gen, SameTimeShuffleDataset, AllTimeShuffleDataset, DropDayDataset
from dltranz.loss import get_loss
from dltranz.models import model_by_type
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model
from dltranz.util import init_logger, get_conf
from dltranz.experiment import get_epoch_score_metric, update_model_stats
from dltranz.metric_learn.inference_tools import infer_part_of_data, save_scores, score_data
from metric_learning import prepare_embeddings, shuffle_client_list_reproducible

logger = logging.getLogger(__name__)


def prepare_target_gender(data):
    mapping = {
        'F': 0,
        'M': 1,
    }

    for rec in data:
        rec['target'] = mapping.get(rec['target_gender'])
        yield rec


def prepare_target_age(data):
    for rec in data:
        age = int(rec['target_age'])
        if age < 10 or age > 90:
            rec['target'] = None
        elif age < 35:
            rec['target'] = 0
        elif age < 45:
            rec['target'] = 1
        elif age < 60:
            rec['target'] = 2
        else:
            rec['target'] = 3

        yield rec


def read_consumer_data(path, conf):
    logger.info(f'Data loading...')

    col_target_name = conf['col_target_name']

    data = read_data_gen(path)

    if 'max_rows' in conf['dataset']:
        data = islice(data, conf['dataset.max_rows'])

    if col_target_name == 'target_gender':
        data = prepare_target_gender(data)
    elif col_target_name == 'target_age':
        data = prepare_target_age(data)
    else:
        raise AttributeError(f'Unknown col_target_name: {col_target_name}')

    data = (rec for rec in data if rec['target'] is not None and not np.isnan(rec['target']))
    data = prepare_embeddings(data, conf, is_train=False)
    data = shuffle_client_list_reproducible(conf, data)
    data = list(data)

    logger.info(f'Loaded data with target: {len(data)}')

    return data


def create_ds(train_data, valid_data, conf):
    train_ds = ConvertingTrxDataset(TrxDataset(train_data, y_dtype=np.int64))
    valid_ds = ConvertingTrxDataset(TrxDataset(valid_data, y_dtype=np.int64))

    return train_ds, valid_ds


class ClippingDataset(Dataset):
    def __init__(self, delegate, min_len=250, max_len=350, rate_for_min=0.9):
        super().__init__()

        self.delegate = delegate
        self.min_len = min_len
        self.max_len = max_len
        self.rate_for_min = rate_for_min
        self.style = delegate.style

    def __len__(self):
        return len(self.delegate)

    def __getitem__(self, item):
        x, y = self.delegate[item]

        seq_len = len(x['event_time'])
        if seq_len <= 5:
            return x, y

        new_len = random.randint(self.min_len, self.max_len)
        if new_len > seq_len * self.rate_for_min:
            new_len = math.ceil(seq_len * self.rate_for_min)

        avail_pos = seq_len - new_len
        pos = random.randint(0, avail_pos)

        x = deepcopy(x)
        x = {k: v[pos:pos+new_len] for k, v in x.items()}
        return x, y


def run_experiment(train_ds, valid_ds, params, model_f):
    model = model_f(params)

    train_ds = DropoutTrxDataset(train_ds, params['train.trx_dropout'], params['train.max_seq_len'])
    if 'DropDayDataset' in params['train']:
        train_ds = DropDayDataset(train_ds)
        logger.info('DropDayDataset used')
    if 'ClippingDataset' in params['train']:
        train_ds = ClippingDataset(train_ds, **params['train.ClippingDataset'])
        logger.info('ClippingDataset used')
    if params['train.same_time_shuffle']:
        train_ds = SameTimeShuffleDataset(train_ds)
        logger.info('SameTimeShuffle used')
    if params['train.all_time_shuffle']:
        train_ds = AllTimeShuffleDataset(train_ds)
        logger.info('AllTimeShuffle used')

    train_loader = DataLoader(
        train_ds,
        batch_size=params['train.batch_size'],
        shuffle=True,
        num_workers=params['train.num_workers'],
        collate_fn=padded_collate)

    valid_loader = create_validation_loader(valid_ds, params['valid'])

    loss = get_loss(params)
    opt = get_optimizer(model, params)
    scheduler = get_lr_scheduler(opt, params)

    metric_name = params['score_metric']
    metrics = {metric_name: get_epoch_score_metric(metric_name)()}
    handlers = []

    scores = fit_model(model, train_loader, valid_loader, loss, opt, scheduler, params, metrics, handlers)

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
        i_train = set(i_train.tolist())
        i_valid = set(i_valid.tolist())
        
        i_train_data = [rec for i, rec in enumerate(train_data) if i in i_train]
        i_valid_data = [rec for i, rec in enumerate(train_data) if i in i_valid]

        if nrows > 0: i_train_data = i_train_data[:nrows]

        train_ds, valid_ds = create_ds(i_train_data, i_valid_data, conf)
        model, _ = run_experiment(train_ds, valid_ds, conf['params'], model_f)
        result = {
          "fold_id": i,
          "model_name": conf.get('stats.model_name', None),
          "feature_name": conf.get('stats.feature_name', None),
        }

        # inference
        columns = conf['output.columns']
        train_scores = infer_part_of_data(i, i_valid_data, columns, model, conf)
        save_scores(train_scores, i, conf['output.valid'])
        result['scores_valid'] = score_data(conf, i_valid_data, train_scores)

        test_scores = infer_part_of_data(i, test_data, columns, model, conf)
        save_scores(test_scores, i, conf['output.test'])
        result['scores_test'] = score_data(conf, test_data, test_scores)

        results.append(result)

    # results
    stats_file = conf.get('stats.path', None)
    if stats_file is not None:
        with open(stats_file, 'w') as f:
            json.dump(results, f)
