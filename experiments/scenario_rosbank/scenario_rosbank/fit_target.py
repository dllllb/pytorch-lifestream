import datetime
import json
import logging
import math
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
from dltranz.experiment import get_epoch_score_metric
from dltranz.metric_learn.inference_tools import infer_part_of_data, save_scores, score_data
from metric_learning import prepare_embeddings

logger = logging.getLogger(__name__)


def read_consumer_data(path, conf):
    logger.info(f'Data loading...')

    data = read_data_gen(path)

    data = (rec for rec in data if rec['target_target_flag'] is not None)
    data = (rec for rec in data if not np.isnan(int(rec['target_target_flag'])))
    data = prepare_embeddings(data, conf, is_train=False)
    data = list(data)

    for rec in data:
        rec['target'] = int(rec['target_target_flag'])

    logger.info(f'Loaded data with target: {len(data)}')

    return data


class ClippingDataset(Dataset):
    def __init__(self, delegate, min_len=250, max_len=350, rate_for_min=0.9):
        super().__init__()

        self.delegate = delegate
        self.min_len = min_len
        self.max_len = max_len
        self.rate_for_min = rate_for_min

    def __len__(self):
        return len(self.delegate)

    def __getitem__(self, item):
        item = self.delegate[item]

        seq_len = len(item['event_time'])
        if seq_len <= 5:
            return item

        new_len = random.randint(self.min_len, self.max_len)
        if new_len > seq_len * self.rate_for_min:
            new_len = math.ceil(seq_len * self.rate_for_min)

        avail_pos = seq_len - new_len
        pos = random.randint(0, avail_pos)

        item = deepcopy(item)
        item['feature_arrays'] = {k: v[pos:pos+new_len] for k, v in item['feature_arrays'].items()}
        item['event_time'] = item['event_time'][pos:pos+new_len]
        return item


def create_ds(train_data, valid_data, conf):
    if 'clip_seq' in conf['params.train']:
        train_data = ClippingDataset(train_data,
                                     min_len=conf['params.train.clip_seq.min_len'],
                                     max_len=conf['params.train.clip_seq.max_len'],
                                     )

    train_ds = ConvertingTrxDataset(TrxDataset(train_data, y_dtype=np.int64))
    valid_ds = ConvertingTrxDataset(TrxDataset(valid_data, y_dtype=np.int64))

    return train_ds, valid_ds


def run_experiment(train_ds, valid_ds, params, model_f):
    model = model_f(params)

    train_ds = DropoutTrxDataset(train_ds, params['train.trx_dropout'], params['train.max_seq_len'])
    train_loader = DataLoader(
        train_ds,
        batch_size=params['train.batch_size'],
        drop_last=True,
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
    # min_seq_len = conf['dataset'].get('min_seq_len', 1)
    min_seq_len = 1

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
        i_train_data = [rec for i, rec in enumerate(train_data)
                        if i in i_train and len(rec['event_time']) >= min_seq_len]

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
