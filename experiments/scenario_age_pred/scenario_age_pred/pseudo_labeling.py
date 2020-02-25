import datetime
import logging
import numpy as np
import random
import sys
import torch

from collections import defaultdict
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import StratifiedKFold

from dltranz.data_load import TrxDataset, ConvertingTrxDataset, DropoutTrxDataset
from dltranz.experiment import get_epoch_score_metric, update_model_stats
from dltranz.loss import get_loss
from dltranz.metric_learn.inference_tools import infer_part_of_data, save_scores
from dltranz.models import model_by_type
from dltranz.seq_encoder import PaddedBatch
from dltranz.train import get_optimizer, get_lr_scheduler, fit_model
from dltranz.util import init_logger, get_conf
from scenario_age_pred.fit_target import read_consumer_data, EpochTrackingDataLoader, SubsamplingDataset

logger = logging.getLogger(__name__)


class ZipDataset(Dataset):
    def __init__(self, LabeledSet, UnLabeledSet):
        self.LabeledSet = LabeledSet
        self.UnLabeledSet = UnLabeledSet
        self.n = 0

    def __len__(self):
        return len(self.LabeledSet)

    def __getitem__(self, i):
        j = random.randint(0,len(self.UnLabeledSet)-1)
        return (self.LabeledSet[i], self.UnLabeledSet[j])


class ModelPLWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return {'labeled':self.model(x['labeled']), 'unlabeled':self.model(x['unlabeled'])}


def padded_zip_collate(batch):
    new_lx_ = defaultdict(list)
    new_ux_ = defaultdict(list)
    for (lx, y), (ux, _) in batch:
        for k, v in lx.items():
            new_lx_[k].append(v)
        for k, v in ux.items():
            new_ux_[k].append(v)

    l_lengths = torch.IntTensor([len(e) for e in next(iter(new_lx_.values()))])
    new_lx = {k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True) for k, v in new_lx_.items()}

    u_lengths = torch.IntTensor([len(e) for e in next(iter(new_ux_.values()))])
    new_ux = {k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True) for k, v in new_ux_.items()}

    new_y = torch.tensor([y for (lx, y), (ux, _) in batch])

    return ({'labeled':PaddedBatch(new_lx, l_lengths), 'unlabeled':PaddedBatch(new_ux, u_lengths)}, new_y)


def prepare_parser(parser):
    pass


def create_ds(labeled_data, unlabeled_data, valid_data, conf):
    if 'SubsamplingDataset' in conf['params.train']:
        labeled_data = SubsamplingDataset(labeled_data, **conf['params.train.SubsamplingDataset'])
        unlabeled_data = SubsamplingDataset(unlabeled_data, **conf['params.train.SubsamplingDataset'])

    labeled_ds = ConvertingTrxDataset(TrxDataset(labeled_data, y_dtype=np.int64))
    unlabeled_ds = ConvertingTrxDataset(TrxDataset(unlabeled_data, y_dtype=np.int64))
    valid_ds = ConvertingTrxDataset(TrxDataset(valid_data, y_dtype=np.int64))

    return labeled_ds, unlabeled_ds, valid_ds


def run_experiment(labeled_ds, unlabeled_ds, valid_ds, params, model_f):
    model = model_f(params)

    # pseudo-labeling training
    labeled_ds = DropoutTrxDataset(labeled_ds, params['train.trx_dropout'], params['train.max_seq_len'])
    unlabeled_ds = DropoutTrxDataset(unlabeled_ds, params['train.trx_dropout'], params['train.max_seq_len'])
    zip_ds = ZipDataset(labeled_ds, unlabeled_ds)
    zip_loader = EpochTrackingDataLoader(
        zip_ds,
        batch_size=params['train.batch_size'],
        shuffle=True,
        num_workers=params['train.num_workers'],
        collate_fn=padded_zip_collate)
    zip_loader.preparing_dataset = zip_ds.LabeledSet.core_dataset.delegate.data
    
    valid_ds_ = DropoutTrxDataset(valid_ds, 0, params['valid.max_seq_len'])
    zip_valid_ds = ZipDataset(valid_ds_, valid_ds_)
    zip_valid_loader = DataLoader(
        zip_valid_ds,
        batch_size=params['valid.batch_size'],
        shuffle=False,
        num_workers=params['valid.num_workers'],
        collate_fn=padded_zip_collate
    )

    metric_name = params['score_metric']
    metric = get_epoch_score_metric(metric_name)()
    handlers = []
    pl_model = ModelPLWrapper(model)
    loss = get_loss(params)
    opt = get_optimizer(model, params)
    scheduler = get_lr_scheduler(opt, params)
    scores = fit_model(pl_model, zip_loader, zip_valid_loader, loss, opt, scheduler, params, {metric_name: metric}, handlers)

    return model, {
        **scores,
        'finish_time': datetime.datetime.now().isoformat(),
    }


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
    assert nrows>0, "params.labeled_amount should be defined > 0"
    
    target_values = [rec['target'] for rec in train_data]
    for i, (i_train, i_valid) in enumerate(skf.split(train_data, target_values)):
        logger.info(f'Train fold: {i}')
        i_train_data = [rec for i, rec in enumerate(train_data) if i in i_train]
        i_valid_data = [rec for i, rec in enumerate(train_data) if i in i_valid]

        i_labeled_data = i_train_data[:nrows]
        i_unlabeled_data = i_train_data[nrows:]

        labeled_ds, unlabeled_ds, valid_ds = create_ds(i_labeled_data, i_unlabeled_data, i_valid_data, conf)
        model, result = run_experiment(labeled_ds, unlabeled_ds, valid_ds, conf['params'], model_f)

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
