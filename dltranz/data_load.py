import random
from functools import partial
from collections import defaultdict
from multiprocessing.pool import Pool

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, Sampler, Dataset
from torch.utils.data.dataloader import DataLoader

from dltranz.seq_encoder import PaddedBatch


def default_preprocess(data, conf, drop_unknown=True):
    random.seed(conf['params.seed'])

    lag_days = conf['params.lag_days']
    max_days = conf['params.max_days']
    transaction_start_date = np.datetime64(conf.get('dataset.transaction_start_date', '1970-01-01T00:00'))

    for rec in data:
        target = rec['target']
        if target == -1 and drop_unknown:
            continue

        zero_take_prob = rec.get('zero_take_prob', 1.)

        if target == 0 and zero_take_prob < 1.0 and random.random() > zero_take_prob:
            continue

        feature_arrays = rec['feature_arrays']

        min_seq_len = rec.get('min_seq_len', 0)

        tranz_dates = rec['event_time']
        n_tranz = len(tranz_dates)
        if n_tranz < min_seq_len:
            continue

        rec['seq_len'] = n_tranz
        if n_tranz > 0:
            lagged_application_date = rec['application_date'] - np.timedelta64(lag_days, 'D')
            first_tranz_date = lagged_application_date - np.timedelta64(max_days, 'D')
            first_tranz_date = max(transaction_start_date, first_tranz_date)

            app_date_pos = np.searchsorted(tranz_dates, lagged_application_date)
            window_start_pos = np.searchsorted(tranz_dates, first_tranz_date)

            if (app_date_pos - window_start_pos) < min_seq_len:
                continue

            feature_arrays_tl = {
                k: arr[window_start_pos:app_date_pos]
                for k, arr in feature_arrays.items()
            }
            rec['feature_arrays'] = feature_arrays_tl
            rec['event_time'] = tranz_dates[window_start_pos:app_date_pos]

        yield rec


def read_spark_dataset_gen(data_files):
    import sparkpickle
    for path in data_files:
        with open(path, 'rb') as f:
            for row in sparkpickle.load_gen(f):
                yield dict(row)


def data_read_main(path, prepare_gen):
    import sparkpickle
    with open(path, 'rb') as f:
        rec_gen = (dict(e) for e in sparkpickle.load_gen(f))
        prepared_rec_gen = prepare_gen(rec_gen)
        return list(prepared_rec_gen)


def read_dataset_mthread(files, n_workers, prepare_gen):
    if n_workers > 1:
        with Pool(n_workers) as p:
            for chunk in p.imap_unordered(partial(data_read_main, prepare_gen=prepare_gen), files):
                for rec in chunk:
                    yield rec
    else:
        for rec in prepare_gen(read_spark_dataset_gen(files)):
            yield rec


def to_torch_compatible(a):
    if a.dtype == np.int8:
        return a.astype(np.int16)
    return a


def features2torch(seq):
    for rec in seq:
        rec['feature_arrays'] = {k: torch.from_numpy(to_torch_compatible(v)) for k, v in rec['feature_arrays'].items()}
        yield rec


class DropoutTrxDatasetIpoteka(Dataset):
    def get_targets(self):
        self.core_dataset.get_targets()

    def __init__(self, dataset: Dataset, trx_dropout, seq_len):
        self.core_dataset = dataset
        self.trx_dropout = trx_dropout
        self.max_seq_len = seq_len

    def __len__(self):
        return len(self.core_dataset)

    def __getitem__(self, idx):
        inp, y = self.core_dataset[idx]

        outs = dict()
        for key, x in inp.items():
            seq_len = len(next(iter(x.values())))

            if self.trx_dropout > 0:
                idx = np.random.choice(seq_len, size=int(seq_len * (1 - self.trx_dropout)+1), replace=False)
                idx = np.sort(idx)
            else:
                idx = np.arange(seq_len)

            idx = idx[-self.max_seq_len:]
            new_x = {k: v[idx] for k, v in x.items()}
            outs[key] = new_x

        return outs, y


class DropoutTrxDataset(Dataset):
    def __init__(self, dataset: Dataset, trx_dropout, seq_len):
        self.core_dataset = dataset
        self.trx_dropout = trx_dropout
        self.max_seq_len = seq_len

    def __len__(self):
        return len(self.core_dataset)

    def __getitem__(self, idx):
        item = self.core_dataset[idx]
        if type(item) is list:
            return [self._one_item(t) for t in item]
        else:
            return self._one_item(item)

    def _one_item(self, item):
        x, y = item

        seq_len = len(next(iter(x.values())))

        if self.trx_dropout > 0 and seq_len > 0:
            idx = np.random.choice(seq_len, size=int(seq_len * (1 - self.trx_dropout)+1), replace=False)
            idx = np.sort(idx)
        else:
            idx = np.arange(seq_len)

        idx = idx[-self.max_seq_len:]
        new_x = {k: v[idx] for k, v in x.items()}

        return new_x, y


class LastKTrxDataset(Dataset):
    def __init__(self, dataset: Dataset, share):
        self.core_dataset = dataset
        self.share = share

    def __len__(self):
        return len(self.core_dataset)

    def __getitem__(self, idx):
        x, y = self.core_dataset[idx]

        seq_len = len(next(iter(x.values())))

        if self.share < 1.:
            start_idx = int(seq_len*(1-self.share))
            idx_to_take = list(range(seq_len))[start_idx:]
            new_x = {k: v[idx_to_take] for k, v in x.items()}
        else:
            new_x = x

        return new_x, y


class TrxDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]['feature_arrays']
        y = self.data[idx]['target']

        return x, y


class ConvertingTrxDataset(Dataset):
    def __init__(self, delegate):
        self.delegate = delegate

    def __len__(self):
        return len(self.delegate)

    def __getitem__(self, idx):
        item = self.delegate[idx]
        if type(item) is list:
            return [self._one_item(t) for t in item]
        else:
            return self._one_item(item)

    def _one_item(self, item):
        x, y = item
        x = {k: torch.from_numpy(self.to_torch_compatible(v)) for k, v in x.items()}
        return x, y

    @staticmethod
    def to_torch_compatible(a):
        if a.dtype == np.int8:
            return a.astype(np.int16)
        return a


def pad_sequence(sequence, alignment, max_len=None, pad_value=0.0):
    def get_pad(x, max_len):
        if alignment == 'left':
            return max_len - len(x), 0
        if alignment == 'right':
            return 0, max_len - len(x)
        else:
            raise AttributeError(f"Unknown `alignment`: {alignment}")

    if max_len is None:
        max_len = max(x.size()[0] for x in sequence)
    return torch.stack([torch.nn.functional.pad(x, get_pad(x, max_len), value=pad_value) for x in sequence])


def padded_collate(batch):
    new_x_ = defaultdict(list)
    for x, _ in batch:
        for k, v in x.items():
            new_x_[k].append(v)

    lengths = torch.IntTensor([len(e) for e in next(iter(new_x_.values()))])

    new_x = {k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True) for k, v in new_x_.items()}
    new_y = torch.tensor([y for _, y in batch])

    return PaddedBatch(new_x, lengths), new_y


def padded_collate_ipoteka(batch):

    def padded_collate_(batch, key):
        new_x_ = {}
        for x, _ in batch:
            for k, v in x[key].items():
                if k in new_x_:
                    new_x_[k].append(v)
                else:
                    new_x_[k] = [v]

        lengths = torch.LongTensor([len(e) for e in next(iter(new_x_.values()))])

        new_x = {k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True) for k, v in new_x_.items()}
        new_y = torch.tensor([y for _, y in batch])

        return PaddedBatch(new_x, lengths), new_y

    batches, target = dict(), None
    for key in batch[0][0].keys():
        b, target = padded_collate_(batch, key)
        batches[key] = b

    return batches, target


class ZeroDownSampler(Sampler):
    def __init__(self, targets):
        super().__init__(None)
        self.pos_ids = np.array([idx for idx, y in enumerate(targets) if y == 1])
        self.neg_ids = np.array([idx for idx, y in enumerate(targets) if y == 0])

    def __iter__(self):
        neg_idx = np.random.choice(len(self.neg_ids), size=min(len(self.pos_ids), len(self.neg_ids)), replace=False)
        pos_idx = np.random.choice(len(self.pos_ids), size=len(self.pos_ids), replace=False)

        sampler_order = np.concatenate((self.pos_ids[pos_idx], self.neg_ids[neg_idx]))
        np.random.shuffle(sampler_order)
        return iter(sampler_order)

    def __len__(self):
        return len(self.pos_ids)*2


def create_weighted_random_sampler(targets):
    n_samples = len(targets)
    n_pos = sum(targets)
    n_neg = n_samples - n_pos
    w_neg = np.min([n_pos / n_neg, 1.])

    weights = [1.0 if y == 1 else w_neg for y in targets]
    n_take = int(np.min([n_pos * 2, n_pos + n_neg]))
    return WeightedRandomSampler(weights, n_take)


def create_train_loader(dataset, params, sampler=None):
    if isinstance(list(next(iter(dataset))[0].values())[0], dict):
        return create_train_loader_ipoteka(dataset, params, sampler)
    else:
        return create_train_loader_common(dataset, params, sampler)


def create_validation_loader(dataset, params):
    if isinstance(list(next(iter(dataset))[0].values())[0], dict):
        return create_validation_loader_ipoteka(dataset, params)
    else:
        return create_validation_loader_common(dataset, params)


def create_train_loader_common(dataset, params, sampler=None):
    dataset = DropoutTrxDataset(dataset, params['trx_dropout'], params['max_seq_len'])

    valid_loader = DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=params['num_workers'],
        collate_fn=padded_collate)

    return valid_loader


def create_validation_loader_common(dataset, params):
    dataset = DropoutTrxDataset(dataset, 0, params['max_seq_len'])

    valid_loader = DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=params['num_workers'],
        collate_fn=padded_collate)

    return valid_loader


def create_train_loader_ipoteka(dataset, params, sampler=None):
    dataset = DropoutTrxDatasetIpoteka(dataset, params['trx_dropout'], params['seq_len'])

    valid_loader = DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=params['num_workers'],
        collate_fn=padded_collate_ipoteka)

    return valid_loader


def create_validation_loader_ipoteka(dataset, params):
    dataset = DropoutTrxDatasetIpoteka(dataset, 0, params['seq_len'])

    valid_loader = DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=params['num_workers'],
        collate_fn=padded_collate_ipoteka)

    return valid_loader
