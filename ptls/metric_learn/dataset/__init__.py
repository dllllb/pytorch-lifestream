# coding: utf-8
"""

"""
import functools
import operator

from torch.utils.data import DataLoader

from ptls.data_load import padded_collate, ConvertingTrxDataset, DropoutTrxDataset, ProcessDataset
from ptls.metric_learn.dataset.splitting_dataset import SplittingDataset, SeveralSplittingsDataset
from ptls.frames.coles import split_strategy
from ptls.metric_learn.dataset.target_enumerator_dataset import TargetEnumeratorDataset
from ptls.metric_learn.dataset.complex_target_dataset import ComplexTargetDataset


def nested_list_to_flat_with_collate(collate_fn):
    """Deprecated
    """
    def _func(batch):
        batch = functools.reduce(operator.iadd, batch)
        return collate_fn(batch)

    return _func


def collate_splitted_rows(batch):
    # add Y and flatten samples
    batch = functools.reduce(operator.iadd, batch)
    return padded_collate(batch)


def create_train_data_loader(data, conf):
    dataset = SplittingDataset(
        data,
        split_strategy.create(**conf.params.train.split_strategy)
    )
    dataset = TargetEnumeratorDataset(dataset)
    dataset = ConvertingTrxDataset(dataset)
    dataset = DropoutTrxDataset(
        dataset,
        trx_dropout=conf.params.train.trx_dropout,
        seq_len=conf.params.train.max_seq_len)

    data_loader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_splitted_rows,
        num_workers=conf.params.train.num_workers
    )
    return data_loader


def create_valid_data_loader(data, conf, sample_process_fun=None):
    dataset = SplittingDataset(
        data,
        split_strategy.create(**conf.params.valid.split_strategy)
    )
    dataset = TargetEnumeratorDataset(dataset)
    dataset = ConvertingTrxDataset(dataset)
    if sample_process_fun is not None:
        dataset = ProcessDataset(dataset, sample_process_fun)
    dataset = DropoutTrxDataset(dataset, trx_dropout=0.0, seq_len=conf.params.valid.max_seq_len)

    data_loader = DataLoader(
        dataset,
        batch_size=conf.params.valid.batch_size,
        shuffle=False,
        collate_fn=collate_splitted_rows,
        num_workers=conf.params.valid.num_workers,
    )
    return data_loader


def create_valid_complex_data_loader(data, conf, sample_process_fun=None):
    split_strategies = [
        split_strategy.create(**split_strategy_params) for split_strategy_params in
        conf.params.valid.split_strategies
    ]
    split_counts = [
        split_strategy_params.split_count for split_strategy_params in conf.params.valid.split_strategies
    ]
    dataset = SeveralSplittingsDataset(
        data,
        split_strategies
    )
    dataset = ComplexTargetDataset(dataset, split_counts)
    dataset = ConvertingTrxDataset(dataset)
    if sample_process_fun is not None:
        dataset = ProcessDataset(dataset, sample_process_fun)
    dataset = DropoutTrxDataset(dataset, trx_dropout=0.0, seq_len=conf.params.valid.max_seq_len)

    data_loader = DataLoader(
        dataset,
        batch_size=conf.params.valid.batch_size,
        shuffle=False,
        collate_fn=collate_splitted_rows,
        num_workers=conf.params.valid.num_workers,
    )
    return data_loader
