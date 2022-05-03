# coding: utf-8
import logging
import torch

from torch.utils.data import Dataset, IterableDataset

from ptls.data_load import IterableProcessingDataset

logger = logging.getLogger(__name__)


class SplittingDataset(Dataset):
    def __init__(self, base_dataset, splitter):
        self.base_dataset = base_dataset
        self.splitter = splitter

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        row = self.base_dataset[idx]

        feature_arrays = row['feature_arrays']
        local_date = row['event_time']

        indexes = self.splitter.split(local_date)
        data = [{k: v[ix] for k, v in feature_arrays.items()} for ix in indexes]
        return data


class MapSplittingDataset(Dataset):
    def __init__(self, base_dataset, splitter, a_chain):
        self.base_dataset = base_dataset
        self.splitter = splitter
        self.a_chain = a_chain

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        feature_arrays = self.base_dataset[idx]
        local_date = feature_arrays['event_time']
        indexes = self.splitter.split(local_date)
        return [self.a_chain({k: v[ix] for k, v in feature_arrays.items()}) for ix in indexes]


class IterableSplittingDataset(IterableProcessingDataset):
    def __init__(self, splitter, a_chain, col_time='event_time'):
        super().__init__()

        self.splitter = splitter
        self.a_chain = a_chain
        self.col_time = col_time

    def __iter__(self):
        for row in self._src:
            local_date = row[self.col_time]
            yield [self.a_chain({k: v[ix] for k, v in row.items()}) for ix in self.splitter.split(local_date)]


class SeveralSplittingsDataset(Dataset):
    def __init__(self, base_dataset, splitters):
        self.base_dataset = base_dataset
        self.splitters = splitters

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        row = self.base_dataset[idx]

        feature_arrays = row['feature_arrays']
        local_date = row['event_time']

        data = []
        for splitter in self.splitters:
            indexes = splitter.split(local_date)
            data += [{k: v[ix] for k, v in feature_arrays.items()} for ix in indexes]
        return data

