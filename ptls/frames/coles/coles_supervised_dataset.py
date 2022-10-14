from functools import reduce
from operator import iadd
from typing import List

import numpy as np
import torch

from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict
from ptls.frames.coles.split_strategy import AbsSplit


class ColesSupervisedDataset(FeatureDict, torch.utils.data.Dataset):
    """Dataset for ptls.frames.coles.ColesSupervisedModule

    Parameters
    ----------
    data:
        source data with feature dicts
    splitter:
        object from from `ptls.frames.coles.split_strategy`.
        Used to split original sequence into subsequences which are samples from one client.
    cols_classes:
        column names with class labels for auxiliary supervised loss calculation
    col_time:
        column name with event_time
    """
    def __init__(self,
                 data,
                 splitter: AbsSplit,
                 cols_classes: List[str],
                 col_time='event_time',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)  # required for mixin class

        self.data = data
        self.splitter = splitter
        self.col_time = col_time
        self.cols_classes = cols_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_arrays = self.data[idx]
        return self.get_splits(feature_arrays), self.get_classes(feature_arrays)

    def __iter__(self):
        for feature_arrays in self.data:
            yield self.get_splits(feature_arrays), self.get_classes(feature_arrays)

    def get_splits(self, feature_arrays):
        local_date = feature_arrays[self.col_time]
        indexes = self.splitter.split(local_date)
        return [{k: v[ix] for k, v in feature_arrays.items() if self.is_seq_feature(k, v)} for ix in indexes]

    def get_classes(self, feature_arrays):
        res = [feature_arrays.get(col, -1) for col in self.cols_classes]
        res = [r if ~np.isnan(r) else -1 for r in res]
        return res

    @staticmethod
    def collate_fn(batch):
        class_labels = [i for i, (seq, labels) in enumerate(batch) for _ in seq]
        seq_samples = [seq for seq, labels in batch]
        target_labels = [labels for seq, labels in batch for _ in seq]
        batch = reduce(iadd, seq_samples)
        padded_batch = collate_feature_dict(batch)
        return padded_batch, torch.LongTensor(class_labels), torch.LongTensor(target_labels)


class ColesSupervisedIterableDataset(ColesSupervisedDataset, torch.utils.data.IterableDataset):
    pass
