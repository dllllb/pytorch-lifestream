from functools import reduce
from operator import iadd

import torch

from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict
from ptls.frames.coles.split_strategy import AbsSplit


class ClusterDataset(FeatureDict, torch.utils.data.Dataset):
    """Dataset for ptls.frames.coles.CoLESModule

    Parameters
    ----------
    data:
        source data with feature dicts
    splitter:
        object from from `ptls.frames.coles.split_strategy`.
        Used to split original sequence into subsequences which are samples from one client.
    col_time:
        column name with event_time
    """

    def __init__(self,
                 data,
                 splitter: AbsSplit,
                 col_time='event_time',
                 col_idx='inn',
                 idx_dict=None,
                 non_sequential_features=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)  # required for mixin class

        self.data = data
        self.splitter = splitter
        self.col_time = col_time
        self.non_sequential_features = non_sequential_features
        self.col_idx = col_idx
        self.idx_dict = idx_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_arrays = self.data[idx]
        return self.get_splits(feature_arrays)

    def __iter__(self):
        for feature_arrays in self.data:
            yield self.get_splits(feature_arrays)

    def get_splits(self, feature_arrays):
        idx = self.idx_dict[feature_arrays[self.col_idx]]
        local_date = feature_arrays[self.col_time]
        indexes = self.splitter.split(local_date)
        splits = [{k: v[ix] for k, v in feature_arrays.items() if self.is_seq_feature(k, v)} for ix in indexes]
        return splits, idx

    @staticmethod
    def collate_fn(batch):
        cluster_class_labels = [sample[1] for sample in batch for _ in sample[0]]
        coles_class_labels = [i for i, class_samples in enumerate(batch) for _ in class_samples[0]]
        batch = reduce(iadd, [sample[0] for sample in batch])
        padded_batch = collate_feature_dict(batch)
        return padded_batch, {"cluster_target": torch.LongTensor(cluster_class_labels),
                              "coles_target": torch.LongTensor(coles_class_labels)}


class ClusterIterableDataset(ClusterDataset, torch.utils.data.IterableDataset):
    pass
