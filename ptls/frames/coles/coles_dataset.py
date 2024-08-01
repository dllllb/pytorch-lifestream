from functools import reduce
from operator import iadd

import joblib
import torch
from joblib import Parallel, delayed

from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict
from ptls.frames.coles.split_strategy import AbsSplit


class ColesDataset(FeatureDict, torch.utils.data.Dataset):
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
                 *args, **kwargs):
        super().__init__(*args, **kwargs)  # required for mixin class

        self.data = data
        self.splitter = splitter
        self.col_time = col_time

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_arrays = self.data[idx]
        return self.get_splits(feature_arrays)

    def __iter__(self):
        for feature_arrays in self.data:
            yield self.get_splits(feature_arrays)

    def _create_split_subset(self, idx, feature_arrays):
        return {k: v[idx] for k, v in feature_arrays.items() if not isinstance(v, int)}

    def get_splits(self, feature_arrays):
        local_date = feature_arrays[self.col_time]
        indexes = self.splitter.split(local_date)
        with joblib.parallel_backend(backend='threading'):
            parallel = Parallel()
            result_dict = parallel(delayed(self._create_split_subset)(idx, feature_arrays)
                                   for idx in indexes)

        return result_dict

    @staticmethod
    def collate_fn(batch):
        class_labels = torch.LongTensor(reduce(iadd, list(map(lambda x: [x[0] for _ in x[1]], enumerate(batch)))))
        padded_batch = collate_feature_dict(reduce(iadd, batch))
        return padded_batch, class_labels


class ColesIterableDataset(ColesDataset, torch.utils.data.IterableDataset):
    pass
