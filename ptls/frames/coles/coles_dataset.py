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

    Args:
        data: source data with feature dicts
        splitter: object from `ptls.frames.coles.split_strategy`.
            Used to split original sequence into subsequences which are samples from one client.
        col_time: column name with event_time
        n_jobs: number of workers requested by the callers. 
            Passing n_jobs=-1 means requesting all available workers for instance matching the number of
            CPU cores on the worker host(s).
    """

    def __init__(self,
                 data: dict,
                 splitter: AbsSplit,
                 col_time: str = 'event_time',
                 n_jobs: int = 1,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)  # required for mixin class

        self.data = data
        self.splitter = splitter
        self.col_time = col_time
        self.n_jobs = n_jobs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        feature_arrays = self.data[idx]
        return self.get_splits(feature_arrays)

    def __iter__(self):
        for feature_arrays in self.data:
            yield self.get_splits(feature_arrays)

    def _create_split_subset(self, idx, feature_arrays):
        return {k: v[idx] for k, v in feature_arrays.items() if self.is_seq_feature(k, v)}

    def get_splits(self, feature_arrays: dict):
        local_date = feature_arrays[self.col_time]
        indexes = self.splitter.split(local_date)
        with joblib.parallel_backend(backend='threading', n_jobs=self.n_jobs):
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
