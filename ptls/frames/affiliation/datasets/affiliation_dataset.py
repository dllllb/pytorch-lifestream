from functools import reduce
from operator import iadd

import torch

from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict
from ptls.frames.coles.split_strategy import AbsSplit


class AffiliationDataset(FeatureDict, torch.utils.data.Dataset):
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

    def get_splits(self, feature_arrays):
        local_date = feature_arrays[self.col_time]
        long_indexes, pos_indexes, neg_indexes = self.splitter.split(local_date)
        long = {i: {k: v[ix] for k, v in feature_arrays.items() if self.is_seq_feature(k, v)}
                for i, ix in long_indexes.items()}
        short = {i: [{k: v[ix] for k, v in feature_arrays.items() if self.is_seq_feature(k, v)} for ix in ixx]
                 for i, ixx in short_indexes.items()}
        return long, short

    @staticmethod
    def collate_fn(batch):
        long_batch, short_batch = list(), list()
        long_labels, short_labels = list(), list()
        global_i = 0

        for i, sample in enumerate(batch):
            for k in batch[0]:
                long_batch.append(sample[0][k])
                long_labels.append(global_i)
                short_batch.extend(sample[1][k])
                short_labels.extend([global_i for _ in range(len(sample[1][k]))])
                global_i += 1

        long_padded_batch = collate_feature_dict(long_batch)
        short_padded_batch = collate_feature_dict(short_batch)
        return long_padded_batch, short_padded_batch, torch.LongTensor(long_labels), torch.LongTensor(short_labels)


class AffiliationIterableDataset(AffiliationDataset, torch.utils.data.IterableDataset):
    pass
