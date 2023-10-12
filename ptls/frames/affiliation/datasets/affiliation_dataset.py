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
        pos = {i: [{k: v[ix] for k, v in feature_arrays.items() if self.is_seq_feature(k, v)} for ix in ixx]
               for i, ixx in pos_indexes.items()}
        neg = {i: [{k: v[ix] for k, v in feature_arrays.items() if self.is_seq_feature(k, v)} for ix in ixx]
               for i, ixx in neg_indexes.items()}
        return long, pos, neg

    @staticmethod
    def get_random_inds(i, bs, n):
        return

    @staticmethod
    def collate_fn(batch):
        long_batch, short_batch = list(), list()
        labels = list()
        n_repeats = list()
        bs = len(batch)

        for i, sample in enumerate(batch):
            long, pos, neg = sample
            for k in long:
                long_batch.append(long[k])

                short_batch.extend(pos[k])
                labels.extend([1 for _ in range(len(pos[k]))])

                short_batch.extend(neg[k])
                inds = AffiliationDataset.get_random_inds(i, bs, int(len(neg[k])/2))
                for ind in inds:
                    _, pos_, neg_ = batch[ind]
                    short_batch.append(pos_[0][0])
                    short_batch.append(neg_[0][0])
                labels.extend([0 for _ in range(len(neg[k]) + int(len(neg[k])/2))])

                n_repeats.append(len(pos[k]) + len(neg[k]) + int(len(neg[k])/2))

        long_padded_batch = collate_feature_dict(long_batch)
        short_padded_batch = collate_feature_dict(short_batch)
        return long_padded_batch, short_padded_batch, torch.LongTensor(labels), torch.LongTensor(n_repeats)


class AffiliationIterableDataset(AffiliationDataset, torch.utils.data.IterableDataset):
    pass
