import torch
import random

from ptls.data_load import padded_collate_wo_target
from ptls.data_load.utils import collate_feature_dict
from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.augmentations.random_slice import RandomSlice
from ptls.data_load.augmentations.sequence_pair_augmentation import sequence_pair_augmentation


class MlmDataset(torch.utils.data.Dataset):
    """

    Parameters
    ----------
    data:
        List with all sequences
    min_len:
        RandomSlice params.
    max_len:
        RandomSlice params.
    rate_for_min:
        RandomSlice params.
    """
    def __init__(self,
                 data,
                 min_len, max_len, rate_for_min: float = 1.0,
                 *args, **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.data = data
        self.r_slice = RandomSlice(min_len, max_len, rate_for_min)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        feature_arrays = self.data[item]
        return self.process(feature_arrays)

    def __iter__(self):
        for feature_arrays in self.data:
            yield self.process(feature_arrays)

    def process(self, feature_arrays):
        feature_arrays = {k: v for k, v in feature_arrays.items() if FeatureDict.is_seq_feature(k, v)}
        return self.r_slice(feature_arrays)

    @staticmethod
    def collate_fn(batch):
        return collate_feature_dict(batch)


class MlmIterableDataset(MlmDataset, torch.utils.data.IterableDataset):
    pass

class MLMNSPDataset(MlmDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def collate_fn(batch):
        max_lenght = max([len(next(iter(rec.values()))) for rec in batch])

        lefts, rights = [], []
        for rec in batch:
            left, right = sequence_pair_augmentation(rec, max_lenght=max_lenght)
            lefts.append(left)
            rights.append(right)
        
        #
        lefts = lefts * 2
        rights_ = rights[:]
        random.shuffle(rights_)
        rights += rights_

        targets = torch.cat([
            torch.ones(len(batch), dtype=torch.int64),
            torch.zeros(len(batch), dtype=torch.int64),
        ])
        
        concated = [{k: torch.cat([l[k], r[k]]) for k in l.keys()} for l, r in zip(lefts, rights)]
        
        augmented_batch =  padded_collate_wo_target(concated)
        return augmented_batch, targets.float()
