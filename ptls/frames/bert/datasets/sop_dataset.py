from functools import reduce
from operator import iadd

import torch

from ptls.data_load.augmentations.sequence_pair_augmentation import sequence_pair_augmentation
from ptls.data_load.utils import collate_feature_dict
from ptls.frames.coles import ColesDataset


class SopDataset(ColesDataset):
    def get_splits(self, feature_arrays):
        return [sequence_pair_augmentation(item)
                for item in super().get_splits(feature_arrays)]

    @staticmethod
    def collate_fn(batch):
        batch = reduce(iadd, batch)
        targets = torch.randint(low=0, high=2, size=(len(batch),))

        lefts = [left if target else right for (left, right), target in zip(batch, targets)]
        rights = [right if target else left for (left, right), target in zip(batch, targets)]

        return (
            (
                collate_feature_dict(lefts),
                collate_feature_dict(rights)
            ),
            targets.long()
        )


class SopIterableDataset(SopDataset, torch.utils.data.IterableDataset):
    pass
