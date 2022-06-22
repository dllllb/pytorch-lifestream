import torch
from ptls.data_load import padded_collate_wo_target
from ptls.frames.coles import ColesDataset
from ptls.data_load.augmentations.sequence_pair_augmentation import sequence_pair_augmentation
from functools import reduce
from operator import iadd
import random


class NspDataset(ColesDataset):
    def get_splits(self, feature_arrays):
        return [sequence_pair_augmentation(item)
                for item in super().get_splits(feature_arrays)]

    @staticmethod
    def collate_fn(batch):
        batch = reduce(iadd, batch)
        lefts = [left for left, _ in batch] * 2

        rights = [right for _, right in batch]
        rights_ = rights[:]
        random.shuffle(rights_)
        rights += rights_

        targets = torch.cat([
            torch.ones(len(batch), dtype=torch.int64),
            torch.zeros(len(batch), dtype=torch.int64),
        ])

        return (
            (
                padded_collate_wo_target(lefts),
                padded_collate_wo_target(rights)
            ),
            targets.float(),
        )


class NspIterableDataset(NspDataset, torch.utils.data.IterableDataset):
    pass
