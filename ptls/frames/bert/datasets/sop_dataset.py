import torch
from ptls.data_load import padded_collate_wo_target
from ptls.frames.coles import ColesDataset
from ptls.data_load.augmentations.sequence_pair_augmentation import sequence_pair_augmentation
from functools import reduce
from operator import iadd


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
                padded_collate_wo_target(lefts),
                padded_collate_wo_target(rights)
            ),
            targets.float()
        )


class SopIterableDataset(SopDataset, torch.utils.data.IterableDataset):
    pass
