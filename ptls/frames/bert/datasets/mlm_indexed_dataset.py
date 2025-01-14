import random

import numpy as np
import torch

from ptls.data_load import padded_collate_wo_target
from ptls.data_load.augmentations import sequence_pair_augmentation
from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict


class MlmIndexedDataset(torch.utils.data.Dataset):
    """
    A dataset class for Masked Language Modeling (MLM) with indexed sequences.

    Attributes:
        data (list): List containing all sequences.
        seq_len (int): Length of the sampled sequence.
        step_rate (float): Defines the step of window moving. `step = seq_len * step_rate`.
                          When `step_rate == 1.0`, `step = seq_len` and windows do not intersect.
                          When `step_rate < 1.0`, sampled windows intersect.
                          When `step_rate > 1.0`, sampled windows do not intersect and there are missing transactions.
        random_shift (int): Moves the window start position randomly within the interval (-random_shift, random_shift).
        random_crop (int): Reduces the length of the sampled sequence randomly within the interval (0, random_crop).

    Args:
        data (list): List with all sequences.
        seq_len (int): Length of the sampled sequence.
        step_rate (float, optional): Step rate for window moving. Defaults to 1.0.
        random_shift (int, optional): Random shift for window start position. Defaults to 0.
        random_crop (int, optional): Random crop for reducing sequence length. Defaults to 0.
    """

    def __init__(
        self,
        data,
        seq_len: int,
        step_rate: float = 1.0,
        random_shift: int = 0,
        random_crop: int = 0,
    ):
        self.data = data
        self.seq_len = seq_len
        self.step = int(seq_len * step_rate)
        self.random_shift = random_shift
        self.random_crop = random_crop

        assert self.random_crop < self.seq_len

        self.ix = []
        for item_id, v in enumerate(self.data):
            et = next(iter(v.values()))
            for start_pos in range(0, len(et), self.step):
                self.ix.append([item_id, start_pos])
        self.ix = np.array(self.ix)

    def __len__(self):
        return self.ix.shape[0]

    def __getitem__(self, item):
        item_id, start_pos = self.ix[item]
        v = self.data[item_id]
        seq_len = FeatureDict.get_seq_len(v)

        if self.random_shift > 0:
            start_pos = start_pos + random.randint(
                -self.random_shift, self.random_shift
            )
        start_pos = max(start_pos, 0)
        start_pos = min(start_pos, seq_len - self.step)
        len_reduce = 0 if self.random_crop == 0 else random.randint(0, self.random_crop)

        return {
            k: v[start_pos : start_pos + self.seq_len - len_reduce]
            for k, v in v.items()
            if FeatureDict.is_seq_feature(k, v)
        }

    @staticmethod
    def collate_fn(batch):
        return collate_feature_dict(batch)


class MLMNSPIndexedDataset(MlmIndexedDataset):
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

        targets = torch.cat(
            [
                torch.ones(len(batch), dtype=torch.int64),
                torch.zeros(len(batch), dtype=torch.int64),
            ]
        )

        concated = [
            {k: torch.cat([l[k], r[k]]) for k in l.keys()}
            for l, r in zip(lefts, rights)
        ]

        augmented_batch = padded_collate_wo_target(concated)
        return augmented_batch, targets.float()
