import numpy as np
import torch
import random

from ptls.data_load.utils import collate_feature_dict
from ptls.data_load.feature_dict import FeatureDict


class MlmIndexedDataset(torch.utils.data.Dataset):
    """

    Parameters
    ----------
    data:
        List with all sequences
    seq_len:
        Length of sampled sequence
    step_rate:
        Define step of window moving. `step = seq_len * step_rate`.
        When `step_rate == 1.0` then `step = seq_len` ans windows aren't intersect
        When `step_rate < 1.0` then sampled windows are intersect
        When `step_rate > 1.0` then sampled windows aren't intersect and there are missing transactions
    random_shift:
        Move window start position in (-random_shift, random_shift) interval randomly
    random_crop:
        Reduce lenght of sampled sequence in (0, random_crop) interval randomly
    """
    def __init__(self,
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
            start_pos = start_pos + random.randint(-self.random_shift, self.random_shift)
        start_pos = max(start_pos, 0)
        start_pos = min(start_pos, seq_len - self.step)
        len_reduce = 0 if self.random_crop == 0 else random.randint(0, self.random_crop)

        return {k: v[start_pos: start_pos + self.seq_len - len_reduce]
                for k, v in v.items()
                if FeatureDict.is_seq_feature(k, v)}

    @staticmethod
    def collate_fn(batch):
        return collate_feature_dict(batch)
