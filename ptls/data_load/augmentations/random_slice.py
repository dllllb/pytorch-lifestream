import random
from typing import Tuple

import numpy as np

from ptls.data_load.feature_dict import FeatureDict


class RandomSlice(FeatureDict):
    """
    This class is used as 'f_augmentation' argument for 
    ptls.data_load.datasets.augmentation_dataset.AugmentationDataset (AugmentationIterableDataset).
    """
    def __init__(self, 
                 min_len: int, 
                 max_len: int, 
                 rate_for_min: float = 1.0
                 ):
        super().__init__()

        self.min_len = min_len
        self.max_len = max_len
        self.rate_for_min = rate_for_min

    def __call__(self, x: dict) -> dict:
        seq_len = self.get_seq_len(x)

        idx = self.get_idx(seq_len)
        new_x = self.seq_indexing(x, idx)
        return new_x

    def get_idx(self, seq_len: int) -> np.ndarray:
        new_idx = np.arange(seq_len)

        min_len, max_len = self.get_min_max(seq_len)
        if max_len < min_len:
            return new_idx
        new_len = random.randint(min_len, max_len)

        avail_pos = seq_len - new_len
        pos = random.randint(0, avail_pos)
        return new_idx[pos:pos+new_len]

    def get_min_max(self, seq_len: int) -> Tuple[int, int]:
        max_len = int(min(self.max_len, seq_len))
        min_len = int(min(self.min_len, seq_len * self.rate_for_min))
        if min_len < 1:
            min_len = 1
        return min_len, max_len
