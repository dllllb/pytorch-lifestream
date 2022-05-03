import random
import math
import numpy as np


class RandomSlice:
    def __init__(self, min_len, max_len, rate_for_min=1.0):
        super().__init__()

        self.min_len = min_len
        self.max_len = max_len
        self.rate_for_min = rate_for_min

    def __call__(self, x):
        seq_len = len(next(iter(x.values())))

        idx = self.get_idx(seq_len)
        new_x = {k: v[idx] for k, v in x.items()}
        return new_x

    def get_idx(self, seq_len):
        new_idx = np.arange(seq_len)

        min_len, max_len = self.get_min_max(seq_len)
        if max_len < min_len:
            return new_idx
        new_len = random.randint(min_len, max_len)

        avail_pos = seq_len - new_len
        pos = random.randint(0, avail_pos)
        return new_idx[pos:pos+new_len]

    def get_min_max(self, seq_len):
        max_len = int(min(self.max_len, seq_len))
        min_len = int(min(self.min_len, seq_len * self.rate_for_min))
        if min_len < 1:
            min_len = 1
        return min_len, max_len
