import numpy as np
import torch
from typing import List, Dict
import random

from ptls.data_load import padded_collate_wo_target


class MLMDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data: List[Dict],
                 seq_len: int,
                 step_rate: float = 1.0,
                 random_shift: int = 0,
                 random_crop: int = 0,
                 ):
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
        super().__init__()

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
        et = next(iter(v.values()))

        if self.random_shift > 0:
            start_pos = start_pos + random.randint(-self.random_shift, self.random_shift)
        start_pos = max(start_pos, 0)
        start_pos = min(start_pos, len(et) - self.step)
        len_reduce = 0 if self.random_crop == 0 else random.randint(0, self.random_crop)

        return {k: v[start_pos: start_pos + self.seq_len - len_reduce] for k, v in v.items()}

    def data_loader(self, shuffle: bool = False, num_workers: int = 0, batch_size: int = 512):
        """Returns torch.DataLoader with self Dataset

        Parameters
        ----------
        shuffle: passed to dataloader
        num_workers: passed to dataloader
        batch_size:  passed to dataloader

        Returns
        -------
        torch.utils.data.DataLoader
        """
        return torch.utils.data.DataLoader(
            dataset=self,
            collate_fn=padded_collate_wo_target,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size
        )
