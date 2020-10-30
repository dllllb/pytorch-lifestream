from torch.utils.data.dataset import IterableDataset
import numpy as np
import torch


class SeqLenFilter(IterableDataset):
    def __init__(self, min_seq_len=None, max_seq_len=None, target_col=None):
        """

        Args:
            min_seq_len: if set than drop sequences shorter than `min_seq_len`
            max_seq_len: if set than drop sequences longer than `max_seq_len`
            target_col: field for sequence length detection, if None, any iterable field will be used
        """
        self._min_seq_len = min_seq_len
        self._max_seq_len = max_seq_len
        self._target_col = target_col

        self._src = None

    def __call__(self, src):
        self._src = src
        return self

    def target_call(self, rec):
        if self._target_col is None:
            self._target_col = next(k for k, v in rec.items() if type(v) in (list, np.ndarray, torch.tensor))
        return self._target_col

    def __iter__(self):
        for rec in self._src:
            seq_len = len(rec[self.target_call(rec)])
            if self._min_seq_len is not None and seq_len < self._min_seq_len:
                continue
            if self._max_seq_len is not None and seq_len > self._max_seq_len:
                continue
            yield rec
