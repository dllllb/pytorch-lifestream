from torch.utils.data.dataset import IterableDataset
import numpy as np
import torch


class SeqLenFilter(IterableDataset):
    def __init__(self, min_seq_len=None, max_seq_len=None, seq_len_col=None, target_col=None):
        """

        Args:
            min_seq_len: if set than drop sequences shorter than `min_seq_len`
            max_seq_len: if set than drop sequences longer than `max_seq_len`
            seq_len_col: field where sequence length stored, if None, `target_col` used
            target_col: field for sequence length detection, if None, any iterable field will be used
        """
        self._min_seq_len = min_seq_len
        self._max_seq_len = max_seq_len
        self._target_col = target_col
        self._seq_len_col = seq_len_col

        self._src = None

    def __call__(self, src):
        self._src = src
        return self

    def target_call(self, rec):
        if self._target_col is None:
            self._target_col = next(k for k, v in rec.items() if type(v) in (list, np.ndarray, torch.tensor))
        return self._target_col

    def get_len(self, rec):
        if self._seq_len_col is not None:
            return rec[self._seq_len_col]
        return len(rec[self.target_call(rec)])

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            seq_len = self.get_len(features)
            if self._min_seq_len is not None and seq_len < self._min_seq_len:
                continue
            if self._max_seq_len is not None and seq_len > self._max_seq_len:
                continue
            yield rec
