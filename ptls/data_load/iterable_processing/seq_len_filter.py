import numpy as np
import torch
import logging

from ptls.data_load import IterableProcessingDataset

logger = logging.getLogger(__name__)


class SeqLenFilter(IterableProcessingDataset):
    def __init__(self, min_seq_len=None, max_seq_len=None, seq_len_col=None, sequence_col=None):
        """

        Args:
            min_seq_len: if set than drop sequences shorter than `min_seq_len`
            max_seq_len: if set than drop sequences longer than `max_seq_len`
            seq_len_col: field where sequence length stored, if None, `target_col` used
            sequence_col: field for sequence length detection, if None, any iterable field will be used
        """
        super().__init__()

        self._min_seq_len = min_seq_len
        self._max_seq_len = max_seq_len
        self._sequence_col = sequence_col
        self._seq_len_col = seq_len_col

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            seq_len = self.get_len(features)
            if self._min_seq_len is not None and seq_len < self._min_seq_len:
                continue
            if self._max_seq_len is not None and seq_len > self._max_seq_len:
                continue
            yield rec

    def get_sequence_col(self, rec):
        if self._sequence_col is None:
            arrays = [k for k, v in rec.items() if type(v) in (list, np.ndarray, torch.Tensor)]
            if len(arrays) == 0:
                raise ValueError(f'Can not find field with sequence from record: {rec}')
            self._sequence_col = arrays[0]
        return self._sequence_col

    def get_len(self, rec):
        if self._seq_len_col is not None:
            return rec[self._seq_len_col]
        return len(rec[self.get_sequence_col(rec)])
