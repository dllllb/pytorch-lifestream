import numpy as np
import torch

from ptls.data_load import IterableProcessingDataset


class TakeFirstTrx(IterableProcessingDataset):
    def __init__(self, take_first_fraction=0.5, seq_len_col=None, sequence_col=None):
        """
        Args:
            take_first_fraction: control the fraction of transactions to keep
                                 EXAMPLE: take_first_fraction=0.75 -> the last 0.25 of all user
                                          transactions will be chosen as user target distribution
                                          and therefore will be cutted off
            seq_len_col:  field where sequence length stored, if None, `target_col` used
            sequence_col: field for sequence length detection, if None, any
                          iterable field will be used
        """
        super().__init__()

        self._take_first_fraction = take_first_fraction
        self._sequence_col = sequence_col
        self._seq_len_col = seq_len_col

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            seq_len = self.get_len(features)
            take_first_n = int(seq_len * self._take_first_fraction)
            for key, val in features.items():
                if type(val) in (list, np.ndarray, torch.Tensor):
                    features[key] = val[:take_first_n]
            rec = (features, rec[1]) if type(rec) is tuple else features
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
