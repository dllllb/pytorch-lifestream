import logging
from itertools import compress
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

    def _valid_seq_len(self, seq_len):
        min_len_check = seq_len > self._min_seq_len if self._min_seq_len is not None else True
        max_len_check = seq_len < self._max_seq_len if self._max_seq_len is not None else True
        return all([min_len_check, max_len_check])

    def transform(self, features):
        valid_seq_len = self.get_len(features)
        return features if valid_seq_len else None

    def get_sequence_col(self, rec):
        iter_record = list(iter(rec))
        iter_record = list(compress(iter_record, [self.is_seq_feature(i) for i in iter_record]))
        if len(iter_record) == 0:
            raise ValueError(f'Can not find field with sequence from record: {rec}')
        else:
            return all([self._valid_seq_len(len(feature)) for feature in iter_record])

    def get_len(self, rec):
        if self._seq_len_col is not None:
            return rec[self._seq_len_col]
        else:
            return self.get_sequence_col(rec)
