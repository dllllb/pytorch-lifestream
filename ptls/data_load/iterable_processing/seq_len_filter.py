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

    def _valid_seq_len(self, seq_feature):
        if self.is_seq_feature(seq_feature):
            _ = len(seq_feature)
            min_len_check = _ > self._min_seq_len if self._min_seq_len is not None else True
            max_len_check = _ < self._max_seq_len if self._max_seq_len is not None else True
        return all([min_len_check, max_len_check])

    def transform(self, features):
        return features if self.get_len(features[list(features.keys())[0]]) else None

    def get_sequence_col(self, rec: dict):
        #filter_func = list(filter(lambda x: 'CI' in rec.get(x), rec))
        return all([self._valid_seq_len(feature) for feature in rec.values()])

    def get_len(self, rec):
        return rec[self._seq_len_col] if self._seq_len_col is not None else self.get_sequence_col(rec)
