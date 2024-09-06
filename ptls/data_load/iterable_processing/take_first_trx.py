from ptls.data_load import IterableProcessingDataset


class TakeFirstTrx(IterableProcessingDataset):
    def __init__(self, take_first_fraction=0.5, seq_len_col=None, sequence_col=None):
        """
        Filter sequences by length. Drop sequences shorter than `min_seq_len` and longer than `max_seq_len`.

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
                if self.is_seq_feature(key, val):
                    features[key] = val[:take_first_n]
            rec = (features, rec[1]) if type(rec) is tuple else features
            yield rec

    def get_len(self, rec):
        if self._seq_len_col is not None:
            return rec[self._seq_len_col]
        return len(rec[self.get_sequence_col(rec)])
