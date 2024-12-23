from ptls.data_load import IterableProcessingDataset


class TakeFirstTrx(IterableProcessingDataset):
    def __init__(self, 
                 take_first_fraction: float = 0.5, 
                 seq_len_col: str = None, 
                 sequence_col: str = None):
        """
        Trim sequences by taking the first `take_first_fraction` portion of each sequence.
        The remaining part is discarded.

        Args:
            take_first_fraction: Controls the fraction of the sequence to keep.
                                 EXAMPLE: take_first_fraction=0.75 -> the first 75% of the user's
                                          transactions will be kept, and the last 25% will be discarded.
            seq_len_col:  Field where the sequence length is stored. If None, `target_col` is used.
            sequence_col: Field for detecting sequence length. If None, any iterable field will be used.
        """
        super().__init__()
        self._take_first_fraction = take_first_fraction
        self._sequence_col = sequence_col
        self._seq_len_col = seq_len_col

    def __iter__(self) -> iter:
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            seq_len = self.get_len(features)
            take_first_n = int(seq_len * self._take_first_fraction)
            for key, val in features.items():
                if self.is_seq_feature(key, val):
                    features[key] = val[:take_first_n]
            rec = (features, rec[1]) if type(rec) is tuple else features
            yield rec

    def get_len(self, rec) -> int:
        if self._seq_len_col is not None:
            return rec[self._seq_len_col]
        return len(rec[self.get_sequence_col(rec)])
