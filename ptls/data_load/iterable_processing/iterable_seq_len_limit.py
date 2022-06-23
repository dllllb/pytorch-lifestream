from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset
from ptls.data_load.augmentations.seq_len_limit import SeqLenLimit


class ISeqLenLimit(IterableProcessingDataset):
    def __init__(self, max_seq_len, strategy='tail'):
        super().__init__()

        self.proc = SeqLenLimit(max_seq_len, strategy)

    def process(self, features):
        return self.proc(features)
