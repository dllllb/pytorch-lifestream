import numpy as np

from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


class TargetEmptyFilter(IterableProcessingDataset):
    def __init__(self, target_col):
        """Drop records where value in `target_col` is undefined

        Args:
            target_col: field where `y` is stored
        """
        super().__init__()

        self._target_col = target_col

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            y = features[self._target_col]
            if y is None:
                continue
            if type(y) is not str and np.isnan(y):
                continue
            yield rec
