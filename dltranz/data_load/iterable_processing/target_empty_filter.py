from torch.utils.data.dataset import IterableDataset
import numpy as np


class TargetEmptyFilter(IterableDataset):
    def __init__(self, target_col):
        """Drop records where value in `target_col` is undefined

        Args:
            target_col: field where `y` is stored
        """
        self._target_col = target_col

        self._src = None

    def __call__(self, src):
        self._src = src
        return self

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            y = features[self._target_col]
            if y is None:
                continue
            if type(y) is not str and np.isnan(y):
                continue
            yield rec
