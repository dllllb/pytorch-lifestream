import numpy as np

from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


class UnifiedProcessingDataset(IterableProcessingDataset):
    def __init__(self, mode: str, 
                 target_col: str = None, 
                 drop_from_features: bool = True, 
                 id_col: str = None, 
                 target_values: dict = None,
                 func=int):
        super().__init__()
        self.mode = mode
        self._target_col = target_col
        self._drop_from_features = drop_from_features
        self._id_col = id_col
        self._target_values = target_values
        self.func = func

    def __iter__(self):
        for rec in self._src:
            if self.mode == 'TargetEmptyFilter':
                features = rec[0] if isinstance(rec, tuple) else rec
                y = features[self._target_col]
                if y is None or (not isinstance(y, str) and np.isnan(y)):
                    continue
                yield rec

            elif self.mode == 'TargetExtractor':
                features = rec[0] if isinstance(rec, tuple) else rec
                y = features[self._target_col]
                if self._drop_from_features:
                    features = {k: v for k, v in features.items() if k != self._target_col}
                yield features, y

            elif self.mode == 'FakeTarget':
                yield rec, 0

            elif self.mode == 'TargetJoin':
                features = rec[0] if isinstance(rec, tuple) else rec
                _id = features[self._id_col]
                y = self.func(self._target_values[_id])
                yield features, y

            elif self.mode == 'TargetMove':
                features = rec[0] if isinstance(rec, tuple) else rec
                target = int(features[self._target_col])
                yield features, target
            else:
                raise ValueError("Unsupported mode")