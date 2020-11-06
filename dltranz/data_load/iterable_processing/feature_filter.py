import numpy as np
import torch

from dltranz.data_load.iterable_processing_dataset import IterableProcessingDataset


class FeatureFilter(IterableProcessingDataset):
    def __init__(self, feature_names=None, drop_non_iterable=False):
        """

        Args:
            feature_names: feature name for keep
        """
        super().__init__()

        self._feature_names = set(feature_names) if feature_names is not None else None
        self._drop_non_iterable = drop_non_iterable

    def process(self, features):
        if self._drop_non_iterable:
            return {k: v for k, v in features.items() if type(v) in (np.ndarray, torch.Tensor)}
        if self._feature_names is not None:
            return {k: v for k, v in features.items() if k in self._feature_names}
