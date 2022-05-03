import numpy as np
import torch

from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


class FeatureFilter(IterableProcessingDataset):
    def __init__(self, keep_feature_names=None, drop_feature_names=None, drop_non_iterable=False):
        """

        Args:
            keep_feature_names: feature name for keep
        """
        super().__init__()

        if type(keep_feature_names) is str:
            keep_feature_names = [keep_feature_names]
        if type(drop_feature_names) is str:
            drop_feature_names = [drop_feature_names]

        self._keep_feature_names = set(keep_feature_names) if keep_feature_names is not None else None
        self._drop_feature_names = set(drop_feature_names) if drop_feature_names is not None else None

        self._drop_non_iterable = drop_non_iterable

    def process(self, features):
        if self._drop_non_iterable:
            features = {k: v for k, v in features.items() if type(v) in (np.ndarray, torch.Tensor)}
        if self._keep_feature_names is not None:
            features = {k: v for k, v in features.items() if k in self._keep_feature_names}
        if self._drop_feature_names is not None:
            features = {k: v for k, v in features.items() if k not in self._drop_feature_names}
        return features
