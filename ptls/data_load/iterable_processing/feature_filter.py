from typing import Optional, Set, Union

from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset
from collections import defaultdict

class FeatureFilter(IterableProcessingDataset):
    """
    Filter features by name. Keep only features with names from keep_feature_names.
    Drop features with names from drop_feature_names.
    Drop non-iterable features if drop_non_iterable is True.

    Args:
        keep_feature_names: feature names to keep
        drop_feature_names: feature names to drop
        drop_non_iterable: whether to drop non-iterable features
    """

    def __init__(self,
                 keep_feature_names: Optional[Union[str, Set[str]]] = None,
                 drop_feature_names: Optional[Union[str, Set[str]]] = None,
                 drop_non_iterable: bool = True
                 ):
        super().__init__()

        self._keep_feature_names = self._to_set(keep_feature_names)
        self._drop_feature_names = self._to_set(drop_feature_names)
        self._drop_non_iterable = drop_non_iterable

    def _to_set(self, feature_names: Optional[Union[str, Set[str]]]) -> Optional[Set[str]]:
        """Helper method to convert input to a set."""
        if feature_names is None:
            return None
        if isinstance(feature_names, str):
            return {feature_names}
        return set(feature_names)

    def process(self, features: dict) -> dict:
        for name in self._drop_feature_names:
            features.pop(name, None)

        if self._drop_non_iterable:
            return {name: val for name, val in features.items() if self.is_seq_feature(name, val) or self.is_keep(name)}

        return features

    def is_keep(self, k: str) -> bool:
        """Check if the feature name should be kept."""
        return self._keep_feature_names is None or k in self._keep_feature_names
