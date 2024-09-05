from typing import Union

from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


class FeatureFilter(IterableProcessingDataset):
    """
    Filter features by name. Keep only features with names from keep_feature_names.
    Drop features with names from drop_feature_names.
    Drop non-iterable features if drop_non_iterable is True.

    Args:
        keep_feature_names: feature name for keep
        drop_feature_names: feature name for drop
        drop_non_iterable: drop non-iterable features

    """

    def __init__(self,
                 keep_feature_names: Union[str, list] = (),
                 drop_feature_names: Union[str, list] = (),
                 drop_non_iterable: bool = True
                 ):
        super().__init__()

        if isinstance(keep_feature_names, str):
            keep_feature_names = [keep_feature_names]
        if isinstance(drop_feature_names, str):
            drop_feature_names = [drop_feature_names]

        self._keep_feature_names = set(keep_feature_names) if keep_feature_names is not None else None
        self._drop_feature_names = set(drop_feature_names) if drop_feature_names is not None else None

        self._drop_non_iterable = drop_non_iterable

    def process(self, features: dict) -> dict:

        for name in self._drop_feature_names:
            if name not in self._keep_feature_names:
                features.pop(name, None)

        if self._drop_non_iterable:
            return {name: val for name, val in features.items() if self.is_seq_feature(name, val) or self.is_keep(name)}
        return features

    def is_keep(self, k: str) -> bool:
        if self._keep_feature_names is None:
            return False
        return k in self._keep_feature_names
