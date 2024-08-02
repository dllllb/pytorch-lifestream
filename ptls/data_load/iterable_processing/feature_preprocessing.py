import numpy as np
from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset

'''
class FeatureBinScaler(IterableProcessingDataset):
    def __init__(self, feature_bins: dict, idx_starts_from=0):
        """ Apply binarization by given levels

        Params
        ------
            feature_bins: keys are feature names, values is list of bins boarders
        """
        super().__init__()

        self._feature_bins = {name: np.asarray(sorted(bins)) for name, bins in feature_bins.items()}
        self._idx_starts_from = idx_starts_from

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec

            # replace values by its nearest bin idx (idx starts from self._idx_starts_from)
            for name, bins in self._feature_bins.items():
                features[name] = self.find_bin(features[name], bins) + self._idx_starts_from
            yield rec

    @staticmethod
    def find_bin(col, bins):
        idx = np.abs(col.reshape(-1, 1) - bins).argmin(axis=1)
        return idx


class FeatureFilter(IterableProcessingDataset):
    def __init__(self, keep_feature_names=None, drop_feature_names=None, drop_non_iterable=True):
        """

        Args:
            keep_feature_names: feature name for keep
        """
        super().__init__()

        keep_feature_names = [keep_feature_names] if isinstance(keep_feature_names,STR_OBJ) else keep_feature_names
        drop_feature_names = [drop_feature_names] if isinstance(drop_feature_names,STR_OBJ) else drop_feature_names

        self._keep_feature_names = set(keep_feature_names) if keep_feature_names is not None else None
        self._drop_feature_names = set(drop_feature_names) if drop_feature_names is not None else None

        self._drop_non_iterable = drop_non_iterable

    def process(self, features):
        if self._drop_feature_names is not None:
            features = {k: v for k, v in features.items()
                        if k not in self._drop_feature_names or self.is_keep(k)}
        if self._drop_non_iterable:
            features = {k: v for k, v in features.items()
                        if self.is_seq_feature(k, v) or self.is_keep(k)}
        return features

    def is_keep(self, k):
        if self._keep_feature_names is None:
            return False
        return k in self._keep_feature_names


class FeatureRename(IterableProcessingDataset):
    def __init__(self, feature_names: dict):
        """Rename features in dict

        Params:
            feature_names: keys are original names, values are target names
        """
        super().__init__()

        self._feature_names = feature_names

    def process(self, features):
        return {self._feature_names.get(k, k): v for k, v in features.items()}


class FeatureTypeCast(IterableProcessingDataset):
    def __init__(self, feature_types):
        """

        Args:
            feature_names: feature name for keep
        """
        super().__init__()

        self._feature_types = feature_types

    def process(self, features):
        return {k: self._feature_types.get(k, lambda x: x)(v)
                for k, v in features.items()}
    '''


class FeaturePreprocessing(IterableProcessingDataset):
    def __init__(self, mode, feature_bins=None, idx_starts_from=0, keep_feature_names=None,
                 drop_feature_names=None, drop_non_iterable=True, feature_names=None, feature_types=None):
        super().__init__()
        self.mode = mode

        self._feature_bins = {name: np.asarray(sorted(bins)) for name, bins in feature_bins.items()} if feature_bins else None
        self._idx_starts_from = idx_starts_from

        keep_feature_names = [keep_feature_names] if isinstance(keep_feature_names, str) else keep_feature_names
        drop_feature_names = [drop_feature_names] if isinstance(drop_feature_names, str) else drop_feature_names

        self._keep_feature_names = set(keep_feature_names) if keep_feature_names is not None else None
        self._drop_feature_names = set(drop_feature_names) if drop_feature_names is not None else None
        self._drop_non_iterable = drop_non_iterable

        self._feature_names = feature_names
        self._feature_types = feature_types

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if isinstance(rec, tuple) else rec

            if self.mode == 'FeatureBinScaler':
                for name, bins in self._feature_bins.items():
                    features[name] = self.find_bin(features[name], bins) + self._idx_starts_from
                yield rec

            elif self.mode == 'FeatureFilter':
                features = self.process_feature_filter(features)
                yield (features, rec[1]) if isinstance(rec, tuple) else features

            elif self.mode == 'FeatureRename':
                features = self.process_feature_rename(features)
                yield (features, rec[1]) if isinstance(rec, tuple) else features

            elif self.mode == 'FeatureTypeCast':
                features = self.process_feature_type_cast(features)
                yield (features, rec[1]) if isinstance(rec, tuple) else features

            else:
                raise ValueError("Unsupported mode")

    @staticmethod
    def find_bin(col, bins):
        idx = np.abs(col.reshape(-1, 1) - bins).argmin(axis=1)
        return idx

    def process_feature_filter(self, features):
        if self._drop_feature_names is not None:
            features = {k: v for k, v in features.items() if k not in self._drop_feature_names or self.is_keep(k)}
        if self._drop_non_iterable:
            features = {k: v for k, v in features.items() if self.is_seq_feature(k, v) or self.is_keep(k)}
        return features

    def is_keep(self, k):
        if self._keep_feature_names is None:
            return False
        return k in self._keep_feature_names

    def process_feature_rename(self, features):
        return {self._feature_names.get(k, k): v for k, v in features.items()}

    def process_feature_type_cast(self, features):
        return {k: self._feature_types.get(k, lambda x: x)(v) for k, v in features.items()}
