import numpy as np

from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


class FeatureBinScaler(IterableProcessingDataset):
    def __init__(self, feature_bins: dict, idx_starts_from=0):
        """ Apply binarization by given levels

        Params
        ------
            feature_bins: keys are feature names, values is list of bins boarders
        """
        super().__init__()

        self._feature_bins = {name:np.asarray(sorted(bins)) for name, bins in feature_bins.items()}
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