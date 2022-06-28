from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


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
