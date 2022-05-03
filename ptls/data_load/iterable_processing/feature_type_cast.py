from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


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
