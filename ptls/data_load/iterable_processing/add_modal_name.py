from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


class AddModalName(IterableProcessingDataset):
    """
    Add prefix to feature names.

    Args:
        cols: list of feature names
        source: prefix to add
    """

    def __init__(self, cols, source):
        super().__init__()
        self._cols = cols
        self._source = source

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            for feature in self._cols:
                if feature in features:
                    features[self._source + '_' + feature] = features[feature]
                    del features[feature]
            yield rec
