from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


class TargetExtractor(IterableProcessingDataset):
    def __init__(self, target_col, drop_from_features=True):
        """Extract value from `target_col` and mention it as `y`

        for x, * in seq:
            y = x[target_col]
            yield x, y

        Args:
            target_col: field where `y` is stored

        """
        super().__init__()

        self._target_col = target_col
        self._drop_from_features = drop_from_features

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            y = features[self._target_col]
            if self._drop_from_features:
                features = {k: v for k, v in features.items() if k != self._target_col}
            yield features, y


class FakeTarget(IterableProcessingDataset):
    def __init__(self):
        """Create target equal 0 (for consistency)

        for x in seq:
            yield x, 0

        """
        super().__init__()

    def __iter__(self):
        for rec in self._src:
            yield rec, 0
