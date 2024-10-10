from torch.utils.data.dataset import IterableDataset
from ptls.data_load.feature_dict import FeatureDict


class IterableProcessingDataset(FeatureDict, IterableDataset):
    def __init__(self):
        super().__init__()
        self._src = None
        self._sequence_col = None

    def __call__(self, src):
        self._src = src
        return iter(self)

    def __iter__(self):
        """
        For record transformation. Redefine __iter__ for filter
        """
        for rec in self._src:
            if isinstance(rec, tuple):
                features = rec[0]
                new_features = self.process(features)
                yield tuple([new_features, *rec[1:]])
            else:
                features = rec
                new_features = self.process(features)
                yield new_features

    def process(self, features):
        raise NotImplementedError()

    def get_sequence_col(self, rec):
        if self._sequence_col is None:
            arrays = [k for k, v in rec.items() if self.is_seq_feature(k, v)]
            if len(arrays) == 0:
                raise ValueError(f'Can not find field with sequence from record: {rec}')
            self._sequence_col = arrays[0]
        return self._sequence_col
