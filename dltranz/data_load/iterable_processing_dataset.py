from torch.utils.data.dataset import IterableDataset


class IterableProcessingDataset(IterableDataset):
    def __init__(self):
        self._src = None

    def __call__(self, src):
        self._src = src
        return iter(self)

    def __iter__(self):
        """For record transformation. Redefine __iter__ for filter"""
        for rec in self._src:
            if type(rec) is tuple:
                features = rec[0]
                new_features = self.process(features)
                yield tuple([new_features, *rec[1:]])
            else:
                features = rec
                new_features = self.process(features)
                yield new_features

    def process(self, features):
        raise NotImplementedError()
