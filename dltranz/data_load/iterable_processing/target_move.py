from dltranz.data_load.iterable_processing_dataset import IterableProcessingDataset


class TargetMove(IterableProcessingDataset):
    def __init__(self, target_col):
        super().__init__()
        self.target_col = target_col

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            target = int(features[self.target_col])
            yield features, target

