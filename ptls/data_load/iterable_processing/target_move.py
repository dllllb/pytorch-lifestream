from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


class TargetMove(IterableProcessingDataset):
    """Deprecated. Only single dict allowed.
    Store target as scalar value in a feature dictionary.
    Filter that takes target from sample dict and places it in a tuple with dict.
    
    Args:
        target_col (str): Name of target column in sample dict. Default is 'target'.
    Yields:
        tuple: A tuple containing the input dictionary and the target value.
    
    """
    def __init__(self, target_col: str = 'target'):
        super().__init__()
        self.target_col = target_col

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            target = int(features[self.target_col])
            yield features, target

