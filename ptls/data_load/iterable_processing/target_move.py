from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


class TargetMove(IterableProcessingDataset):
    """Filter that takes target from sample dict and place it in tuple with dict
    Parameters
     ----------
    target_col : str. Default: 'target'
         Name of target columns in sample dict.
         
    Filter transformation:
    list(input_dict) -> list((input_dict, target_val))
    (in place of list there could be other iterable, for example generator)
    """
    def __init__(self, target_col='target'):
        super().__init__()
        self.target_col = target_col

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            target = int(features[self.target_col])
            yield features, target

