import numpy as np
import torch
from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset


class ToTorch(IterableProcessingDataset):
    """Filter that transforms each numpy.ndarray in sample dict to torch.Tensor
         
    Filter transformation:
    list({key1: int, key2: np.ndarray, ...}) -> list({key1: int, key2: torch.Tensor, ...})
    (in place of list there could be other iterable, for example generator)
    """
    def __init__(self):
        super().__init__()

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            for k, v in features.items():
                if isinstance(v, np.ndarray):
                    features[k] = torch.from_numpy(v)
            yield rec

