import numpy as np
import torch
from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset

class FilterNonArray(IterableProcessingDataset):
    def __init__(self):
        super().__init__()

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            to_del = []
            for k, v in features.items():
                if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                    continue
                else:
                    to_del.append(k)

            for k in to_del:
                del features[k]
            yield rec
