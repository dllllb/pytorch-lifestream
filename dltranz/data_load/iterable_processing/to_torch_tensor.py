import numpy as np
import torch
from dltranz.data_load.iterable_processing_dataset import IterableProcessingDataset


class toTorchTensor(IterableProcessingDataset):
    def __init__(self):
        super().__init__()

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            for k, v in features.items():
                if isinstance(v, np.ndarray):
                    features[k] = torch.Tensor(v)
            yield features

