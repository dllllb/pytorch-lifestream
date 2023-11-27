from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset
import numpy as np
import torch

class DeleteNan(IterableProcessingDataset):
    def __init__(self):
        """This class is used for deletion None values in sources, where was no transaction."""
        super().__init__()

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec

            for name, value in features.items():
                if value is None:
                    features[name] = torch.Tensor([])
            yield rec