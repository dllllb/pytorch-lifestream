import numpy as np
import torch
from ptls.data_load.feature_dict import FeatureDict


class DropDay(FeatureDict):
    """
    This class is used as 'f_augmentation' argument for 
    ptls.data_load.datasets.augmentation_dataset.AugmentationDataset (AugmentationIterableDataset).
    """
    def __init__(self, event_time_name: str = 'event_time'):
        self.event_time_name = event_time_name

    def __call__(self, x: dict) -> dict:
        mask = self.get_perm_ix(x[self.event_time_name])
        new_x = self.seq_indexing(x, mask)
        return new_x

    @staticmethod
    def get_perm_ix(event_time: torch.Tensor) -> torch.Tensor:
        days = torch.unique(event_time, sorted=True)
        ix = np.random.choice(len(days), 1)[0]
        mask = event_time != days[ix]
        return mask
