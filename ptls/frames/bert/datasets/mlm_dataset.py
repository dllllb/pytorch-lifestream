import torch

from ptls.data_load import padded_collate_wo_target
from ptls.data_load.augmentations.random_slice import RandomSlice


class MlmDataset(torch.utils.date.Dataset):
    """

    Parameters
    ----------
    data:
        List with all sequences
    min_len:
        RandomSlice params.
    max_len:
        RandomSlice params.
    rate_for_min:
        RandomSlice params.
    """
    def __init__(self,
                 data,
                 min_len, max_len, rate_for_min: float = 1.0,
                 *args, **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.data = data
        self.r_slice = RandomSlice(min_len, max_len, rate_for_min)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        feature_arrays = self.data[item]
        return self.r_slice(feature_arrays)

    def __iter__(self):
        for feature_arrays in self.data:
            yield self.r_slice(feature_arrays)

    @staticmethod
    def collate_fn(batch):
        return padded_collate_wo_target(batch)


class MlmIterableDataset(MlmDataset, torch.utils.data.IterableDataset):
    pass
