import torch

from ptls.data_load import AugmentationChain


class AugmentationDataset(torch.utils.data.Dataset):
    def __init__(self, data, f_augmentations: list = None):
        self.data = data
        self.f_augmentations = AugmentationChain(f_augmentations)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        rec = self.data[item]
        return self.f_augmentations(rec)
    

class AugmentationIterableDataset(torch.utils.data.IterableDataset):
    """
    AugmentationDataset for IterableDataset.
    'data' argument must be `torch.utils.data.IterableDataset`

    """
    def __init__(self, data, f_augmentations: list = None):
        assert isinstance(data, torch.utils.data.IterableDataset)
        self.data = data
        self.f_augmentations = AugmentationChain(f_augmentations)
    
    def __iter__(self):
        for rec in self.data:
            yield self.f_augmentations(rec)
