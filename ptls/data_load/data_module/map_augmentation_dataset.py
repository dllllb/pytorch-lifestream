import warnings

from torch.utils.data import Dataset


class MapAugmentationDataset(Dataset):
    """
    This class is used as 'f_augmentation' argument for
    ptls.data_load.datasets.augmentation_dataset.AugmentationDataset (AugmentationIterableDataset).

    Args:
        base_dataset: dataset to apply augmentation to
        a_chain: augmentation chain to apply to the dataset

    """
    def __init__(self, 
                 base_dataset: Dataset, 
                 a_chain: callable = None):
        warnings.warn('Use `ptls.data_load.datasets.AugmentationDataset`', DeprecationWarning)
        self.base_dataset = base_dataset
        self.a_chain = a_chain

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple:
        row = self.base_dataset[idx]
        if type(row) is tuple:
            feature_arrays, target = row
            if self.a_chain is not None:
                feature_arrays = self.a_chain(feature_arrays)
            return feature_arrays, target
        else:
            feature_arrays = row
            if self.a_chain is not None:
                feature_arrays = self.a_chain(feature_arrays)
            return feature_arrays
