from torch.utils.data import Dataset


class MapAugmentationDataset(Dataset):
    def __init__(self, base_dataset, a_chain=None):
        self.base_dataset = base_dataset
        self.a_chain = a_chain

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        feature_arrays, target = self.base_dataset[idx]
        if self.a_chain is not None:
            feature_arrays = self.a_chain(feature_arrays)
        return feature_arrays, target
