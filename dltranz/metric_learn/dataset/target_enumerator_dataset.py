import logging

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TargetEnumeratorDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        row = self.base_dataset[idx]
        data = [(x, idx) for x in row]
        return data
