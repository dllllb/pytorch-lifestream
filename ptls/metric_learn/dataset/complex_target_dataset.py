import logging

from functools import reduce
from operator import add
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ComplexTargetDataset(Dataset):
    def __init__(self, base_dataset, split_counts):
        self.base_dataset = base_dataset
        self.split_counts = split_counts

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        row = self.base_dataset[idx]
        splitter_numbers = [[i]*split_count for i, split_count in enumerate(self.split_counts)]
        splitter_numbers = reduce(add, splitter_numbers)
        data = [(x, (n, idx)) for x, n in zip(row, splitter_numbers)]
        return data
