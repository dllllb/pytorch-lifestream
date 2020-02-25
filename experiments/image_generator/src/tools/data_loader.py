
import numpy as np
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset


def collate_list_fn(batch):
    return default_collate([i for row in batch for i in row])


class OnlyXDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item][0]


class MixedDataset(Dataset):
    def __init__(self, neg_dataset, pos_dataset):
        super().__init__()

        self.neg_dataset = neg_dataset
        self.pos_dataset = pos_dataset

    def __len__(self):
        return len(self.neg_dataset) + len(self.pos_dataset)

    def __getitem__(self, item):
        if item % 2 == 0:
            return self.neg_dataset[item // 2], np.zeros(1, dtype=np.float32)
        else:
            return self.pos_dataset[item // 2], np.ones(1, dtype=np.float32)


class ReplaceYDataset(Dataset):
    def __init__(self, dataset, y=0):
        super().__init__()

        self.dataset = dataset
        self.y = y

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item][0], self.y


class FlattenMLDataset(Dataset):
    def __init__(self, dataset, y=0):
        super().__init__()

        self.dataset = dataset
        self.y = y

        self.item_len = len(self.dataset[0])

    def __len__(self):
        return len(self.dataset) * self.item_len

    def __getitem__(self, item):
        return self.dataset[item // self.item_len][item % self.item_len][0], self.y
