import torch
import logging

logger = logging.getLogger(__name__)


class PersistDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = [rec for rec in data]
        logger.info(f'Loaded {len(self.data)} records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
