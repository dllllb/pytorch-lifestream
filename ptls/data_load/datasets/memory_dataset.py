import logging
import warnings

import torch

from ptls.data_load import IterableChain
from ptls.data_load.iterable_processing.to_torch_tensor import ToTorch

logger = logging.getLogger(__name__)


class MemoryMapDataset(torch.utils.data.Dataset):
    def __init__(self, data, i_filters=None):
        if i_filters is None:
            self.processed_data = [rec for rec in data]
        else:
            post_processor_filter = IterableChain(*i_filters)
            self.processed_data = [rec for rec in post_processor_filter(data)]
        logger.info(f'Loaded {len(self.processed_data)} records')

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, item):
        return self.processed_data[item]


class MemoryIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, data, i_filters=None):
        raise NotImplementedError()
        """Multiprocessing case aren't defined
        """
        if i_filters is None:
            i_filters = []
        self.data = data
        self.post_processor_filter = IterableChain(ToTorch(), *i_filters)

    def __iter__(self):
        for rec in self.post_processor_filter(self.data):
            yield rec

