import logging
import warnings
from typing import List

import numpy as np
import torch

from ptls.data_load import IterableChain

logger = logging.getLogger(__name__)


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self,
                 clients,
                 seq_len=512,
                 post_processing=None,
                 i_filters: List = None):
        self.clients = clients
        self.seq_len = seq_len
        if i_filters is not None:
            self.post_processing = IterableChain(*i_filters)
        else:
            self.post_processing = post_processing
        if post_processing is not None:
            warnings.warn('`post_processing` parameter is deprecated, use `i_filters`')

    def __getitem__(self, ind):
        item = self.clients[ind].gen_seq(self.seq_len)
        if self.post_processing is not None:
            item = self.post_processing(item)
        return item

    def __len__(self):
        return len(self.clients)

    @staticmethod
    def to_torch(x):
        if type(x) is np.ndarray and x.dtype.kind in ('i', 'f'):
            return torch.from_numpy(x)
        return x
