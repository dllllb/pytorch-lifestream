import logging

import numpy as np
import torch

from ptls.data_load.utils import init_worker

logger = logging.getLogger(__name__)


class FilterDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset,
                 post_processing=None,
                 shuffle_files=False,
                 shuffle_seed=42):

        self.base_dataset = dataset

        self.post_processing = post_processing
        self.shuffle_files = shuffle_files
        self.shuffle_seed = shuffle_seed
        self.rs = None

        self._worker_id = None
        self._num_workers = None
        self._shuffle_seed = None
        self._schema = None

    def _get_my_ids(self):
        return list(range(self._worker_id, len(self.base_dataset), self._num_workers))

    def _get_gen(self, my_ids):
        for ind in my_ids:
            rec = self.base_dataset[ind]
            rec = {k: self.to_torch(v) for k, v in rec.items()}
            yield rec

    def __iter__(self):
        init_worker(self)

        my_ids = self._get_my_ids()
        if self.shuffle_files:
            rs = np.random.RandomState(self._shuffle_seed % 2**32)
            rs.shuffle(my_ids)

        gen = self._get_gen(my_ids)
        if self.post_processing is not None:
            gen = self.post_processing(gen)
        return gen

    @staticmethod
    def to_torch(x):
        if type(x) is np.ndarray and x.dtype.kind in ('i', 'f'):
            return torch.from_numpy(x)
        return x
