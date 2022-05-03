import torch
import logging
import numpy as np

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

    def _init_worker(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            self._worker_id = 0
            self._num_workers = 1
            self._shuffle_seed = self.shuffle_seed
        else:  # in a worker process
            self._worker_id = worker_info.id
            self._num_workers = worker_info.num_workers
            self._shuffle_seed = worker_info.seed
        logger.debug(f'Started [{self._worker_id:02d}/{self._num_workers:02d}]')

    def _get_gen(self, my_ids):
        for ind in my_ids:
            rec = self.base_dataset[ind]
            rec = {k: self.to_torch(v) for k, v in rec.items()}
            yield rec

    def __iter__(self):
        self._init_worker()

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
