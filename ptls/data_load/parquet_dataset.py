from itertools import chain
import os

from glob import glob
import torch
import logging

import numpy as np

from ptls.data_load import read_pyarrow_file

logger = logging.getLogger(__name__)


class ParquetFiles:
    def __init__(self, file_path, take_ixes=None):
        #if os.path.splitext(file_path)[1] == '.parquet':
        file_path = os.path.join(file_path, '*.parquet')
        self._data_files = glob(file_path)
        if (take_ixes):
            self._data_files = np.array(self._data_files)[take_ixes].tolist()

    @property
    def data_files(self):
        return self._data_files


class ParquetDataset(torch.utils.data.IterableDataset):
    """Lazy (IterableDataset) load from parquet files

    File structure:
    *.parquet

    File structure example:
    data/
        part1.parquet
        part2.parquet
        ...

    """
    def __init__(self, data_files,
                 post_processing=None,
                 shuffle_files=False, cache_schema=True, shuffle_seed=42):
        self.data_files = data_files
        self.post_processing = post_processing
        self.shuffle_files = shuffle_files
        self.cache_schema = cache_schema
        self.shuffle_seed = shuffle_seed
        self.rs = None

        self._worker_id = None
        self._num_workers = None
        self._shuffle_seed = None
        self._schema = None

    def _get_my_files(self):
        files = [(i, name) for i, name in enumerate(self.data_files)]
        my_files = [name for i, name in files if i % self._num_workers == self._worker_id]

        return my_files

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

    def __iter__(self):
        self._init_worker()

        my_files = self._get_my_files()
        if self.shuffle_files:
            rs = np.random.RandomState(self._shuffle_seed % 2**32)
            rs.shuffle(my_files)

        logger.debug(f'Iter [{self._worker_id:02d}/{self._num_workers:02d}]: {my_files}')
        gen = chain(*[self.iter_file(name) for name in my_files])
        if self.post_processing is not None:
            gen = self.post_processing(gen)
        return gen

    def iter_file(self, file_name):
        """

        :param file_name:
        :return: [(customer_id, features)]
        """
        logger.debug(f'[{self._worker_id}/{self._num_workers}] Iter file "{file_name}"')
        for rec in read_pyarrow_file(file_name, use_threads=True):
            rec = {k: self.to_torch(v) for k, v in rec.items()}
            yield rec

    @staticmethod
    def to_torch(x):
        if type(x) is np.ndarray and x.dtype.kind in ('i', 'f'):
            return torch.from_numpy(x)
        return x
