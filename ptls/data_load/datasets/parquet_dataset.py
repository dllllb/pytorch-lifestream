import logging
import os
import warnings
from glob import glob
from itertools import chain
from typing import Union, List

import numpy as np
import torch
import torch.distributed as dist
import pyarrow.parquet as pq
from omegaconf import ListConfig

from ptls.data_load import read_pyarrow_file
from ptls.data_load import IterableChain

logger = logging.getLogger(__name__)

def iter_with_max_num (iter, max_num = None):
    num = 0
    for i in iter:
        yield i
        num += 1
        if max_num is not None and num >= max_num:
            if dist.is_initialized():
                print("STOPPING WORKER on GPU", dist.get_rank())
            break


class ParquetFiles:
    """Helper file which search parquet files in specified path.

    Spark saves parquet files in a partitioned.
    This means that the output name given when saving is not a file, but a directory.
    `ParquetDataset` required file names so
    `ParquetFiles` is an adapter to scan spark output path and find parquet files.

    One single parquet file taken for example from pandas will be passed as is without directory scan.

    Parameters
    ----------
    file_path:
        one path or list of paths
    take_ixes:
        indexes for final file list selecting.
        Be careful using `take_ixes` cause file list is unsorted

    """
    def __init__(self, file_path: Union[str, List[str], ListConfig], take_ixes=None):
        if type(file_path) not in (list, ListConfig):
            file_path = [file_path]

        self._data_files = []
        for path in file_path:
            self._data_files.extend(self.scan_path(path))

        if take_ixes is not None:
            warnings.warn('Be careful using `take_ixes` with unsorted data file list')
            self._data_files = np.array(self._data_files)[take_ixes].tolist()

    @staticmethod
    def scan_path(path):
        if os.path.isfile(path):
            return [path]
        if os.path.isdir(path):
            file_path = os.path.join(path, '*.parquet')
            return glob(file_path)
        return []

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

    Each file is read sequentially

    Parameters
    ----------
    data_files:
        ParquetFile object with list of files or just list of files
    post_processing:
        - deprecated, use i_filters
    i_filters:
        - list of `ptls.data_load.iterable_processing` filters
    shuffle_files:
        - shuffle data_files before reading when True.
    cache_schema:
        - dict schema (feature names) will be read once
    shuffle_seed:
        - random seed for shuffle_files

    """
    def __init__(self, data_files: Union[ParquetFiles, List[str]],
                 post_processing=None,
                 i_filters: List = None,
                 shuffle_files=False, cache_schema=True, shuffle_seed=42):
        if type(data_files) is ParquetFiles:
            self.data_files = data_files.data_files
        else:
            self.data_files = data_files
        if i_filters is not None:
            self.post_processing = IterableChain(*i_filters)
        else:
            self.post_processing = post_processing
        if post_processing is not None:
            warnings.warn('`post_processing` parameter is deprecated, use `i_filters`')
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

class DistributedParquetDataset(ParquetDataset):
    """Modification of ParquetDataset for working with DDP
    Each GPU processes its own set of files
    Make sure that number of parquet files > number of GPUs

    max_items_per_file is used to ensure that dataloader on different GPUs produce the same amount of batches
    it's better to calculate it manually (minimal number of rows in parquet files) before training


    Parameters
    ----------
    data_files:
        ParquetFile object with list of files or just list of files
    post_processing:
        - deprecated, use i_filters
    i_filters:
        - list of `ptls.data_load.iterable_processing` filters
    shuffle_files:
        - shuffle data_files before reading when True.
    cache_schema:
        - dict schema (feature names) will be read once
    shuffle_seed:
        - random seed for shuffle_files
    max_items_per_file:
        - if passed, worker reads max_items_per_file rows from parquet file to ensure equal amount of examples for different GPUs, 
        else this quantity is calculated before training which take significant amount of time. 
        You should calculate it by yourself by counting minimal number of rows in all parquet files
    repeat_items:
        - whether to start reading same files again on the worker for preventing deadlocks 
        (caused by inability to calculate exact number of yielded items per worker 
        and as a result inability to yield the same number of items on different GPUs)

    """
    def __init__(self, data_files: Union[ParquetFiles, List[str]],
                 post_processing=None,
                 i_filters: List = None,
                 shuffle_files=False, cache_schema=True, shuffle_seed=42, max_items_per_file = None, repeat_items = True):
        super().__init__(data_files = data_files,
                 post_processing=post_processing,
                 i_filters = i_filters,
                 shuffle_files=shuffle_files, cache_schema=cache_schema, shuffle_seed=shuffle_seed)
        self.max_items_per_file = max_items_per_file
        self.items_per_worker = None
        self.repeat_items = repeat_items

    def _calc_min_items_per_worker(self):
        nums = []
        for rank in range(dist.get_world_size()):
            per_gpu = 0
            for fname in self.data_files[rank::dist.get_world_size()]:
                if self.max_items_per_file is not None:
                    per_gpu += self.max_items_per_file
                else:
                    per_gpu += pq.read_table(fname).shape[0]
            nums.append(per_gpu)
        return min(nums) // self._num_workers

    def _get_my_files(self):
        my_files = [name for i, name in enumerate(sorted(self.data_files)) if i % self.real_num_workers == self.real_worker_id]
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

        self.real_worker_id = self._worker_id
        self.real_num_workers = self._num_workers
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            self.real_num_workers = self._num_workers * world_size
            self.real_worker_id = rank + self._worker_id * world_size
        logger.debug(f'Started [{self.real_worker_id:02d}/{self.real_num_workers:02d}]')

    def __iter__(self):
        self._init_worker()
        if dist.is_initialized() and self.items_per_worker is None:
            self.items_per_worker = self._calc_min_items_per_worker()

        my_files = self._get_my_files()
        if self.shuffle_files:
            rs = np.random.RandomState(self._shuffle_seed % 2**32)
            rs.shuffle(my_files)

        logger.debug(f'Iter [{self._worker_id:02d}/{self._num_workers:02d}]: {my_files}')
        if self.repeat_items:
            gen = chain(*[self.iter_file(name) for _ in range(2) for name in my_files])
        else:
            gen = chain(*[self.iter_file(name) for name in my_files])

        if self.post_processing is not None:
            gen = self.post_processing(gen)
        return iter_with_max_num(gen, self.items_per_worker) 