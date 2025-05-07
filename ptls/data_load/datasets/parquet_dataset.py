import logging
import os
import warnings
from glob import glob
from itertools import chain
from typing import Union, List

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from omegaconf import ListConfig

from ptls.data_load import read_pyarrow_file
from ptls.data_load.utils import init_worker

logger = logging.getLogger(__name__)


def iter_with_max_num(iterated, max_num: int = None):
    num = 0
    for i in iterated:
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
    """
    Lazy (IterableDataset) load from parquet files.
    File structure: *.parquet.
    Each file is read sequentially.

    File structure example:
        data/
            part1.parquet
            part2.parquet
            ...

    Args:
        data_files: ParquetFile object with list of files or just list of files
        i_filters: list of `ptls.data_load.iterable_processing` filters
        shuffle_files: shuffle data_files before reading when True.
        cache_schema: dict schema (feature names) will be read once
        shuffle_seed: random seed for shuffle_files

    """

    def __init__(self, 
                 data_files: Union[ParquetFiles, List[str]],
                 i_filters: List = None,
                 shuffle_files: bool = False, 
                 cache_schema: bool =True, 
                 shuffle_seed: int = 42
                 ):
        is_parquet = isinstance(data_files, ParquetFiles)
        self.data_files = data_files.data_files if is_parquet else data_files
        self.postprocessing_func = i_filters
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

    def __apply_postproc(self, sample):
        for func in self.postprocessing_func:
            sample = func(sample)
        return sample

    def __iter__(self):
        init_worker(self)
        my_files = self._get_my_files()
        if self.shuffle_files:
            rs = np.random.RandomState(self._shuffle_seed % 2**32)
            rs.shuffle(my_files)
        logger.debug(f'Iter [{self._worker_id:02d}/{self._num_workers:02d}]: {my_files}')
        gen = chain(*[self.iter_file(name) for name in my_files])
        if self.postprocessing_func is not None:
            gen = self.__apply_postproc(gen)
        return gen
    
    def iter_file(self, file_name):
        """
        Iterates over parquet file

        Args:
            file_name: parquet file name

        Returns:
            [(customer_id, features)]
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
    """
    Modification of ParquetDataset for working with DDP. Each GPU processes has its own set of files.
    Make sure that number of parquet files > number of GPUs

    Args:
        data_files: ParquetFile object with list of files or just list of files
        i_filters: list of `ptls.data_load.iterable_processing` filters
        shuffle_files: shuffle data_files before reading when True.
        cache_schema: dict schema (feature names) will be read once
        shuffle_seed: random seed for shuffle_files
        max_items_per_file: if passed, worker reads max_items_per_file rows from parquet file to ensure equal amount
                            of examples for different GPUs, else this quantity is calculated before training which take
                            significant amount of time. You should calculate it by yourself by counting minimal number
                            of rows in all parquet files
        repeat_items: whether to start reading same files again on the worker for preventing deadlocks (caused by
                      inability to calculate exact number of yielded items per worker and as a result inability to
                      yield the same number of items on different GPUs)

    """

    def __init__(self, 
                 data_files: Union[ParquetFiles, List[str]],
                 i_filters: List = None,
                 shuffle_files: bool = False, 
                 cache_schema: bool = True, 
                 shuffle_seed: int = 42, 
                 max_items_per_file: int = None, 
                 repeat_items: bool = True):
        super().__init__(data_files=data_files,
                         i_filters=i_filters,
                         shuffle_files=shuffle_files, 
                         cache_schema=cache_schema, 
                         shuffle_seed=shuffle_seed)
        self.max_items_per_file = max_items_per_file
        self.items_per_worker = None
        self.repeat_items = repeat_items
        self.real_worker_id = None
        self.real_num_workers = None

    def _calc_min_items_per_worker(self):
        nums = []
        for rank in range(dist.get_world_size()):
            per_gpu = 0
            for filename in self.data_files[rank::dist.get_world_size()]:
                if self.max_items_per_file is not None:
                    per_gpu += self.max_items_per_file
                else:
                    per_gpu += pq.read_table(filename).shape[0]
            nums.append(per_gpu)
        return min(nums) // self._num_workers

    def _get_my_files(self):
        my_files = [name for i, name in enumerate(self.data_files) if
                    i % self.real_num_workers == self.real_worker_id]
        return my_files

    def _init_worker(self):

        init_worker(self)
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
        if self.postprocessing_func is not None:
            gen = self.__apply_postproc(gen)
        return iter_with_max_num(gen, self.items_per_worker)


def read_pyarrow_file(path, use_threads=True):
    p_table = pq.read_table(
        source=path,
        use_threads=use_threads,
    )

    col_indexes = p_table.column_names

    def get_records():
        for rb in p_table.to_batches():
            col_arrays = [rb.column(i) for i, _ in enumerate(col_indexes)]
            col_arrays = [a.to_numpy(zero_copy_only=False) for a in col_arrays]  # Keep as NumPy arrays

            # Pre-allocate dictionary
            records = [{} for _ in range(len(col_arrays[0]))]

            if 'trans_time' in col_indexes:
                date_batch = col_arrays[col_indexes.index('trans_time')].astype('datetime64[s]')
                records = [
                    {
                        'local_day': (date_batch[i].astype('datetime64[D]') - date_batch[i]).astype(
                            'datetime64[M]').astype(np.int16) + 1,
                        'local_month': date_batch[i].astype('datetime64[M]').astype(np.int16) / 12 + 1,
                        'local_weekday': (date_batch[i].astype('datetime64[D]').astype(np.int16) + 3) / 7,
                        'hour': ((date_batch[i] - date_batch[i].astype('datetime64[D]')).astype(
                            np.int32) // 3600 + 1).astype(np.int16),
                        **{n: a[i] for n, a in zip(col_indexes, col_arrays) if n != 'trans_time'}
                    }
                    for i in range(len(date_batch))
                ]
            else:
                for i in range(len(col_arrays[0])):
                    records[i] = {n: a[i] for n, a in zip(col_indexes, col_arrays)}

            yield from records

    return get_records()
