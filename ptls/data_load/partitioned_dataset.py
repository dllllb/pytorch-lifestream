import logging
import os
from collections import defaultdict, Counter
from itertools import chain

import numpy as np
import torch
from torch.utils.data import DataLoader

from ptls.data_load import read_pyarrow_file

from pydoc import locate

logger = logging.getLogger(__name__)


class PartitionedDataFiles:
    """PartitionedDataFiles take a structure from partitioned parquet file.
    Structure is an indexes of main partitions (date) and subpartitions (hashes)

    File structure:
    mon=YYYY-MM-DD/
        hash_id=xxx/
            *.parquet

    File structure example:
    data/
        mon=2014-01-31/
            hash_id=1/
                part1.parquet
                part2.parquet
                ...
            hash_id=2/
                part1.parquet
                part2.parquet
                ...
            ...

        mon=2014-02-28/
            hash_id=1/
                ...
            hash_id=2/
                ...


    Data is a matrix: `mon` x `hash_id`

    `hash_id` is a value of `customer_id`. So if we have to collect all data for specific `customer_id`,
    we should calculate `hash_id = H` and get data from all month from `hash_id=H` subpartitions.
    """

    def __init__(self, data_path, dt_start=None, dt_end=None, dt_dtype='datetime64'):
        self._data_path = data_path
        self.to_dtype = locate(dt_dtype)
        self.dt_start = self.to_dtype(dt_start) if dt_start else None
        self.dt_end = self.to_dtype(dt_end) if dt_end else None

        self._dt_parts = None
        self._hash_parts = None

        self._index_files()

    @property
    def data_path(self):
        return self._data_path

    @property
    def dt_parts(self):
        return self._dt_parts

    @property
    def hash_parts(self):
        return self._hash_parts

    def _index_files(self):
        dt_parts = os.listdir(self._data_path)
        dt_parts = [x for x in dt_parts if x.find('=') > 1]
        dt_dates = np.array([x.split('=')[1] for x in dt_parts]).astype(self.to_dtype)

        dt_dates_sort_ix = np.argsort(dt_dates)
        dt_parts = [dt_parts[i] for i in dt_dates_sort_ix]
        dt_dates = dt_dates[dt_dates_sort_ix]

        mask = np.ones_like(dt_dates, dtype=np.bool)
        if self.dt_start is not None:
            mask &= dt_dates >= self.dt_start
        if self.dt_end is not None:
            mask &= dt_dates <= self.dt_end
        self._dt_parts = [x for x, f in zip(dt_parts, mask) if f]

        if len(self._dt_parts) == 0:
            raise AttributeError(f'Empty file list for "{self._data_path}" '
                                 f'in interval [{self.dt_start}, {self.dt_end}]')

        hash_parts = sorted(os.listdir(os.path.join(self._data_path, self._dt_parts[0])))
        for dt_part in self._dt_parts[1:]:
            hash_part_2 = sorted(os.listdir(os.path.join(self._data_path, dt_part)))
            if hash_part_2 != hash_parts:
                raise AttributeError(f'Hash sets in "{self._dt_parts[0]}" and "{dt_part}" are different')

        self._hash_parts = hash_parts


class PartitionedDataset(torch.utils.data.IterableDataset):
    """Lazy (IterableDataset) load from partitioned parquet file

    File structure:
    mon=YYYY-MM-DD/
        hash_id=xxx/
            *.parquet

    File structure example:
    data/
        mon=2014-01-31/
            hash_id=1/
                part1.parquet
                part2.parquet
                ...
            hash_id=2/
                part1.parquet
                part2.parquet
                ...
            ...

        mon=2014-02-28/
            hash_id=1/
                ...
            hash_id=2/
                ...


    Data is a matrix: `mon` x `hash_id`

    `hash_id` is a value of `customer_id`. So if we have to collect all data for specific `customer_id`,
    we should calculate `hash_id = H` and get data from all month from `hash_id=H` subpartitions.
    """
    def __init__(self, data_path, dt_parts, hash_parts,
                 col_id, file_level_processing=None, post_processing=None,
                 shuffle_files=False, cache_schema=True, shuffle_seed=42):
        self.data_path = data_path
        self.dt_parts = dt_parts
        self.hash_parts = hash_parts
        self.col_id = col_id
        self.file_level_processing = file_level_processing
        self.post_processing = post_processing
        self.shuffle_files = shuffle_files
        self.cache_schema = cache_schema
        self.shuffle_seed = shuffle_seed
        self.rs = None

        self._worker_id = None
        self._num_workers = None
        self._shuffle_seed = None
        self._schema = None

    def _get_my_hashes(self):
        files = [(i, name) for i, name in enumerate(self.hash_parts)]
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

        my_hashes = self._get_my_hashes()

        if self.shuffle_files:
            rs = np.random.RandomState(self._shuffle_seed % 2**32)
            rs.shuffle(my_hashes)

        logger.debug(f'Iter [{self._worker_id:02d}/{self._num_workers:02d}]: {my_hashes}')
        gen = chain(*[self.iter_hash(name) for name in my_hashes])
        if self.post_processing is not None:
            gen = self.post_processing(gen)
        return gen

    def iter_hash(self, hash_part_name):
        """Read all clients for all dt_parts from hash and join dt_parts into a single record

        :param hash_part_name:
        :return:
        """
        logger.debug(f'[{self._worker_id:2d}/{self._num_workers}] Start iter hash {hash_part_name}')
        client_partitioned_data = defaultdict(list)
        for dt_part in self.dt_parts:
            path = os.path.join(self.data_path, dt_part, hash_part_name)
            gen = self.iter_file(path)
            if self.file_level_processing is not None:
                gen = self.file_level_processing(gen)
            for features in gen:
                _id = features[self.col_id]
                cl_feature_list = client_partitioned_data[_id]
                cl_feature_list.append(features)

        for cli_id, features in client_partitioned_data.items():
            yield self.join_features(features)

    def iter_file(self, file_name):
        """

        :param file_name:
        :return: [(customer_id, features)]
        """
        logger.debug(f'[{self._worker_id}/{self._num_workers}] Iter file "{file_name}"')
        for rec in read_pyarrow_file(file_name, use_threads=True):
            yield rec

    def join_features(self, features):
        schema = self.get_schema(features)

        joined_record = {k: self.join_key(k, dtype, [v[k] for v in features]) for k, dtype in schema.items()}
        return joined_record

    def get_schema(self, rows):
        if self._schema is not None and self.cache_schema:
            return self._schema

        schema = set((k, type(v)) for row in rows for k, v in row.items())
        duplicated_keys = [k for k, v in Counter(k for k, v in schema).items() if v > 1]
        if len(duplicated_keys) > 0:
            raise AttributeError(f'Found duplicated_keys {duplicated_keys} '
                                 f'with types {[(k, v) for k, v in schema if k in duplicated_keys]}')
        self._schema = {k: v for k, v in schema}
        return self._schema

    def join_key(self, key, dtype, values):
        if dtype is np.ndarray:
            x = np.concatenate(values)
            if x.dtype.kind in ('i', 'f'):
                return torch.from_numpy(x)
            return x
        elif key == self.col_id:
            return values[0]
        else:
            return values
