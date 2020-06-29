from glob import glob
from itertools import chain

import torch
import logging

import numpy as np

logger = logging.getLogger(__name__)


class DataFiles:
    """
    Collect list of files based on `path_wc` and split it on train and valid (if possible)
    Returns list of file names for train and valid
    """
    def __init__(self, path_wc, valid_size=0.5, is_sort=True, seed=None):
        self.path_wc = path_wc
        self.valid_size = valid_size
        self.is_sort = is_sort
        self.seed = seed

        self._files = None
        self._n = None
        self._train_files = None
        self._valid_files = None

        self.load()
        self.split()
        self.sort()

    @property
    def train(self):
        return self._train_files

    @property
    def valid(self):
        return self._valid_files

    def load(self):
        self._files = glob(self.path_wc)
        self._n = len(self._files)
        _msg = f'Found {self._n} files in "{self.path_wc}"'
        if self._n == 0:
            raise AttributeError(_msg)
        logger.info(_msg)

    def size_select(self):
        if self.valid_size == 0.0:
            return self._n, 0

        _valid = self._n * self.valid_size
        if _valid < 1.0:
            _valid = 1
        else:
            _valid = int(round(_valid))

        _train = self._n - _valid
        if _train == 0:
            raise AttributeError(', '.join([
                'Incorrect train size',
                f'N: {self._n}',
                f'train: {_train}',
                f'valid: {_valid}',
                f'rate: {self.valid_size:.3f}',
            ]))
        return _train, _valid

    def split(self):
        train_size, valid_size = self.size_select()

        if valid_size == 0:
            self._train_files = self._files
            self._valid_files = None
        else:
            valid_ix = np.arange(self._n)
            rs = np.random.RandomState(self.seed)
            valid_ix = rs.choice(valid_ix, size=valid_size, replace=False)

            self._train_files = [rec for i, rec in enumerate(self._files) if i not in valid_ix]
            self._valid_files = [rec for i, rec in enumerate(self._files) if i in valid_ix]

    def sort(self):
        if not self.is_sort:
            return
        if self._train_files is not None:
            self._train_files = sorted(self._train_files)
        if self._valid_files is not None:
            self._valid_files = sorted(self._valid_files)


class LazyDataset(torch.utils.data.IterableDataset):
    def __init__(self, files, f_iter_file):
        self.files = files
        self.f_iter_file = f_iter_file

    def _get_my_files(self, num_workers, worker_id):
        files = [(i, name) for i, name in enumerate(self.files)]
        my_files = [name for i, name in files if i % num_workers == worker_id]
        return my_files

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            num_workers = 1
            worker_id = 0
        else:  # in a worker process
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        my_files = self._get_my_files(num_workers, worker_id)

        return chain(*[self.f_iter_file(name) for name in my_files])
