from glob import glob
from itertools import chain

import torch
import logging

logger = logging.getLogger(__name__)


class DataFiles:
    """
    Collect list of files based on `path_wc`
    Returns list of file names for future usage
    """
    def __init__(self, path_wc):
        self.path_wc = path_wc

        self._files = None
        self._n = None

        self.load()

    @property
    def files(self):
        return self._files

    def load(self):
        self._files = sorted(glob(self.path_wc))
        self._n = len(self._files)
        _msg = f'Found {self._n} files in "{self.path_wc}"'
        if self._n == 0:
            raise AttributeError(_msg)
        logger.info(_msg)


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
