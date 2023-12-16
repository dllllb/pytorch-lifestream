import logging
import os
import warnings
from glob import glob
from itertools import chain
from typing import Union, List

import numpy as np
import torch
from omegaconf import ListConfig

from ptls.data_load import read_pyarrow_file
from ptls.data_load import IterableChain

logger = logging.getLogger(__name__)


class State:
    def __init__(self):
        pass


class HMM:
    def __init__(self, states, hidden_states, state_transition_tensor, hidden_state_transition_matrix):
        self.states = states
        self.n_states = len(self.states)
        self.hidden_states = hidden_states
        self.n_hidden_states = len(self.hidden_states)
        self.state_transition_tensor = state_transition_tensor
        self.hidden_state_transition_matrix = hidden_state_transition_matrix

        assert self.state_transition_tensor.dim == 3
        assert self.hidden_state_transition_matrix.dim == 2
        assert self.state_transition_tensor.shape[0] == self.state_transition_tensor.shape[1] == self.n_states
        assert self.state_transition_tensor.shape[2] == self.hidden_state_transition_matrix.shape[0]
        assert self.hidden_state_transition_matrix.shape[0] == \
               self.hidden_state_transition_matrix.shape[1] == \
               self.n_hidden_states
        assert self.check_states_for_inds(self.states)
        assert self.check_states_for_inds(self.hidden_states)

        self.state = None
        self.h_state = None
        self.reset()

    def gen_next(self):
        return

    def reset(self):
        self.state = np.random.choice(self.states)
        self.h_state = np.random.choice(self.hidden_states)
        return self.state, self.h_state

    @staticmethod
    def check_states_for_inds(states):
        return sorted([s.ind for s in states]) == [i for i in range(len(states))]


class SyntheticDataset(torch.utils.data.IterableDataset):
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

    @staticmethod
    def to_torch(x):
        if type(x) is np.ndarray and x.dtype.kind in ('i', 'f'):
            return torch.from_numpy(x)
        return x
