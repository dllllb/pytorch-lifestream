import logging
from abc import ABC, abstractmethod
from typing import Iterable, List

import joblib
import torch
from joblib import delayed, Parallel
from pymonad.either import Either

from ptls.data_load import IterableChain

logger = logging.getLogger(__name__)


class MemoryMapDataset(torch.utils.data.Dataset):
    """
    Dataset for in-memory data processing. It is useful for small datasets that can be loaded into memory.

    Args:
        data: iterable data
        i_filters: list of iterable filters
        n_jobs: number of workers requested by the callers
    """
    def __repr__(self):
        return 'In-Memory Dataset'
    
    def __init__(self, 
                 data, 
                 i_filters: List[Iterable] = None, 
                 n_jobs: int = -1):
        self.n_jobs = n_jobs
        
        if i_filters is None:
            self.processed_data = [rec for rec in data]
        else:
            post_processor_filter = IterableChain(*i_filters)
            self.processed_data = [rec for rec in post_processor_filter(data)]
        
        logger.info(f'Loaded {len(self.processed_data)} records')

    def __apply_filters(self, data, i_filters):
        def _iterable_filtration(sample, i_filters):
            for f in i_filters:
                sample = f(sample)
            return sample

        with joblib.parallel_backend(backend='threading', n_jobs=self.n_jobs):
            parallel = Parallel(verbose=1)
            processed_data = parallel(delayed(_iterable_filtration)(row, i_filters)
                                      for row in data)
        return processed_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, item):
        return self.processed_data[item]


class MemoryIterableDataset(torch.utils.data.IterableDataset, ABC):
    def __init__(self, data, i_filters=None):
        super().__init__()
        self.data = data
        self.i_filters = i_filters
        self.processed_data = Either(data, monoid=[i_filters, i_filters is None]).either(
            left_function=lambda filters: IterableChain(*filters)(data),
            right_function=lambda x: [rec for rec in x])
    @abstractmethod
    def __iter__(self):
        """This method must be implemented in subclasses"""
        pass
