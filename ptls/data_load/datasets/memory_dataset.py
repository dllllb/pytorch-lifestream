import logging
from typing import Iterable, List

import joblib
import pandas as pd
import torch
from pymonad.either import Either
from joblib import parallel_backend, parallel_config, delayed, Parallel
from ptls.data_load import IterableChain
from ptls.data_load.iterable_processing.to_torch_tensor import ToTorch
from ptls.preprocessing.dask.dask_client import DaskServer

logger = logging.getLogger(__name__)


class MemoryMapDataset(torch.utils.data.Dataset):

    def __repr__(self):
        return 'In-Memory Dataset'

    def __init__(self, data, i_filters: List[Iterable] = None):
        self.dask_client = DaskServer().client
        self.processed_data = Either(data, monoid=[i_filters, i_filters is None]).either(
            left_function=lambda filters: self.__apply_filters(data, filters),
            right_function=lambda x: [rec for rec in x])
        logger.info(f'Loaded {len(self.processed_data)} records')
        self.dask_client.shutdown()

    def __apply_filters(self, data, i_filters):
        def _iterable_filtration(sample, i_filters):
            for f in i_filters:
                sample = f.transform(sample)
            return sample

        with joblib.parallel_backend(backend='dask'):
            parallel = Parallel(verbose=1)
            processed_data = parallel(delayed(_iterable_filtration)(row, i_filters)
                                      for row in data.itertuples())
        return pd.DataFrame(processed_data)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, item):
        return self.processed_data[item]


class MemoryIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, data, i_filters=None):
        raise NotImplementedError()
        """Multiprocessing case aren't defined
        """
        if i_filters is None:
            i_filters = []
        self.data = data
        self.post_processor_filter = IterableChain(ToTorch(), *i_filters)

    def __iter__(self):
        for rec in self.post_processor_filter(self.data):
            yield rec
