import logging
import warnings

import torch

from torch.utils.data import DataLoader
from typing import List, Dict

from ptls.data_load import IterableChain
from ptls.data_load.augmentations.build_augmentations import build_augmentations
from ptls.data_load.data_module.coles_data_module import coles_collate_fn
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing.feature_filter import FeatureFilter
from ptls.data_load.iterable_processing.filter_non_array import FilterNonArray
from ptls.data_load.iterable_processing.seq_len_filter import SeqLenFilter
from ptls.data_load.iterable_processing.to_torch_tensor import ToTorch
from ptls.frames.coles import split_strategy
from ptls.metric_learn.dataset.splitting_dataset import MapSplittingDataset

logger = logging.getLogger(__name__)


class PostProcessDataset(torch.utils.data.Dataset):
    def __init__(self, parent, post_processor):
        warnings.warn('Use `ptls.data_load.datasets.MemoryMapDataset`')
        super().__init__()
        self.parent = parent
        self.post_processor = post_processor

    def __len__(self):
        return len(self.parent)

    def __getitem__(self, idx):
        return next(self.post_processor([self.parent[idx]]))


r"""Generate a train data loader

Parameters
----------
data: List[Dict]
The dataset
min_seq_len: int. Default: 0.
The minimal length of sequences used for training. The shorter sequences would be skipped.
seq_split_strategy: str. Options: 'SampleSlices',  Default: 'SampleSlices'.
The strategy for splitting sequence to subsequences
split_count: int. Default: 5.
An amount of subsequences from one sequence.
split_cnt_min: int. Default: 1.
The minimal amount of events in each subsequence.
split_cnt_max: int. Default: 4000.
    The maximal amount of events in each subsequence.
num_workers: int. Default: 0.
The number of workers for the dataloader. 0 = single-process loader
train_batch_size: int. Default: 512.
The number of samples (before splitting to subsequences) in each batch
"""
def train_data_loader(
    data: List[Dict],
    min_seq_len: int = 0,
    seq_split_strategy: str = 'SampleSlices',
    split_count: int = 5,
    split_cnt_min: int = 1,
    split_cnt_max: int = 4000,
    drop_cols: List[str] = list(),
    num_workers: int = 0,
    batch_size: int = 512):
    warnings.warn('Use `train_dataloader()` method of `ptls.frames.PtlsDataModule` '
                  'with `ptls.frames.coles.ColesDataset` or `ptls.frames.coles.ColesIterableDataset`',
                  DeprecationWarning)

    dataset = MemoryMapDataset(
        data,
        i_filters=[
            SeqLenFilter(min_seq_len=min_seq_len),
            FilterNonArray(),
            ToTorch(),
            FeatureFilter(drop_feature_names=drop_cols),
        ],
    )

    split_strategy_params = {
        'split_strategy': seq_split_strategy,
        'split_count': split_count,
        'cnt_min': split_cnt_min,
        'cnt_max': split_cnt_max
    }

    dataset = MapSplittingDataset(
        base_dataset=dataset,
        splitter=split_strategy.create(**split_strategy_params),
        a_chain=build_augmentations({}),
    )

    return DataLoader(
        dataset=dataset,
        collate_fn=coles_collate_fn,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size
    )
